"""Tests for Cargo Mode: dock/load/unload/depart lifecycle, AAD mismatch
rejection, receipt verification, embedding spec binding, conditional
completion, and the BusTerminal surface.

Covers:
- EmbeddingSpec canonical form and fingerprint
- CargoManifest AAD construction and expiry
- Full lifecycle: dock -> load -> unload -> depart -> verify receipt
- AAD mismatch: wrong regime, wrong embedding spec, expired manifest
- Receipt HMAC verification (positive and negative)
- CompletionCondition firing (verified TTL only; external conditions fail closed)
- BusTerminal multi-bus management
- Session closed after depart
- Token reissue counting
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from schemen_gate import GateReleaseIdentity, current_release_identity
from schemen_gate._cargo import (
    CargoAuthenticationError,
    CargoExpiredError,
    CargoItem,
    CargoManifest,
    CargoManifestOperation,
    CargoSessionClosedError,
    CompletionCondition,
    CompletionKind,
    EmbeddingSpec,
    InferenceStreamFrame,
    OperationKind,
    VectorBridge,
    VectorPayload,
    compute_items_hash,
    compute_payload_hash,
    hash_vocabulary,
)
from schemen_gate._cargo_impl import (
    BusTerminal,
    DefaultRegimeBus,
    create_manifest,
    derive_cargo_access_key,
)
from schemen_gate._rag import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GATE_KEY = os.urandom(32)
N_DIMS = 64
N_REGIMES = 4
RNG = np.random.default_rng(42)
PARTITION_KEY = "regime-0"
SUBJECT_ID = "test-subject"
MODEL_DIGEST = "test-model-digest"
OPERATION = CargoManifestOperation.LOAD_AND_RETRIEVE.value
POLICY_VERSION = "test-policy-v1"

EMBED_SPEC = EmbeddingSpec(
    model_id="test-embed-v1",
    dimensions=N_DIMS,
    vocabulary_hash=hashlib.sha256(b"test-vocab").hexdigest(),
    pooling="cls",
)


def _make_adapter_and_bus():
    """Build a GatedRAGAdapter + DefaultRegimeBus for testing."""
    store = InMemoryVectorStore()
    pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
    pmap.register(PARTITION_KEY, regime_id=0, mode=PartitionMode.READ_WRITE)

    def embed_fn(content):
        if isinstance(content, np.ndarray):
            return content
        h = hashlib.sha256(content.encode()).digest()
        vec = np.frombuffer(h * (N_DIMS // 32 + 1), dtype=np.uint8)[:N_DIMS]
        return vec.astype(np.float64) / 255.0

    adapter = GatedRAGAdapter(store=store, partition_map=pmap, embed_fn=embed_fn)
    bus = DefaultRegimeBus(
        adapter=adapter,
        partition_key=PARTITION_KEY,
        regime_id=0,
        gate_key_secret=GATE_KEY,
        embedding_spec=EMBED_SPEC,
    )
    return adapter, bus


def _make_items(n: int = 3) -> list[CargoItem]:
    """Generate test cargo items."""
    items = []
    for i in range(n):
        h = hashlib.sha256(f"doc-{i}".encode()).digest()
        vec = np.frombuffer(h * (N_DIMS // 32 + 1), dtype=np.uint8)[:N_DIMS]
        items.append(
            CargoItem(
                content=f"Test document {i}",
                embedding=vec.astype(np.float64) / 255.0,
                kind="document",
                doc_id=f"doc-{i}",
            )
        )
    return items


def _make_manifest(
    regime_id: int = 0,
    embedding_spec: EmbeddingSpec | None = None,
    expires_epoch: int | None = None,
    item_count: int = 3,
    completion_conditions: tuple[CompletionCondition, ...] = (),
    max_reissues: int = 0,
    operation: CargoManifestOperation | str = CargoManifestOperation.LOAD_AND_RETRIEVE,
) -> CargoManifest:
    """Build a test manifest."""
    items = _make_items(item_count)
    return CargoManifest(
        manifest_id=CargoManifest.generate_id(),
        tenant_id="tenant-test",
        regime_id=regime_id,
        embedding_spec=embedding_spec or EMBED_SPEC,
        architecture="test-arch",
        payload_hash=compute_payload_hash(items),
        payload_kind="rag_documents",
        item_count=item_count,
        parent_lockbox_hash="",
        issued_at_epoch=int(time.time()),
        subject_id=SUBJECT_ID,
        model_digest=MODEL_DIGEST,
        operation=(operation.value if isinstance(operation, CargoManifestOperation) else operation),
        policy_version=POLICY_VERSION,
        partition_key=PARTITION_KEY,
        gate_embeddings_at_rest=False,
        expires_epoch=expires_epoch,
        completion_conditions=completion_conditions,
        max_reissues=max_reissues,
    )


# ---------------------------------------------------------------------------
# EmbeddingSpec
# ---------------------------------------------------------------------------


class TestEmbeddingSpec:
    def test_canonical_roundtrip(self):
        canonical = EMBED_SPEC.to_canonical()
        restored = EmbeddingSpec.from_canonical(canonical)
        assert restored == EMBED_SPEC

    def test_fingerprint_deterministic(self):
        assert EMBED_SPEC.fingerprint() == EMBED_SPEC.fingerprint()

    def test_different_spec_different_fingerprint(self):
        other = EmbeddingSpec(
            model_id="other-model",
            dimensions=1024,
            vocabulary_hash=hashlib.sha256(b"other-vocab").hexdigest(),
            pooling="mean",
        )
        assert EMBED_SPEC.fingerprint() != other.fingerprint()

    def test_canonical_contains_all_fields(self):
        canonical = EMBED_SPEC.to_canonical()
        assert EMBED_SPEC.model_id in canonical
        assert str(EMBED_SPEC.dimensions) in canonical
        assert EMBED_SPEC.vocabulary_hash in canonical
        assert EMBED_SPEC.pooling in canonical


# ---------------------------------------------------------------------------
# CargoManifest
# ---------------------------------------------------------------------------


class TestCargoManifest:
    def test_aad_deterministic(self):
        m = _make_manifest()
        assert m.to_aad() == m.to_aad()

    def test_aad_uses_release_bound_v6_schema(self):
        m = _make_manifest()
        assert b'"schema":"schemen/cargo-manifest-v7"' in m.to_aad()
        assert b'"gate_release"' in m.to_aad()

    def test_aad_contains_embedding_spec(self):
        m = _make_manifest()
        aad = m.to_aad().decode()
        assert EMBED_SPEC.model_id in aad
        assert str(EMBED_SPEC.dimensions) in aad

    def test_not_expired_when_no_ttl(self):
        m = _make_manifest(expires_epoch=None)
        assert not m.is_expired()

    def test_expired_when_past_epoch(self):
        m = _make_manifest(expires_epoch=int(time.time()) - 100)
        assert m.is_expired()

    def test_not_expired_when_future_epoch(self):
        m = _make_manifest(expires_epoch=int(time.time()) + 3600)
        assert not m.is_expired()

    @pytest.mark.parametrize(
        "now",
        [True, "99", float("nan"), float("inf"), float("-inf"), -1],
    )
    def test_invalid_verifier_clock_overrides_are_rejected(self, now):
        manifest = _make_manifest(expires_epoch=100)

        with pytest.raises(ValueError, match="finite non-negative"):
            manifest.is_expired(now)

    def test_generate_id_format(self):
        mid = CargoManifest.generate_id()
        assert mid.startswith("cargo-")
        assert len(mid) > 10

    def test_fingerprint_changes_with_regime(self):
        m1 = _make_manifest(regime_id=0)
        m2 = _make_manifest(regime_id=1)
        assert m1.fingerprint() != m2.fingerprint()

    def test_fingerprint_and_bus_validation_bind_gate_release(self):
        manifest = _make_manifest()
        current = current_release_identity()
        other = GateReleaseIdentity(
            package="schemen-gate",
            version="1.0.1",
            source_repository=current.source_repository,
            source_commit=current.source_commit,
        )
        changed = replace(manifest, gate_release=other)
        assert manifest.fingerprint() != changed.fingerprint()

        _, bus = _make_adapter_and_bus()
        access_key = derive_cargo_access_key(
            GATE_KEY,
            changed.regime_id,
            changed.tenant_id,
            subject_id=changed.subject_id,
            model_digest=changed.model_digest,
            operation=changed.operation,
            policy_version=changed.policy_version,
            partition_key=changed.partition_key,
        )
        with pytest.raises(CargoAuthenticationError, match="Gate release differs"):
            bus.dock(changed, access_key)


# ---------------------------------------------------------------------------
# Full lifecycle: dock -> load -> unload -> depart -> verify
# ---------------------------------------------------------------------------


class TestCargoLifecycle:
    def test_full_lifecycle(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest()
        items = _make_items(3)

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        assert session.regime_id == 0

        load_result = session.load_cargo(items)
        assert load_result.item_count == 3
        assert len(load_result.doc_ids) == 3

        unload_result = session.unload_cargo("Test document 0", top_k=2)
        assert unload_result.regime_id == 0
        assert unload_result.item_count <= 2

        receipt = session.depart()
        assert receipt.manifest_id == manifest.manifest_id
        assert receipt.tenant_id == "tenant-test"
        assert receipt.regime_id == 0
        assert receipt.embedding_spec == EMBED_SPEC
        assert len(receipt.operations) == 2
        assert receipt.operations[0].kind == OperationKind.LOAD
        assert receipt.operations[1].kind == OperationKind.UNLOAD
        assert receipt.docked_at_epoch <= receipt.departed_at_epoch
        assert receipt.receipt_id.startswith("rcpt-")
        assert len(receipt.gate_signature) == 32

    def test_receipt_verifies(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(item_count=2)
        items = _make_items(2)

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(items)
        receipt = session.depart()

        assert bus.verify_receipt(receipt, expected_manifest=manifest)

    def test_receipt_fails_with_wrong_key(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(item_count=1)
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(_make_items(1))
        receipt = session.depart()

        assert not receipt.verify(os.urandom(32))

    def test_load_only_session(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(item_count=2)
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(_make_items(2))
        receipt = session.depart()

        assert len(receipt.operations) == 1
        assert receipt.operations[0].kind == OperationKind.LOAD
        assert receipt.cargo_out_hash == compute_items_hash([])

    def test_unload_only_session(self):
        adapter, bus = _make_adapter_and_bus()
        adapter.ingest("seed doc", "seed doc", PARTITION_KEY, gate_embedding=False)

        manifest = _make_manifest(operation=CargoManifestOperation.RETRIEVE)
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.unload_cargo("seed doc", top_k=1)
        receipt = session.depart()

        assert len(receipt.operations) == 1
        assert receipt.operations[0].kind == OperationKind.UNLOAD
        assert receipt.cargo_in_hash == compute_items_hash([])


# ---------------------------------------------------------------------------
# AAD mismatch rejection
# ---------------------------------------------------------------------------


class TestAADMismatch:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("tenant_id", "other-tenant"),
            ("subject_id", "other-subject"),
            ("model_digest", "other-model"),
            ("operation", CargoManifestOperation.RETRIEVE.value),
            ("policy_version", "other-policy"),
        ],
    )
    def test_client_key_rejects_every_changed_scope_field(self, field, value):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest()
        access_key = derive_cargo_access_key(
            GATE_KEY,
            manifest.regime_id,
            manifest.tenant_id,
            subject_id=manifest.subject_id,
            model_digest=manifest.model_digest,
            operation=manifest.operation,
            policy_version=manifest.policy_version,
            partition_key=manifest.partition_key,
        )

        with pytest.raises(CargoAuthenticationError, match="AAD authentication"):
            bus.dock(replace(manifest, **{field: value}), access_key)

    def test_manifest_cannot_cross_an_alias_for_the_same_regime(self):
        adapter, _ = _make_adapter_and_bus()
        with pytest.raises(CargoAuthenticationError, match="not registered"):
            DefaultRegimeBus(
                adapter=adapter,
                partition_key="regime-0-alias",
                regime_id=0,
                gate_key_secret=GATE_KEY,
                embedding_spec=EMBED_SPEC,
            )

    def test_embedding_tamper_is_rejected_before_ingest(self):
        _, bus = _make_adapter_and_bus()
        items = _make_items(1)
        manifest = _make_manifest(item_count=1)
        items[0].embedding = items[0].embedding.copy()
        items[0].embedding[0] += 1.0

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        with pytest.raises(CargoAuthenticationError, match="payload hash"):
            session.load_cargo(items)

    def test_metadata_tamper_is_rejected_before_ingest(self):
        _, bus = _make_adapter_and_bus()
        items = _make_items(1)
        manifest = _make_manifest(item_count=1)
        items[0].metadata["authority"] = "expanded"

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        with pytest.raises(CargoAuthenticationError, match="payload hash"):
            session.load_cargo(items)

    def test_manifest_replay_is_rejected(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest()
        bus.dock(manifest, gate_key_secret=GATE_KEY)
        with pytest.raises(CargoAuthenticationError, match="already been docked"):
            bus.dock(manifest, gate_key_secret=GATE_KEY)

    def test_wrong_parent_gate_key_rejected(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest()

        with pytest.raises(CargoAuthenticationError, match="AAD authentication"):
            bus.dock(manifest, gate_key_secret=os.urandom(32))

    def test_wrong_regime_rejected(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(regime_id=2)

        with pytest.raises(CargoAuthenticationError, match="regime"):
            bus.dock(manifest, gate_key_secret=GATE_KEY)

    def test_wrong_embedding_spec_rejected(self):
        _, bus = _make_adapter_and_bus()
        wrong_spec = EmbeddingSpec(
            model_id="wrong-model",
            dimensions=1024,
            vocabulary_hash=hashlib.sha256(b"wrong").hexdigest(),
            pooling="mean",
        )
        manifest = _make_manifest(embedding_spec=wrong_spec)

        with pytest.raises(CargoAuthenticationError, match=r"[Ee]mbedding"):
            bus.dock(manifest, gate_key_secret=GATE_KEY)

    def test_expired_manifest_rejected(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(expires_epoch=int(time.time()) - 100)

        with pytest.raises(CargoExpiredError):
            bus.dock(manifest, gate_key_secret=GATE_KEY)


# ---------------------------------------------------------------------------
# Session lifecycle enforcement
# ---------------------------------------------------------------------------


class TestSessionEnforcement:
    def test_load_after_depart_raises(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(
            item_count=2,
            operation=CargoManifestOperation.RETRIEVE,
        )
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.depart()

        with pytest.raises(CargoSessionClosedError):
            session.load_cargo(_make_items(1))

    def test_unload_after_depart_raises(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(operation=CargoManifestOperation.RETRIEVE)
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.depart()

        with pytest.raises(CargoSessionClosedError):
            session.unload_cargo("query")

    def test_double_depart_raises(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(operation=CargoManifestOperation.RETRIEVE)
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.depart()

        with pytest.raises(CargoSessionClosedError):
            session.depart()

    def test_concurrent_loads_consume_a_manifest_once(self):
        _, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(item_count=1),
            gate_key_secret=GATE_KEY,
        )

        class BlockingAdapter:
            def __init__(self):
                self.partition_map = bus._adapter.partition_map
                self.calls = 0
                self.first_entered = threading.Event()
                self.release_first = threading.Event()
                self.lock = threading.Lock()

            def ingest_many(self, items, partition_key, *, gate_embedding):
                assert gate_embedding is False
                with self.lock:
                    self.calls += 1
                    call_number = self.calls
                if call_number == 1:
                    self.first_entered.set()
                    assert self.release_first.wait(timeout=2)
                return [metadata["doc_id"] for _, _, metadata in items]

        adapter = BlockingAdapter()
        session._adapter = adapter
        second_started = threading.Event()

        def load(second: bool):
            if second:
                second_started.set()
            return session.load_cargo(_make_items(1))

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(load, False)
            assert adapter.first_entered.wait(timeout=2)
            second = executor.submit(load, True)
            assert second_started.wait(timeout=2)
            time.sleep(0.05)
            adapter.release_first.set()
            outcomes = []
            for future in (first, second):
                try:
                    outcomes.append(future.result(timeout=2))
                except CargoAuthenticationError as exc:
                    outcomes.append(exc)

        assert sum(not isinstance(value, Exception) for value in outcomes) == 1
        assert sum(isinstance(value, CargoAuthenticationError) for value in outcomes) == 1
        assert adapter.calls == 1

    def test_failed_atomic_batch_does_not_create_receipt_state(self):
        _, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(item_count=2),
            gate_key_secret=GATE_KEY,
        )

        class FailingAtomicAdapter:
            def __init__(self):
                self.partition_map = bus._adapter.partition_map
                self.committed = []

            def ingest_many(self, items, partition_key, *, gate_embedding):
                assert gate_embedding is False
                raise RuntimeError("transaction rolled back")

        adapter = FailingAtomicAdapter()
        session._adapter = adapter
        with pytest.raises(RuntimeError, match="rolled back"):
            session.load_cargo(_make_items(2))

        assert adapter.committed == []
        with pytest.raises(CargoAuthenticationError, match="before the signed load"):
            session.depart()

    def test_concurrent_depart_emits_exactly_one_receipt(self):
        _, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(operation=CargoManifestOperation.RETRIEVE),
            gate_key_secret=GATE_KEY,
        )
        barrier = threading.Barrier(3)

        def depart():
            barrier.wait(timeout=2)
            return session.depart()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(depart) for _ in range(2)]
            barrier.wait(timeout=2)
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=2))
                except CargoSessionClosedError as exc:
                    outcomes.append(exc)

        assert sum(not isinstance(value, Exception) for value in outcomes) == 1
        assert sum(isinstance(value, CargoSessionClosedError) for value in outcomes) == 1

    def test_unload_completes_before_concurrent_depart_receipt(self):
        adapter, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(item_count=1),
            gate_key_secret=GATE_KEY,
        )
        session.load_cargo(_make_items(1))

        class BlockingQueryAdapter:
            def __init__(self):
                self.partition_map = adapter.partition_map
                self.entered = threading.Event()
                self.release = threading.Event()

            def query(self, *args, **kwargs):
                self.entered.set()
                assert self.release.wait(timeout=2)
                return adapter.query(*args, **kwargs)

        blocking = BlockingQueryAdapter()
        session._adapter = blocking
        with ThreadPoolExecutor(max_workers=2) as executor:
            unload = executor.submit(session.unload_cargo, "Test document 0", 1)
            assert blocking.entered.wait(timeout=2)
            depart = executor.submit(session.depart)
            time.sleep(0.05)
            assert not depart.done()
            blocking.release.set()
            unload.result(timeout=2)
            receipt = depart.result(timeout=2)

        assert [operation.kind for operation in receipt.operations] == [
            OperationKind.LOAD,
            OperationKind.UNLOAD,
        ]


# ---------------------------------------------------------------------------
# Completion conditions
# ---------------------------------------------------------------------------


class TestCompletionConditions:
    def test_items_exhausted_is_rejected_without_a_trusted_evidence_provider(self):
        _, bus = _make_adapter_and_bus()
        cond = CompletionCondition(
            kind=CompletionKind.ITEMS_EXHAUSTED,
            target=PARTITION_KEY,
        )
        manifest = _make_manifest(
            item_count=2,
            completion_conditions=(cond,),
        )
        with pytest.raises(ValueError, match="unsupported completion condition"):
            bus.dock(manifest, gate_key_secret=GATE_KEY)

    def test_ttl_expired_fires_on_depart(self):
        _, bus = _make_adapter_and_bus()
        cond = CompletionCondition(
            kind=CompletionKind.TTL_EXPIRED,
            target="",
        )
        manifest = _make_manifest(
            expires_epoch=int(time.time()) + 86400,
            completion_conditions=(cond,),
            operation=CargoManifestOperation.RETRIEVE,
        )
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session._manifest = CargoManifest(
            manifest_id=manifest.manifest_id,
            tenant_id=manifest.tenant_id,
            regime_id=manifest.regime_id,
            embedding_spec=manifest.embedding_spec,
            architecture=manifest.architecture,
            payload_hash=manifest.payload_hash,
            payload_kind=manifest.payload_kind,
            item_count=manifest.item_count,
            parent_lockbox_hash=manifest.parent_lockbox_hash,
            issued_at_epoch=manifest.issued_at_epoch,
            subject_id=manifest.subject_id,
            model_digest=manifest.model_digest,
            operation=manifest.operation,
            policy_version=manifest.policy_version,
            partition_key=manifest.partition_key,
            gate_embeddings_at_rest=manifest.gate_embeddings_at_rest,
            expires_epoch=int(time.time()) - 1,
            completion_conditions=manifest.completion_conditions,
            max_reissues=manifest.max_reissues,
        )
        receipt = session.depart()
        assert "ttl_expired" in receipt.completion_conditions_met

    def test_unverifiable_manual_condition_is_rejected_at_dock(self):
        _, bus = _make_adapter_and_bus()
        cond = CompletionCondition(
            kind=CompletionKind.PARTITION_DESTROYED,
            target="regime-0",
            destruction_aad="destroyed-by-test",
        )
        manifest = _make_manifest(completion_conditions=(cond,))
        with pytest.raises(ValueError, match="unsupported completion condition"):
            bus.dock(manifest, gate_key_secret=GATE_KEY)

    def test_external_condition_signal_never_enters_signed_receipt(self):
        _, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(operation=CargoManifestOperation.RETRIEVE),
            gate_key_secret=GATE_KEY,
        )
        with pytest.raises(CargoAuthenticationError, match="External completion"):
            session.signal_condition(
                CompletionKind.PARTITION_DESTROYED,
                "regime-0",
            )
        receipt = session.depart()
        assert "partition_destroyed" not in receipt.completion_conditions_met


# ---------------------------------------------------------------------------
# Token reissue tracking
# ---------------------------------------------------------------------------


class TestReissueTracking:
    def test_reissue_count_tracked(self):
        _, bus = _make_adapter_and_bus()
        manifest = _make_manifest(
            max_reissues=5,
            operation=CargoManifestOperation.RETRIEVE,
        )
        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.record_reissue()
        session.record_reissue()
        receipt = session.depart()
        assert receipt.reissue_count == 2

    def test_reissue_limit_is_enforced(self):
        _, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(max_reissues=1),
            gate_key_secret=GATE_KEY,
        )
        session.record_reissue()
        with pytest.raises(CargoAuthenticationError, match="exhausted"):
            session.record_reissue()

    def test_concurrent_reissues_cannot_exceed_the_limit(self):
        _, bus = _make_adapter_and_bus()
        session = bus.dock(
            _make_manifest(
                max_reissues=1,
                operation=CargoManifestOperation.RETRIEVE,
            ),
            gate_key_secret=GATE_KEY,
        )
        barrier = threading.Barrier(3)

        def reissue():
            barrier.wait(timeout=2)
            session.record_reissue()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(reissue) for _ in range(2)]
            barrier.wait(timeout=2)
            outcomes = []
            for future in futures:
                try:
                    future.result(timeout=2)
                    outcomes.append("ok")
                except CargoAuthenticationError as exc:
                    outcomes.append(exc)

        assert outcomes.count("ok") == 1
        assert sum(isinstance(value, CargoAuthenticationError) for value in outcomes) == 1
        assert session.depart().reissue_count == 1


# ---------------------------------------------------------------------------
# BusTerminal
# ---------------------------------------------------------------------------


class TestBusTerminal:
    def test_register_and_get_bus(self):
        store = InMemoryVectorStore()
        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        pmap.register("r0", regime_id=0)
        pmap.register("r1", regime_id=1)

        adapter = GatedRAGAdapter(store=store, partition_map=pmap)
        terminal = BusTerminal(
            adapter=adapter,
            gate_key_secret=GATE_KEY,
            embedding_spec=EMBED_SPEC,
        )
        terminal.register_bus("r0", 0)
        terminal.register_bus("r1", 1)

        assert set(terminal.list_buses()) == {"r0", "r1"}
        assert terminal.get_bus("r0").regime_id == 0
        assert terminal.get_bus("r1").regime_id == 1

    def test_get_missing_bus_raises(self):
        store = InMemoryVectorStore()
        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        adapter = GatedRAGAdapter(store=store, partition_map=pmap)
        terminal = BusTerminal(
            adapter=adapter,
            gate_key_secret=GATE_KEY,
            embedding_spec=EMBED_SPEC,
        )

        with pytest.raises(KeyError):
            terminal.get_bus("nonexistent")

    def test_terminal_embedding_spec(self):
        store = InMemoryVectorStore()
        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        adapter = GatedRAGAdapter(store=store, partition_map=pmap)
        terminal = BusTerminal(
            adapter=adapter,
            gate_key_secret=GATE_KEY,
            embedding_spec=EMBED_SPEC,
        )
        assert terminal.embedding_spec == EMBED_SPEC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestVectorBridge:
    def test_identity_bridge(self):
        bridge = VectorBridge(64, 64)
        assert bridge.is_identity
        vec = np.random.default_rng(0).standard_normal(64)
        result = bridge.project(vec)
        np.testing.assert_array_equal(result, vec)

    def test_square_custom_projection_is_not_ignored(self):
        projection = np.eye(4) * 2.0
        bridge = VectorBridge(4, 4, projection=projection)
        assert not bridge.is_identity
        np.testing.assert_array_equal(bridge.project(np.ones(4)), np.full(4, 2.0))

    def test_bridge_rejects_wrong_or_nonfinite_vectors(self):
        bridge = VectorBridge(4, 4)
        with pytest.raises(ValueError, match="dimension"):
            bridge.project(np.ones(3))
        with pytest.raises(ValueError, match="finite"):
            bridge.project(np.array([1.0, 2.0, 3.0, np.nan]))

    def test_step_up(self):
        bridge = VectorBridge(64, 128)
        assert not bridge.is_identity
        vec = np.random.default_rng(0).standard_normal(64)
        result = bridge.project(vec)
        assert result.shape == (128,)

    def test_step_down(self):
        bridge = VectorBridge(128, 64)
        vec = np.random.default_rng(0).standard_normal(128)
        result = bridge.project(vec)
        assert result.shape == (64,)

    def test_batch_projection(self):
        bridge = VectorBridge(64, 128)
        vecs = np.random.default_rng(0).standard_normal((5, 64))
        result = bridge.project(vecs)
        assert result.shape == (5, 128)

    def test_projection_hash_deterministic(self):
        b1 = VectorBridge(64, 128)
        b2 = VectorBridge(64, 128)
        assert b1.projection_hash() == b2.projection_hash()

    def test_custom_projection(self):
        proj = np.eye(64, 128)
        bridge = VectorBridge(64, 128, projection=proj)
        vec = np.zeros(64)
        vec[0] = 1.0
        result = bridge.project(vec)
        assert result[0] == 1.0
        assert result.shape == (128,)

    def test_wrong_shape_rejected(self):
        with pytest.raises(ValueError, match="shape"):
            VectorBridge(64, 128, projection=np.eye(32, 64))


class TestVectorPayload:
    def test_unload_vectors(self):
        _adapter, bus = _make_adapter_and_bus()
        items = _make_items(3)
        manifest = _make_manifest()

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(items)

        from schemen_gate._cargo import VectorPayload

        payload = session.unload_vectors("Test document 0", top_k=2)
        assert isinstance(payload, VectorPayload)
        assert payload.regime_id == 0
        assert payload.dim == N_DIMS
        assert payload.count <= 2
        assert len(payload.source_doc_ids) == payload.count

        receipt = session.depart()
        assert len(receipt.operations) == 2

    def test_unload_vectors_with_bridge(self):
        _adapter, bus = _make_adapter_and_bus()
        items = _make_items(3)
        manifest = _make_manifest()

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(items)

        bridge = VectorBridge(N_DIMS, 128)
        payload = session.unload_vectors("Test document 0", top_k=2, bridge=bridge)
        assert payload.dim == 128
        assert payload.bridge is bridge

        session.depart()

    def test_gated_vectors(self):
        _adapter, bus = _make_adapter_and_bus()
        items = _make_items(2)
        manifest = _make_manifest(item_count=2)

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(items)
        payload = session.unload_vectors("Test document 0", top_k=1)

        gated = payload.gated()
        assert gated.shape[-1] == N_DIMS
        mask = payload.gate_mask.to_numpy()
        zero_dims = np.where(mask == 0)[0]
        if len(zero_dims) > 0 and gated.ndim >= 1:
            for vec in gated if gated.ndim == 2 else [gated]:
                np.testing.assert_array_equal(vec[zero_dims], 0.0)

        session.depart()


class TestHelpers:
    def test_compute_items_hash_deterministic(self):
        ids = ["a", "b", "c"]
        assert compute_items_hash(ids) == compute_items_hash(ids)

    def test_compute_items_hash_order_sensitive(self):
        assert compute_items_hash(["a", "b"]) != compute_items_hash(["b", "a"])

    def test_compute_items_hash_has_unambiguous_boundaries(self):
        assert compute_items_hash(["ab", "c"]) != compute_items_hash(["a", "bc"])

    def test_compute_payload_hash(self):
        items = _make_items(3)
        h = compute_payload_hash(items)
        assert len(h) == 64
        assert h == compute_payload_hash(items)

    def test_hash_vocabulary(self):
        vocab = ["hello", "world", "test"]
        h = hash_vocabulary(vocab)
        assert len(h) == 64
        assert h == hash_vocabulary(["world", "hello", "test"])

    def test_create_manifest_factory(self):
        m = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=hashlib.sha256(b"payload").hexdigest(),
            payload_kind="rag_documents",
            item_count=5,
            subject_id=SUBJECT_ID,
            model_digest=MODEL_DIGEST,
            operation=OPERATION,
            policy_version=POLICY_VERSION,
            partition_key=PARTITION_KEY,
            gate_embeddings_at_rest=False,
            ttl_seconds=3600,
            max_reissues=3,
        )
        assert m.tenant_id == "t1"
        assert m.regime_id == 0
        assert m.embedding_spec == EMBED_SPEC
        assert m.expires_epoch is not None
        assert m.expires_epoch > int(time.time())
        assert m.max_reissues == 3
        assert m.manifest_id.startswith("cargo-")


# ===========================================================================
# InferenceStreamFrame — wire protocol for GPU backbone transport
# ===========================================================================


class TestInferenceStreamFrame:
    """Wire serialization/deserialization for vector transport."""

    def test_round_trip(self):
        """Serialize to wire, deserialize back, vectors match."""
        vecs = RNG.standard_normal((5, N_DIMS)).astype(np.float32)
        frame = InferenceStreamFrame(
            vectors=vecs,
            regime_id=0,
            source_doc_ids=[f"doc-{i}" for i in range(5)],
            embedding_spec=EMBED_SPEC,
            bridge_hash="a" * 64,
            sequence_id="stream-1",
            frame_index=0,
            is_final=True,
        )

        wire = frame.to_wire()
        restored = InferenceStreamFrame.from_wire(wire)

        np.testing.assert_array_almost_equal(
            restored.vectors,
            vecs,
            decimal=5,
        )
        assert restored.regime_id == 0
        assert restored.source_doc_ids == [f"doc-{i}" for i in range(5)]
        assert restored.embedding_spec == EMBED_SPEC
        assert restored.bridge_hash == "a" * 64
        assert restored.sequence_id == "stream-1"
        assert restored.frame_index == 0
        assert restored.is_final is True

    def test_wire_format_is_compact(self):
        """Wire format is smaller than JSON-serialized text."""
        import json

        vecs = RNG.standard_normal((8, 768)).astype(np.float32)
        frame = InferenceStreamFrame(
            vectors=vecs,
            regime_id=0,
            source_doc_ids=[f"doc-{i}" for i in range(8)],
        )

        wire = frame.to_wire()
        wire_size = len(json.dumps(wire))

        text_payload = json.dumps({"vectors": vecs.tolist()})
        text_size = len(text_payload)

        assert wire_size < text_size, (
            f"Wire format ({wire_size}) should be smaller than text ({text_size})"
        )

    def test_single_vector_frame(self):
        """Single vector (1D) survives round-trip."""
        vec = RNG.standard_normal(N_DIMS).astype(np.float32)
        frame = InferenceStreamFrame(
            vectors=vec,
            regime_id=2,
            source_doc_ids=["single-doc"],
        )

        assert frame.count == 1
        assert frame.dim == N_DIMS

        wire = frame.to_wire()
        restored = InferenceStreamFrame.from_wire(wire)
        np.testing.assert_array_almost_equal(
            restored.vectors.ravel(),
            vec,
            decimal=5,
        )

    def test_payload_hash_deterministic(self):
        """Same vectors produce same hash."""
        vecs = RNG.standard_normal((3, N_DIMS)).astype(np.float32)
        f1 = InferenceStreamFrame(vectors=vecs, regime_id=0, source_doc_ids=["a"])
        f2 = InferenceStreamFrame(vectors=vecs.copy(), regime_id=0, source_doc_ids=["b"])
        assert f1.payload_hash() == f2.payload_hash()

    def test_payload_hash_changes_with_vectors(self):
        """Different vectors produce different hash."""
        v1 = RNG.standard_normal((3, N_DIMS)).astype(np.float32)
        v2 = RNG.standard_normal((3, N_DIMS)).astype(np.float32)
        f1 = InferenceStreamFrame(vectors=v1, regime_id=0, source_doc_ids=["a"])
        f2 = InferenceStreamFrame(vectors=v2, regime_id=0, source_doc_ids=["a"])
        assert f1.payload_hash() != f2.payload_hash()

    def test_multi_frame_stream(self):
        """Multiple frames compose a stream with sequence tracking."""
        all_vecs = RNG.standard_normal((20, N_DIMS)).astype(np.float32)
        frames = []
        for i in range(4):
            chunk = all_vecs[i * 5 : (i + 1) * 5]
            frames.append(
                InferenceStreamFrame(
                    vectors=chunk,
                    regime_id=0,
                    source_doc_ids=[f"doc-{j}" for j in range(i * 5, (i + 1) * 5)],
                    sequence_id="stream-multi",
                    frame_index=i,
                    is_final=(i == 3),
                )
            )

        assert not frames[0].is_final
        assert frames[3].is_final
        assert all(f.sequence_id == "stream-multi" for f in frames)
        assert [f.frame_index for f in frames] == [0, 1, 2, 3]

        reassembled = np.concatenate([f.vectors for f in frames])
        np.testing.assert_array_equal(reassembled, all_vecs)

    def test_vector_payload_to_stream_frame(self):
        """VectorPayload.to_stream_frame() produces a valid frame."""
        from schemen_gate._rag import PartitionMap, PartitionMode

        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        pmap.register("r0", regime_id=0, mode=PartitionMode.READ_WRITE)
        mask = pmap.mask_for("r0")

        vecs = RNG.standard_normal((3, N_DIMS))
        payload = VectorPayload(
            vectors=vecs,
            gate_mask=mask,
            regime_id=0,
            source_doc_ids=["a", "b", "c"],
            embedding_spec=EMBED_SPEC,
        )

        frame = payload.to_stream_frame(gate=True)
        assert isinstance(frame, InferenceStreamFrame)
        assert frame.count == 3
        assert frame.regime_id == 0
        assert frame.embedding_spec == EMBED_SPEC

        mask_arr = mask.to_numpy()
        zero_dims = np.where(mask_arr == 0)[0]
        for row in frame.vectors:
            np.testing.assert_array_equal(row[zero_dims], 0.0)

    def test_no_embedding_spec_round_trip(self):
        """Frame without embedding_spec survives round-trip."""
        vecs = RNG.standard_normal((2, N_DIMS)).astype(np.float32)
        frame = InferenceStreamFrame(
            vectors=vecs,
            regime_id=1,
            source_doc_ids=["x"],
        )
        wire = frame.to_wire()
        restored = InferenceStreamFrame.from_wire(wire)
        assert restored.embedding_spec is None
        assert restored.bridge_hash is None

    def test_oversized_base64_is_rejected_before_decode(self, monkeypatch):
        import base64

        import schemen_gate._cargo as cargo_module

        monkeypatch.setattr(cargo_module, "_MAX_FRAME_BYTES", 8)
        decode_called = False

        def fail_if_called(*args, **kwargs):
            nonlocal decode_called
            decode_called = True
            raise AssertionError("decoder must not run for an oversized payload")

        monkeypatch.setattr(base64, "b64decode", fail_if_called)
        wire = {
            "v": 1,
            "vectors_b64": "A" * 16,
            "shape": [3],
            "dtype": "float32",
            "regime_id": 0,
            "source_doc_ids": ["a"],
            "frame_index": 0,
            "is_final": True,
        }

        with pytest.raises(ValueError, match="byte limit"):
            InferenceStreamFrame.from_wire(wire)
        assert not decode_called

    def test_base64_boundary_and_malformed_padding(self, monkeypatch):
        import base64

        import schemen_gate._cargo as cargo_module

        monkeypatch.setattr(cargo_module, "_MAX_FRAME_BYTES", 8)
        raw = np.array([1.0, 2.0], dtype=np.float32).tobytes()
        wire = {
            "v": 1,
            "vectors_b64": base64.b64encode(raw).decode("ascii"),
            "shape": [2],
            "dtype": "float32",
            "regime_id": 0,
            "source_doc_ids": ["a"],
            "frame_index": 0,
            "is_final": True,
        }
        np.testing.assert_array_equal(
            InferenceStreamFrame.from_wire(wire).vectors,
            np.array([1.0, 2.0], dtype=np.float32),
        )

        malformed = {**wire, "vectors_b64": "!!!!" + wire["vectors_b64"][4:]}
        with pytest.raises(ValueError, match="Invalid base64"):
            InferenceStreamFrame.from_wire(malformed)

    def test_declared_shape_overflow_is_rejected_before_decode(self, monkeypatch):
        import base64

        decode_called = False

        def fail_if_called(*args, **kwargs):
            nonlocal decode_called
            decode_called = True
            raise AssertionError("decoder must not run for overflowing shape")

        monkeypatch.setattr(base64, "b64decode", fail_if_called)
        wire = {
            "v": 1,
            "vectors_b64": "",
            "shape": [2**63, 2**63],
            "dtype": "float32",
            "regime_id": 0,
            "source_doc_ids": [],
            "frame_index": 0,
            "is_final": True,
        }
        with pytest.raises(ValueError, match="byte limit"):
            InferenceStreamFrame.from_wire(wire)
        assert not decode_called
