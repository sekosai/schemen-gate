"""Data-boundary validation and fail-closed behavior regressions."""

from __future__ import annotations

import base64
import concurrent.futures
import hashlib
import hmac
import struct
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from schemen_gate import GateMask
from schemen_gate._cargo import (
    CargoAuthenticationError,
    CargoItem,
    CargoManifestOperation,
    CargoOperation,
    CompletionCondition,
    CompletionKind,
    EmbeddingSpec,
    InferenceStreamFrame,
    OperationKind,
    VectorBridge,
    VectorPayload,
    compute_items_hash,
    compute_payload_hash,
)
from schemen_gate._cargo_impl import (
    BusTerminal,
    DefaultRegimeBus,
    _derive_cargo_receipt_key,
    create_manifest,
)
from schemen_gate._rag import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
    PartitionModeError,
    RetrievedDoc,
)
from schemen_gate._rag_stores import PgVectorStore
from schemen_gate._von import VONFrame, encode

GATE_KEY = b"f" * 32
PARTITION_KEY = "data-boundary"
SPEC = EmbeddingSpec(
    model_id="data-boundary-model",
    dimensions=4,
    vocabulary_hash=hashlib.sha256(b"data-boundary-vocabulary").hexdigest(),
    pooling="mean",
)


class _CountingStore(InMemoryVectorStore):
    def __init__(self) -> None:
        super().__init__()
        self.insert_many_calls = 0
        self.retrieve_calls = 0

    def insert_many(self, items, partition_key):
        self.insert_many_calls += 1
        return super().insert_many(items, partition_key)

    def retrieve(self, *args, **kwargs):
        self.retrieve_calls += 1
        return super().retrieve(*args, **kwargs)


class _RewritingStore(_CountingStore):
    def insert_many(self, items, partition_key):
        rewritten = [
            (
                embedding,
                content,
                {**(metadata or {}), "doc_id": "attacker-assigned"},
            )
            for embedding, content, metadata in items
        ]
        return super().insert_many(rewritten, partition_key)


def _fixture() -> tuple[_CountingStore, PartitionMap, GatedRAGAdapter, DefaultRegimeBus]:
    store = _CountingStore()
    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=2)
    partition_map.register(PARTITION_KEY, regime_id=0, mode=PartitionMode.READ_WRITE)
    adapter = GatedRAGAdapter(
        store,
        partition_map,
        embed_fn=lambda value: np.asarray(value, dtype=np.float64),
    )
    bus = DefaultRegimeBus(
        adapter,
        PARTITION_KEY,
        0,
        GATE_KEY,
        SPEC,
    )
    return store, partition_map, adapter, bus


def _item(
    *,
    embedding: np.ndarray | None = None,
    kind: str = "document",
    metadata: dict | None = None,
) -> CargoItem:
    return CargoItem(
        content="data-boundary cargo",
        embedding=(np.array([1.0, 0.0, 0.0, 0.0]) if embedding is None else embedding),
        kind=kind,
        metadata={} if metadata is None else metadata,
        doc_id="data-boundary-item",
    )


def _manifest(
    operation: CargoManifestOperation,
    items: tuple[CargoItem, ...],
):
    return create_manifest(
        tenant_id="data-boundary-tenant",
        regime_id=0,
        embedding_spec=SPEC,
        payload_hash=compute_payload_hash(items),
        payload_kind="rag_documents",
        item_count=len(items),
        subject_id="data-boundary-subject",
        model_digest="data-boundary-model-digest",
        operation=operation.value,
        policy_version="data-boundary-v1",
        partition_key=PARTITION_KEY,
        gate_embeddings_at_rest=False,
    )


def test_public_mask_has_no_authority_bearing_base() -> None:
    gate = GateMask.from_indices([0], n_dims=4, regime_id=0)

    public = gate.mask
    assert public.base is None
    public.setflags(write=True)
    public[:] = 1.0

    np.testing.assert_array_equal(
        gate.apply(np.ones(4)),
        np.array([1.0, 0.0, 0.0, 0.0]),
    )


def test_rag_ingest_requires_explicit_storage_gate_policy() -> None:
    _, _, adapter, _ = _fixture()
    with pytest.raises(ValueError, match="gate_embedding must be explicitly"):
        adapter.ingest(
            np.array([1.0, 0.0, 0.0, 0.0]),
            "document",
            PARTITION_KEY,
        )


def test_mask_metadata_is_detached_at_ingress_and_egress() -> None:
    original = {"nested": {"labels": ["original"]}}
    gate = GateMask(
        _mask=np.array([1.0, 0.0]),
        _regime_id=0,
        _metadata=original,
    )

    original["nested"]["labels"][0] = "ingress-mutated"
    public = gate.metadata
    public["nested"]["labels"][0] = "egress-mutated"

    assert gate.metadata["nested"]["labels"] == ["original"]
    assert gate.to_dict()["nested"]["labels"] == ["original"]


def test_bus_registration_rejects_partition_regime_mismatch() -> None:
    store = InMemoryVectorStore()
    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=2)
    partition_map.register(PARTITION_KEY, regime_id=0, mode=PartitionMode.READ_WRITE)
    adapter = GatedRAGAdapter(store, partition_map)
    terminal = BusTerminal(adapter, GATE_KEY, SPEC)

    with pytest.raises(CargoAuthenticationError, match="does not match"):
        terminal.register_bus(PARTITION_KEY, regime_id=1)


def test_session_rechecks_partition_binding_before_crossing() -> None:
    _, partition_map, _, bus = _fixture()
    manifest = _manifest(CargoManifestOperation.RETRIEVE, ())
    session = bus.dock(manifest, GATE_KEY)
    partition_map._entries[PARTITION_KEY].regime_id = 1

    with pytest.raises(CargoAuthenticationError, match="maps to regime"):
        session.unload_cargo(np.array([1.0, 0.0, 0.0, 0.0]))
    with pytest.raises(CargoAuthenticationError, match="maps to regime"):
        session.depart()


def test_store_detaches_embedding_and_nested_metadata_at_both_boundaries() -> None:
    store = InMemoryVectorStore()
    embedding = np.array([1.0, 0.0, 0.0, 0.0])
    metadata = {"nested": {"labels": ["original"]}}
    store.insert(
        embedding,
        "document",
        PARTITION_KEY,
        {"doc_id": "doc", "kind": "document", **metadata},
    )

    embedding[:] = 0.0
    metadata["nested"]["labels"][0] = "caller-mutated"
    first = store.retrieve(
        np.array([1.0, 0.0, 0.0, 0.0]),
        PARTITION_KEY,
    )[0]
    np.testing.assert_array_equal(first.embedding, [1.0, 0.0, 0.0, 0.0])
    assert first.metadata["nested"]["labels"] == ["original"]

    first.embedding[:] = 9.0
    first.metadata["nested"]["labels"][0] = "result-mutated"
    second = store.retrieve(
        np.array([1.0, 0.0, 0.0, 0.0]),
        PARTITION_KEY,
    )[0]
    np.testing.assert_array_equal(second.embedding, [1.0, 0.0, 0.0, 0.0])
    assert second.metadata["nested"]["labels"] == ["original"]


def test_caller_mutation_after_load_cannot_change_receipted_storage() -> None:
    store, _, _, bus = _fixture()
    embedding = np.array([1.0, 0.0, 0.0, 0.0])
    nested = {"nested": {"labels": ["original"]}}
    items = (_item(embedding=embedding, metadata=nested),)
    manifest = _manifest(CargoManifestOperation.LOAD, items)
    session = bus.dock(manifest, GATE_KEY)
    session.load_cargo(items)
    receipt = session.depart()

    embedding[:] = 0.0
    nested["nested"]["labels"][0] = "mutated"
    stored = store.retrieve(
        np.array([1.0, 0.0, 0.0, 0.0]),
        PARTITION_KEY,
    )[0]

    np.testing.assert_array_equal(stored.embedding, [1.0, 0.0, 0.0, 0.0])
    assert stored.metadata["nested"]["labels"] == ["original"]
    assert bus.verify_receipt(receipt, expected_manifest=manifest)


def test_load_manifest_cannot_depart_before_declared_load() -> None:
    _, _, _, bus = _fixture()
    items = (_item(),)
    manifest = _manifest(CargoManifestOperation.LOAD, items)
    session = bus.dock(manifest, GATE_KEY)

    with pytest.raises(CargoAuthenticationError, match="before the signed load"):
        session.depart()

    session.load_cargo(items)
    assert bus.verify_receipt(session.depart(), expected_manifest=manifest)


def test_verifier_rejects_correctly_signed_incomplete_load_receipt() -> None:
    _, _, _, bus = _fixture()
    items = (_item(),)
    manifest = _manifest(CargoManifestOperation.LOAD, items)
    session = bus.dock(manifest, GATE_KEY)
    session.load_cargo(items)
    receipt = session.depart()
    incomplete = replace(
        receipt,
        operations=(),
        cargo_in_hash=compute_items_hash([]),
        gate_signature=b"",
    )
    receipt_key = _derive_cargo_receipt_key(
        GATE_KEY,
        manifest.regime_id,
        manifest.tenant_id,
        subject_id=manifest.subject_id,
        model_digest=manifest.model_digest,
        operation=manifest.operation,
        policy_version=manifest.policy_version,
        partition_key=manifest.partition_key,
    )
    incomplete = replace(
        incomplete,
        gate_signature=hmac.new(
            receipt_key,
            incomplete.body_bytes(),
            hashlib.sha256,
        ).digest(),
    )

    assert incomplete.verify(receipt_key)
    assert not bus.verify_receipt(incomplete, expected_manifest=manifest)


@pytest.mark.parametrize(
    "item",
    [
        _item(embedding=np.arange(7, dtype=np.float64)),
        _item(kind="adapter_weight"),
    ],
)
def test_cargo_rejects_dimension_or_kind_mismatch_before_store(item: CargoItem) -> None:
    store, _, _, bus = _fixture()
    items = (item,)
    manifest = _manifest(CargoManifestOperation.LOAD, items)
    session = bus.dock(manifest, GATE_KEY)

    with pytest.raises(CargoAuthenticationError):
        session.load_cargo(items)
    assert store.insert_many_calls == 0


def test_cargo_rejects_duplicate_explicit_ids_before_store() -> None:
    store, _, _, bus = _fixture()
    items = (_item(), _item(embedding=np.array([0.0, 1.0, 0.0, 0.0])))
    manifest = _manifest(CargoManifestOperation.LOAD, items)
    session = bus.dock(manifest, GATE_KEY)

    with pytest.raises(CargoAuthenticationError, match="doc_ids must be unique"):
        session.load_cargo(items)
    assert store.insert_many_calls == 0


def test_cargo_rejects_store_replacement_of_signed_document_id() -> None:
    store = _RewritingStore()
    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=2)
    partition_map.register(PARTITION_KEY, regime_id=0, mode=PartitionMode.READ_WRITE)
    adapter = GatedRAGAdapter(
        store,
        partition_map,
        embed_fn=lambda value: np.asarray(value, dtype=np.float64),
    )
    bus = DefaultRegimeBus(adapter, PARTITION_KEY, 0, GATE_KEY, SPEC)
    items = (_item(),)
    manifest = _manifest(CargoManifestOperation.LOAD, items)
    session = bus.dock(manifest, GATE_KEY)

    with pytest.raises(CargoAuthenticationError, match="explicitly signed"):
        session.load_cargo(items)
    with pytest.raises(CargoAuthenticationError, match="before the signed load"):
        session.depart()
    assert store.retrieve(np.ones(4), PARTITION_KEY)[0].doc_id == "attacker-assigned"


def test_postgres_connection_serializes_transactions_and_queries() -> None:
    pytest.importorskip("psycopg")
    transaction_entered = threading.Event()
    release_transaction = threading.Event()
    count_attempted = threading.Event()
    count_executed = threading.Event()

    class Result:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class Transaction:
        def __enter__(self):
            transaction_entered.set()
            assert release_transaction.wait(timeout=5)
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeConnection:
        closed = False

        def transaction(self):
            return Transaction()

        def execute(self, statement, parameters=None):
            if parameters and "id" in parameters:
                return Result((parameters["id"],))
            count_executed.set()
            return Result((0,))

    store = PgVectorStore("postgresql://unused", dim=4)
    store._conn = FakeConnection()
    items = [(np.ones(4), "doc", {"doc_id": "signed-id"})]

    def count() -> int:
        count_attempted.set()
        return store.count(PARTITION_KEY)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        insert_future = pool.submit(store.insert_many, items, PARTITION_KEY)
        assert transaction_entered.wait(timeout=5)
        count_future = pool.submit(count)
        assert count_attempted.wait(timeout=5)
        assert not count_executed.is_set()
        release_transaction.set()
        assert insert_future.result(timeout=5) == ["signed-id"]
        assert count_future.result(timeout=5) == 0
    assert count_executed.is_set()


def test_postgres_table_setup_requires_server_pgvector_extension() -> None:
    pytest.importorskip("psycopg")

    class Result:
        def fetchone(self):
            return (None,)

    class FakeConnection:
        closed = False

        def execute(self, statement, parameters=None):
            assert statement == "SELECT to_regtype('vector')"
            return Result()

    store = PgVectorStore("postgresql://unused", dim=4)
    store._conn = FakeConnection()
    with pytest.raises(RuntimeError, match="pgvector extension is required"):
        store.ensure_table()


def test_cargo_hash_rejects_non_vector_and_non_json_metadata() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        compute_payload_hash((_item(embedding=np.ones((2, 2))),))
    with pytest.raises(ValueError, match="non-JSON type"):
        compute_payload_hash((_item(metadata={"ambiguous": (1, 2)}),))
    with pytest.raises(ValueError, match="object key"):
        compute_payload_hash((_item(metadata={1: "ambiguous"}),))
    cyclic: dict = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="contains a cycle"):
        compute_payload_hash((_item(metadata=cyclic),))


def test_embedding_spec_parser_rejects_ambiguous_or_extended_json() -> None:
    duplicate = (
        '{"dimensions":4,"dimensions":4,"model_id":"m","pooling":"mean",'
        '"schema":"schemen/embedding-spec-v2","vocabulary_hash":"' + "0" * 64 + '"}'
    )
    with pytest.raises(ValueError, match="Invalid EmbeddingSpec"):
        EmbeddingSpec.from_canonical(duplicate)

    extended = SPEC.to_canonical()[:-1] + ',"untrusted":true}'
    with pytest.raises(ValueError, match="Invalid EmbeddingSpec"):
        EmbeddingSpec.from_canonical(extended)


def test_protocol_text_rejects_unencodable_or_oversized_values() -> None:
    with pytest.raises(ValueError, match="bounded exact string"):
        EmbeddingSpec(
            model_id="\ud800",
            dimensions=4,
            vocabulary_hash="0" * 64,
            pooling="mean",
        )
    with pytest.raises(ValueError, match="bounded exact string"):
        EmbeddingSpec(
            model_id="x" * 16_385,
            dimensions=4,
            vocabulary_hash="0" * 64,
            pooling="mean",
        )


def test_completion_conditions_are_strict_and_ttl_requires_expiry() -> None:
    with pytest.raises(ValueError, match="missing or unknown"):
        CompletionCondition.from_dict({"kind": "ttl_expired", "target": "", "untrusted": True})

    _, _, _, bus = _fixture()
    manifest = create_manifest(
        tenant_id="data-boundary-tenant",
        regime_id=0,
        embedding_spec=SPEC,
        payload_hash=compute_payload_hash(()),
        payload_kind="rag_documents",
        item_count=0,
        subject_id="data-boundary-subject",
        model_digest="data-boundary-model-digest",
        operation=CargoManifestOperation.RETRIEVE.value,
        policy_version="data-boundary-v1",
        partition_key=PARTITION_KEY,
        gate_embeddings_at_rest=False,
        ttl_seconds=None,
        completion_conditions=(CompletionCondition(kind=CompletionKind.TTL_EXPIRED, target=""),),
    )
    with pytest.raises(ValueError, match="requires a manifest expiry"):
        bus.dock(manifest, GATE_KEY)
    with pytest.raises(ValueError, match="ttl_seconds"):
        create_manifest(
            tenant_id="data-boundary-tenant",
            regime_id=0,
            embedding_spec=SPEC,
            payload_hash=compute_payload_hash(()),
            payload_kind="rag_documents",
            item_count=0,
            subject_id="data-boundary-subject",
            model_digest="data-boundary-model-digest",
            operation=CargoManifestOperation.RETRIEVE.value,
            policy_version="data-boundary-v1",
            partition_key=PARTITION_KEY,
            gate_embeddings_at_rest=False,
            ttl_seconds=True,
        )


def test_receipt_verifier_rejects_wrong_runtime_types() -> None:
    _, _, _, bus = _fixture()
    manifest = _manifest(CargoManifestOperation.RETRIEVE, ())
    assert not bus.verify_receipt(None, expected_manifest=manifest)  # type: ignore[arg-type]
    assert not bus.verify_receipt(None, expected_manifest=None)  # type: ignore[arg-type]


def test_von_parser_rejects_dimension_mismatch_nonfinite_and_trailing_data() -> None:
    wire = encode(np.array([1.0, 2.0, 3.0, 4.0]), max_level=0).to_wire()

    wrong_dimensions = bytearray(wire)
    struct.pack_into("<H", wrong_dimensions, 5, 1)
    with pytest.raises(ValueError, match="expected 1"):
        VONFrame.from_wire(bytes(wrong_dimensions))

    nonfinite_scale = bytearray(wire)
    level_header_offset = 4 + struct.calcsize("<BHB")
    scale_offset = level_header_offset + struct.calcsize("<BBl")
    struct.pack_into("<d", nonfinite_scale, scale_offset, float("nan"))
    with pytest.raises(ValueError, match="must be finite"):
        VONFrame.from_wire(bytes(nonfinite_scale))

    with pytest.raises(ValueError, match="trailing bytes"):
        VONFrame.from_wire(wire + b"untrusted")
    with pytest.raises(ValueError, match="truncated"):
        VONFrame.from_wire(wire[:-1])


def test_von_parser_rejects_duplicate_or_out_of_order_levels() -> None:
    wire = bytearray(encode(np.array([1.0, 2.0, 3.0, 4.0]), max_level=1).to_wire())
    first_header = 4 + struct.calcsize("<BHB")
    second_header = first_header + struct.calcsize("<BBlddI") + 4
    wire[second_header] = 0

    with pytest.raises(ValueError, match="unique, ordered"):
        VONFrame.from_wire(bytes(wire))


def test_von_encoder_rejects_unrepresentable_finite_dynamic_range() -> None:
    with pytest.raises(ValueError, match="dynamic range"):
        encode(np.array([-1e308, 1e308]), max_level=3)


def test_cargo_operation_validates_on_its_actual_dataclass() -> None:
    with pytest.raises(ValueError, match="OperationKind"):
        CargoOperation(
            kind="load",  # type: ignore[arg-type]
            timestamp_epoch=1.0,
            item_count=1,
            items_hash="0" * 64,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        CargoOperation(
            kind=OperationKind.LOAD,
            timestamp_epoch=True,
            item_count=1,
            items_hash="0" * 64,
        )


def test_inference_frame_detaches_and_revalidates_mutable_state() -> None:
    source = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    frame = InferenceStreamFrame(
        vectors=source,
        regime_id=0,
        source_doc_ids=["doc"],
        embedding_spec=SPEC,
    )
    source[:] = 9.0
    assert frame.vectors.base is None
    np.testing.assert_array_equal(frame.vectors, [1.0, 2.0, 3.0, 4.0])

    frame.vectors = np.array([1.0, np.nan, 3.0, 4.0])
    with pytest.raises(ValueError, match="finite"):
        frame.to_wire()


def test_inference_frame_wire_rejects_unknown_fields_and_nonfinite_values() -> None:
    wire = InferenceStreamFrame(
        vectors=np.ones(4, dtype=np.float32),
        regime_id=0,
        source_doc_ids=["doc"],
        embedding_spec=SPEC,
    ).to_wire()
    with pytest.raises(ValueError, match="unknown fields"):
        InferenceStreamFrame.from_wire({**wire, "untrusted": True})

    for required_field in ("frame_index", "is_final"):
        missing = dict(wire)
        missing.pop(required_field)
        with pytest.raises(ValueError, match="missing or unknown fields"):
            InferenceStreamFrame.from_wire(missing)

    raw = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32).tobytes()
    nonfinite = {**wire, "vectors_b64": base64.b64encode(raw).decode("ascii")}
    with pytest.raises(ValueError, match="finite"):
        InferenceStreamFrame.from_wire(nonfinite)


def test_inference_frame_requires_unique_encodable_provenance() -> None:
    with pytest.raises(ValueError, match="frame metadata limit"):
        InferenceStreamFrame(
            vectors=np.ones((2, 4), dtype=np.float32),
            regime_id=0,
            source_doc_ids=["duplicate", "duplicate"],
            embedding_spec=SPEC,
        )
    with pytest.raises(ValueError, match="frame metadata limit"):
        InferenceStreamFrame(
            vectors=np.ones(4, dtype=np.float32),
            regime_id=0,
            source_doc_ids=["\ud800"],
            embedding_spec=SPEC,
        )


def test_adapter_rejects_wrong_dimensions_before_store_access() -> None:
    store, _, adapter, _ = _fixture()
    with pytest.raises(ValueError, match="expected 4"):
        adapter.query(np.ones(3), PARTITION_KEY)
    assert store.retrieve_calls == 0


def test_adapter_bounds_query_count_and_unencodable_text_before_store() -> None:
    store, _, adapter, _ = _fixture()
    with pytest.raises(ValueError, match="top_k"):
        adapter.query(np.ones(4), PARTITION_KEY, top_k=100_001)
    with pytest.raises(ValueError, match="bounded"):
        adapter.query(np.ones(4), PARTITION_KEY, kind="\ud800")
    assert store.retrieve_calls == 0


def test_adapter_rejects_malformed_store_results() -> None:
    class WrongDimensionStore(InMemoryVectorStore):
        def retrieve(self, *args, **kwargs):
            return [
                RetrievedDoc(
                    doc_id="wrong-dimension",
                    content="bad",
                    embedding=np.ones(5),
                    score=1.0,
                    partition_key=PARTITION_KEY,
                )
            ]

    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=2)
    partition_map.register(PARTITION_KEY, regime_id=0)
    adapter = GatedRAGAdapter(WrongDimensionStore(), partition_map)

    with pytest.raises(PartitionModeError, match="expected 4"):
        adapter.query(np.ones(4), PARTITION_KEY)


def test_adapter_requires_store_to_honor_requested_kind() -> None:
    class WrongKindStore(InMemoryVectorStore):
        def retrieve(self, *args, **kwargs):
            return [
                RetrievedDoc(
                    doc_id="wrong-kind",
                    content="bad",
                    embedding=np.ones(4),
                    score=1.0,
                    partition_key=PARTITION_KEY,
                    kind="adapter_weight",
                )
            ]

    partition_map = PartitionMap(GATE_KEY, n_dims=4, n_regimes=2)
    partition_map.register(PARTITION_KEY, regime_id=0)
    adapter = GatedRAGAdapter(WrongKindStore(), partition_map)

    with pytest.raises(PartitionModeError, match="requested kind"):
        adapter.query(np.ones(4), PARTITION_KEY, kind="document")


def test_vector_payload_gates_before_bridge_and_detaches_every_result() -> None:
    source = np.array([[1.0, 2.0, 3.0, 4.0]])
    mask = GateMask.from_indices([0, 2], n_dims=4, regime_id=0)
    projection = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    )
    bridge = VectorBridge(4, 2, projection=projection)
    payload = VectorPayload(
        vectors=source,
        gate_mask=mask,
        regime_id=0,
        source_doc_ids=["doc"],
        bridge=bridge,
        embedding_spec=SPEC,
    )

    source[:] = 99.0
    projection[:] = 99.0
    bridge._projection[:] = 99.0
    public_vectors = payload.vectors
    public_vectors.setflags(write=True)
    public_vectors[:] = 99.0
    public_ids = payload.source_doc_ids
    public_ids[0] = "mutated"

    np.testing.assert_array_equal(payload.vectors, [[12.0, 9.0]])
    np.testing.assert_array_equal(payload.gated(), [[4.0, 3.0]])
    assert payload.source_doc_ids == ["doc"]
    frame = payload.to_stream_frame(gate=True)
    np.testing.assert_array_equal(frame.vectors, [[4.0, 3.0]])
    assert (
        frame.bridge_hash
        == hashlib.sha256(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [2.0, 1.0],
                ],
                dtype=np.float64,
            ).tobytes()
        ).hexdigest()
    )


def test_vector_payload_requires_exact_vector_provenance() -> None:
    mask = GateMask.from_indices([0], n_dims=4, regime_id=0)
    with pytest.raises(ValueError, match="identify every vector"):
        VectorPayload(
            vectors=np.ones((2, 4)),
            gate_mask=mask,
            regime_id=0,
            source_doc_ids=["only-one"],
            embedding_spec=SPEC,
        )
    with pytest.raises(ValueError, match="payload metadata limit"):
        VectorPayload(
            vectors=np.ones((2, 4)),
            gate_mask=mask,
            regime_id=0,
            source_doc_ids=["duplicate", "duplicate"],
            embedding_spec=SPEC,
        )


def test_partition_map_auto_registration_is_atomic() -> None:
    partition_map = PartitionMap(GATE_KEY, n_dims=32, n_regimes=32)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        assigned = list(
            pool.map(
                lambda index: partition_map.register(f"partition-{index}"),
                range(32),
            )
        )

    assert sorted(assigned) == list(range(32))
    assert all(
        partition_map.get_mode(f"partition-{index}") is PartitionMode.IMMUTABLE
        for index in range(32)
    )


def test_release_bundle_contains_no_executable_vendor_wheel() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not list((root / "research" / "cdp" / "experiments" / "vendor").glob("*.whl"))
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert "The reference fetcher" not in readme
    assert "does not ship a network fetcher" in readme
