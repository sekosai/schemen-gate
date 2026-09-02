"""Follow-up 4: Completion-gated receipts.

Extends POC 4 so that session depart() requires a decoded English
summary as a completion condition.  The decoded text becomes part of
the receipt -- human-auditable provenance for every agent action.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from schemen_gate._cargo import (
    CargoItem,
    CargoReceipt,
    EmbeddingSpec,
    compute_payload_hash,
)
from schemen_gate._cargo_impl import BusTerminal, LiveDockingSession, create_manifest
from schemen_gate._rag import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
)

N_DIMS = 64
N_REGIMES = 4
GATE_KEY = hashlib.sha256(b"vectorese-followup-4-completion").digest()

EMBED_SPEC = EmbeddingSpec(
    model_id="poc-completion",
    dimensions=N_DIMS,
    vocabulary_hash=hashlib.sha256(b"completion-vocab").hexdigest(),
    pooling="cls",
)


def _embed(text: str) -> np.ndarray:
    h = hashlib.sha256(text.encode()).digest()
    rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
    vec = rng.standard_normal(N_DIMS)
    return vec / np.linalg.norm(vec)


class SimpleDecoder:
    """Simulates a decoder that produces a text summary from a vector."""

    def __init__(self, dim: int, vocab_size: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((vocab_size, dim))
        self.vocab = [f"tok_{i}" for i in range(vocab_size)]
        self.dim = dim

    def decode(self, vector: np.ndarray, n_tokens: int = 10) -> str:
        vec = np.asarray(vector, dtype=np.float64).ravel()[: self.dim]
        scores = self.W @ vec
        top = np.argsort(scores)[-n_tokens:][::-1]
        return " ".join(self.vocab[i] for i in top)


@dataclass
class CompletionGatedReceipt:
    """Extended receipt that includes a decoded English summary."""

    base_receipt: CargoReceipt
    decoded_summary: str
    decoder_regime_id: int
    decode_timestamp: float
    summary_hash: str

    def verify_summary(self) -> bool:
        expected = hashlib.sha256(self.decoded_summary.encode()).hexdigest()
        return expected == self.summary_hash


class CompletionGatedSession:
    """Wraps a LiveDockingSession and requires decoded output for departure."""

    def __init__(
        self,
        session: LiveDockingSession,
        decoder: SimpleDecoder,
        decoder_regime_id: int = 1,
    ):
        self._session = session
        self._decoder = decoder
        self._decoder_regime_id = decoder_regime_id
        self._operations_log: List[Dict[str, Any]] = []
        self._decoded_summary: Optional[str] = None

    @property
    def session(self) -> LiveDockingSession:
        return self._session

    def log_operation(self, op_type: str, details: Dict[str, Any]) -> None:
        self._operations_log.append(
            {
                "type": op_type,
                "timestamp": time.time(),
                **details,
            }
        )

    def decode_summary(self, context_vector: np.ndarray) -> str:
        """Produce the English summary required for completion."""
        ops_text = "; ".join(f"{op['type']}" for op in self._operations_log)
        combined = context_vector.copy()
        if ops_text:
            ops_vec = _embed(ops_text)
            combined = combined + ops_vec * 0.1
            norm = np.linalg.norm(combined)
            if norm > 1e-12:
                combined = combined / norm

        self._decoded_summary = self._decoder.decode(combined)
        return self._decoded_summary

    def depart_with_completion(
        self,
        context_vector: np.ndarray,
    ) -> CompletionGatedReceipt:
        """Depart only after producing a decoded summary."""
        if self._decoded_summary is None:
            self.decode_summary(context_vector)

        base_receipt = self._session.depart()
        summary_hash = hashlib.sha256(self._decoded_summary.encode()).hexdigest()

        return CompletionGatedReceipt(
            base_receipt=base_receipt,
            decoded_summary=self._decoded_summary,
            decoder_regime_id=self._decoder_regime_id,
            decode_timestamp=time.time(),
            summary_hash=summary_hash,
        )


def _build_adapter():
    store = InMemoryVectorStore()
    pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
    pmap.register("regime-0", regime_id=0, mode=PartitionMode.READ_WRITE)
    adapter = GatedRAGAdapter(store=store, partition_map=pmap, embed_fn=_embed)
    return adapter, pmap


class TestCompletionGatedReceipts:
    def test_receipt_includes_summary(self):
        adapter, _pmap = _build_adapter()
        decoder = SimpleDecoder(N_DIMS)
        terminal = BusTerminal(adapter=adapter, gate_key_secret=GATE_KEY, embedding_spec=EMBED_SPEC)
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [CargoItem(content="test doc", embedding=_embed("test doc"), doc_id="d1")]
        manifest = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="test-subject",
            model_digest="test-model",
            operation="load_and_retrieve",
            policy_version="test-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        gated = CompletionGatedSession(session, decoder)

        gated.log_operation("load", {"count": 1})
        session.load_cargo(items)

        gated.log_operation("query", {"query": "test"})
        context_vec = _embed("test query context")

        receipt = gated.depart_with_completion(context_vec)

        assert receipt.decoded_summary is not None
        assert len(receipt.decoded_summary) > 0
        assert receipt.base_receipt.manifest_id == manifest.manifest_id

    def test_summary_hash_verifies(self):
        adapter, _ = _build_adapter()
        decoder = SimpleDecoder(N_DIMS)
        terminal = BusTerminal(adapter=adapter, gate_key_secret=GATE_KEY, embedding_spec=EMBED_SPEC)
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [CargoItem(content="doc", embedding=_embed("doc"), doc_id="d1")]
        manifest = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="test-subject",
            model_digest="test-model",
            operation="load_and_retrieve",
            policy_version="test-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        gated = CompletionGatedSession(session, decoder)
        session.load_cargo(items)
        receipt = gated.depart_with_completion(_embed("context"))

        assert receipt.verify_summary()

    def test_tampered_summary_fails_verification(self):
        adapter, _ = _build_adapter()
        decoder = SimpleDecoder(N_DIMS)
        terminal = BusTerminal(adapter=adapter, gate_key_secret=GATE_KEY, embedding_spec=EMBED_SPEC)
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [CargoItem(content="doc", embedding=_embed("doc"), doc_id="d1")]
        manifest = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="test-subject",
            model_digest="test-model",
            operation="load_and_retrieve",
            policy_version="test-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        gated = CompletionGatedSession(session, decoder)
        session.load_cargo(items)
        receipt = gated.depart_with_completion(_embed("context"))

        tampered = CompletionGatedReceipt(
            base_receipt=receipt.base_receipt,
            decoded_summary="tampered summary text",
            decoder_regime_id=receipt.decoder_regime_id,
            decode_timestamp=receipt.decode_timestamp,
            summary_hash=receipt.summary_hash,
        )
        assert not tampered.verify_summary()

    def test_different_operations_produce_different_summaries(self):
        adapter, _ = _build_adapter()
        decoder = SimpleDecoder(N_DIMS)

        terminal = BusTerminal(adapter=adapter, gate_key_secret=GATE_KEY, embedding_spec=EMBED_SPEC)
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [CargoItem(content="doc", embedding=_embed("doc"), doc_id="d1")]

        m1 = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="test-subject",
            model_digest="test-model",
            operation="load_and_retrieve",
            policy_version="test-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )
        s1 = bus.dock(m1, gate_key_secret=GATE_KEY)
        g1 = CompletionGatedSession(s1, decoder)
        g1.log_operation("load", {})
        s1.load_cargo(items)
        r1 = g1.depart_with_completion(_embed("query about databases"))

        m2 = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="test-subject",
            model_digest="test-model",
            operation="load_and_retrieve",
            policy_version="test-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )
        s2 = bus.dock(m2, gate_key_secret=GATE_KEY)
        g2 = CompletionGatedSession(s2, decoder)
        g2.log_operation("load", {})
        g2.log_operation("query", {})
        g2.log_operation("unload", {})
        s2.load_cargo(items)
        r2 = g2.depart_with_completion(_embed("query about security"))

        assert r1.decoded_summary != r2.decoded_summary

    def test_base_receipt_still_valid(self):
        """The completion gate doesn't break the underlying receipt."""
        adapter, _ = _build_adapter()
        decoder = SimpleDecoder(N_DIMS)
        terminal = BusTerminal(adapter=adapter, gate_key_secret=GATE_KEY, embedding_spec=EMBED_SPEC)
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [CargoItem(content="doc", embedding=_embed("doc"), doc_id="d1")]
        manifest = create_manifest(
            tenant_id="t1",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="test-subject",
            model_digest="test-model",
            operation="load_and_retrieve",
            policy_version="test-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        gated = CompletionGatedSession(session, decoder)
        session.load_cargo(items)
        receipt = gated.depart_with_completion(_embed("context"))

        assert bus.verify_receipt(
            receipt.base_receipt,
            expected_manifest=manifest,
        )
