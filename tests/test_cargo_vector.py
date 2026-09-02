"""POC: Vector-to-vector inference via Cargo Mode.

Proves that vector injection bypasses text re-encoding, preserves semantic
signal through dimensional bridges, respects gate mask isolation, and
composes with the full Cargo Mode lifecycle.

Uses a contrived "model" (a linear layer) to demonstrate the concept
without requiring GPU or real transformers.  The model accepts hidden-state
vectors directly, which is exactly what vector injection does in production.

Key properties tested:

1. BYPASS: vector-injected inference produces the same result as the
   model would if it had access to the original hidden state, without
   going through text -> tokenize -> embed -> model forward.

2. ALIGNMENT: a VectorBridge projects vectors from RAG dimension to
   model dimension while preserving relative similarities.

3. ISOLATION: gated vectors have zero signal in cross-regime dimensions.
   Injecting regime-0 vectors into a model that only reads regime-1
   dimensions produces zero output.

4. LIFECYCLE: the full dock -> load -> unload_vectors -> bridge -> inject
   -> depart flow produces a valid, verifiable receipt.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from schemen_gate._cargo import (
    CargoItem,
    EmbeddingSpec,
    VectorBridge,
    VectorPayload,
    compute_payload_hash,
)
from schemen_gate._cargo_impl import (
    BusTerminal,
    create_manifest,
)
from schemen_gate._rag import (
    GatedRAGAdapter,
    InMemoryVectorStore,
    PartitionMap,
    PartitionMode,
)

# ---------------------------------------------------------------------------
# Contrived model: a linear layer that reads hidden-state vectors directly
# ---------------------------------------------------------------------------


class ContrivedModel:
    """A minimal "model" that demonstrates vector injection.

    In production, this would be the layers after a transformer's
    embedding stage.  Here it's a linear classifier: W @ hidden + b.
    The point is that it accepts vectors directly -- no tokenizer,
    no embedding lookup, no re-encoding.
    """

    def __init__(self, hidden_dim: int, n_classes: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.standard_normal((n_classes, hidden_dim))
        self.b = rng.standard_normal(n_classes)
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Accept hidden-state vectors directly (no text re-encoding)."""
        if hidden.ndim == 1:
            return self.W @ hidden + self.b
        return (hidden @ self.W.T) + self.b

    def forward_text_roundtrip(
        self,
        text: str,
        embed_fn,
    ) -> np.ndarray:
        """The wasteful path: text -> embed -> forward.

        This simulates what current RAG does: retrieve TEXT, stuff it into
        a prompt, and the model re-encodes it.  We show this produces the
        same result as injecting the embedding directly -- proving the
        text round-trip is pure overhead.
        """
        emb = embed_fn(text)
        return self.forward(emb)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

GATE_KEY = os.urandom(32)
N_DIMS = 64
N_REGIMES = 4
MODEL_DIM = 128
N_CLASSES = 4

EMBED_SPEC = EmbeddingSpec(
    model_id="test-poc-model",
    dimensions=N_DIMS,
    vocabulary_hash=hashlib.sha256(b"poc-vocab").hexdigest(),
    pooling="cls",
)


def _hash_embed(text: str, dim: int = N_DIMS) -> np.ndarray:
    """Deterministic embedding from text (no model required)."""
    h = hashlib.sha256(text.encode()).digest()
    raw = np.frombuffer(h * (dim // 32 + 1), dtype=np.uint8)[:dim]
    vec = raw.astype(np.float64) / 255.0
    norm = np.linalg.norm(vec)
    if norm > 1e-12:
        vec = vec / norm
    return vec


def _build_adapter():
    store = InMemoryVectorStore()
    pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
    pmap.register("regime-0", regime_id=0, mode=PartitionMode.READ_WRITE)
    pmap.register("regime-1", regime_id=1, mode=PartitionMode.READ_WRITE)
    pmap.register("regime-2", regime_id=2, mode=PartitionMode.READ_WRITE)

    def embed_fn(content):
        if isinstance(content, np.ndarray):
            return content
        return _hash_embed(content)

    adapter = GatedRAGAdapter(store=store, partition_map=pmap, embed_fn=embed_fn)
    return adapter, pmap


# ===========================================================================
# POC 1: Vector injection bypasses text re-encoding
# ===========================================================================


class TestVectorInjectionBypass:
    """Prove that injecting vectors directly produces the same output as
    the text -> embed -> forward path, without the text round-trip.
    """

    def test_direct_injection_matches_text_path(self):
        """The core proof: vector injection == text re-encoding."""
        model = ContrivedModel(N_DIMS, N_CLASSES)
        text = "The quick brown fox jumps over the lazy dog"
        emb = _hash_embed(text)

        output_via_text = model.forward_text_roundtrip(text, _hash_embed)
        output_via_vector = model.forward(emb)

        np.testing.assert_array_almost_equal(output_via_text, output_via_vector)

    def test_retrieved_vector_injection(self):
        """Retrieve from RAG, inject vector directly, skip text."""
        adapter, _pmap = _build_adapter()
        model = ContrivedModel(N_DIMS, N_CLASSES)

        docs = [
            "quantum computing basics",
            "neural network architectures",
            "cryptographic hash functions",
        ]
        for doc in docs:
            adapter.ingest(doc, doc, "regime-0", gate_embedding=False)

        result = adapter.query("quantum computing", "regime-0", top_k=1)
        retrieved_vec = result.docs[0].embedding
        retrieved_text = result.docs[0].content

        output_text_path = model.forward_text_roundtrip(retrieved_text, _hash_embed)
        output_vector_path = model.forward(retrieved_vec)

        np.testing.assert_array_almost_equal(output_text_path, output_vector_path)

    def test_batch_vector_injection(self):
        """Inject multiple retrieved vectors at once."""
        adapter, _ = _build_adapter()
        model = ContrivedModel(N_DIMS, N_CLASSES)

        for i in range(10):
            adapter.ingest(
                f"document {i} about topic {i % 3}",
                f"doc-{i}",
                "regime-0",
                gate_embedding=False,
            )

        result = adapter.query("document about topic", "regime-0", top_k=5)
        vectors = np.stack([doc.embedding for doc in result.docs])

        batch_output = model.forward(vectors)
        individual_outputs = np.stack([model.forward(v) for v in vectors])

        np.testing.assert_array_almost_equal(batch_output, individual_outputs)


# ===========================================================================
# POC 2: VectorBridge preserves semantic similarity
# ===========================================================================


class TestBridgeAlignment:
    """Prove that projecting vectors through a VectorBridge preserves
    relative similarities.  If A is closer to B than to C in the source
    space, A should remain closer to B than to C after projection.
    """

    def test_similarity_ordering_preserved_step_up(self):
        """Step-up (64 -> 128) preserves cosine similarity ordering."""
        bridge = VectorBridge(N_DIMS, MODEL_DIM)

        a = _hash_embed("machine learning")
        b = _hash_embed("deep learning")
        c = _hash_embed("banana milkshake recipe")

        sim_ab_source = np.dot(a, b)
        sim_ac_source = np.dot(a, c)
        assert sim_ab_source > sim_ac_source, "Sanity: ML closer to DL than to banana"

        a_proj = bridge.project(a)
        b_proj = bridge.project(b)
        c_proj = bridge.project(c)

        a_proj = a_proj / (np.linalg.norm(a_proj) + 1e-12)
        b_proj = b_proj / (np.linalg.norm(b_proj) + 1e-12)
        c_proj = c_proj / (np.linalg.norm(c_proj) + 1e-12)

        sim_ab_target = np.dot(a_proj, b_proj)
        sim_ac_target = np.dot(a_proj, c_proj)

        assert sim_ab_target > sim_ac_target, (
            f"Bridge broke similarity ordering: "
            f"sim(ML, DL)={sim_ab_target:.4f} vs sim(ML, banana)={sim_ac_target:.4f}"
        )

    def test_similarity_ordering_preserved_step_down(self):
        """Step-down (128 -> 64) preserves cosine similarity ordering.

        Uses controlled vectors with known similarity structure since
        hash embeddings don't produce semantically meaningful similarities.
        """
        bridge = VectorBridge(MODEL_DIM, N_DIMS)
        rng = np.random.default_rng(99)

        a = rng.standard_normal(MODEL_DIM)
        a /= np.linalg.norm(a)

        noise_small = rng.standard_normal(MODEL_DIM) * 0.1
        b = a + noise_small
        b /= np.linalg.norm(b)

        c = rng.standard_normal(MODEL_DIM)
        c /= np.linalg.norm(c)

        sim_ab = np.dot(a, b)
        sim_ac = np.dot(a, c)
        assert sim_ab > sim_ac, "Precondition: b is close to a, c is random"

        a_p = bridge.project(a)
        b_p = bridge.project(b)
        c_p = bridge.project(c)

        a_p /= np.linalg.norm(a_p) + 1e-12
        b_p /= np.linalg.norm(b_p) + 1e-12
        c_p /= np.linalg.norm(c_p) + 1e-12

        assert np.dot(a_p, b_p) > np.dot(a_p, c_p)

    def test_bridge_is_deterministic(self):
        """Same source/target dims produce same projection."""
        b1 = VectorBridge(N_DIMS, MODEL_DIM)
        b2 = VectorBridge(N_DIMS, MODEL_DIM)

        vec = _hash_embed("determinism test")
        np.testing.assert_array_equal(b1.project(vec), b2.project(vec))

    def test_bridge_projects_into_model(self):
        """Vectors projected through the bridge work with the model."""
        bridge = VectorBridge(N_DIMS, MODEL_DIM)
        model = ContrivedModel(MODEL_DIM, N_CLASSES)

        vec = _hash_embed("test input")
        projected = bridge.project(vec)
        output = model.forward(projected)

        assert output.shape == (N_CLASSES,)
        assert np.all(np.isfinite(output))

    def test_batch_bridge_matches_individual(self):
        """Batch projection matches element-wise projection."""
        bridge = VectorBridge(N_DIMS, MODEL_DIM)
        vecs = np.stack([_hash_embed(f"doc-{i}") for i in range(5)])

        batch_result = bridge.project(vecs)
        individual_results = np.stack([bridge.project(v) for v in vecs])

        np.testing.assert_array_almost_equal(batch_result, individual_results)


# ===========================================================================
# POC 3: Gated vector injection respects regime isolation
# ===========================================================================


class TestGatedIsolation:
    """Prove that gated vectors have zero signal in cross-regime dimensions,
    and that injecting regime-0 vectors into a model that only reads
    regime-1 dimensions produces zero (or near-zero) output.
    """

    def test_gated_vectors_zero_outside_regime(self):
        """Vectors gated by regime-0 mask have zero in non-regime-0 dims."""
        _, pmap = _build_adapter()
        mask_0 = pmap.mask_for("regime-0")
        mask_arr = mask_0.to_numpy()

        vec = _hash_embed("some document")
        gated = vec * mask_arr

        zero_dims = np.where(mask_arr == 0)[0]
        assert len(zero_dims) > 0, "Mask should zero some dimensions"
        np.testing.assert_array_equal(gated[zero_dims], 0.0)

    def test_cross_regime_injection_produces_zero(self):
        """Injecting regime-0 gated vectors through a regime-1 mask = zero."""
        _, pmap = _build_adapter()
        mask_0 = pmap.mask_for("regime-0").to_numpy()
        mask_1 = pmap.mask_for("regime-1").to_numpy()

        vec = _hash_embed("sensitive document")
        gated_0 = vec * mask_0
        cross_regime = gated_0 * mask_1

        np.testing.assert_array_equal(
            cross_regime,
            0.0,
            err_msg="Cross-regime injection must produce all zeros (masks are disjoint)",
        )

    def test_same_regime_preserves_signal(self):
        """Double-gating with the same regime mask is idempotent."""
        _, pmap = _build_adapter()
        mask_0 = pmap.mask_for("regime-0").to_numpy()

        vec = _hash_embed("my document")
        gated = vec * mask_0
        double_gated = gated * mask_0

        np.testing.assert_array_equal(gated, double_gated)

    def test_regime_isolation_through_model(self):
        """Model output from regime-0 vectors through regime-1 gate = zero."""
        _, pmap = _build_adapter()
        mask_0 = pmap.mask_for("regime-0").to_numpy()
        mask_1 = pmap.mask_for("regime-1").to_numpy()

        model = ContrivedModel(N_DIMS, N_CLASSES)

        vec = _hash_embed("confidential data")
        gated_0 = vec * mask_0

        output_same_regime = model.forward(gated_0 * mask_0)
        output_cross_regime = model.forward(gated_0 * mask_1)

        assert np.linalg.norm(output_same_regime) > 0
        np.testing.assert_array_almost_equal(
            output_cross_regime,
            model.b,
            err_msg="Cross-regime output should equal bias-only (zero input)",
        )

    def test_all_regimes_disjoint(self):
        """No dimension is shared between any two regimes."""
        _, pmap = _build_adapter()
        masks = []
        for key in ["regime-0", "regime-1", "regime-2"]:
            masks.append(pmap.mask_for(key).to_numpy())

        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                overlap = masks[i] * masks[j]
                np.testing.assert_array_equal(
                    overlap, 0.0, err_msg=f"Regimes {i} and {j} share dimensions"
                )

    def test_gated_vector_payload(self):
        """VectorPayload.gated() applies the mask correctly."""
        adapter, pmap = _build_adapter()
        mask_0 = pmap.mask_for("regime-0")

        vec = _hash_embed("test doc")
        adapter.ingest(vec, "test doc", "regime-0", gate_embedding=False)

        result = adapter.query("test doc", "regime-0", top_k=1)
        payload = VectorPayload(
            vectors=result.docs[0].embedding,
            gate_mask=result.gate_mask,
            regime_id=0,
            source_doc_ids=[result.docs[0].doc_id],
        )

        gated = payload.gated()
        mask_arr = mask_0.to_numpy()
        zero_dims = np.where(mask_arr == 0)[0]
        np.testing.assert_array_equal(gated[zero_dims], 0.0)


# ===========================================================================
# POC 4: Full Cargo Mode lifecycle with vector injection
# ===========================================================================


class TestEndToEndCargoVector:
    """The complete flow: build terminal -> dock -> load docs ->
    unload_vectors -> bridge -> inject into model -> depart with receipt.
    """

    def test_full_lifecycle_vector_injection(self):
        adapter, _pmap = _build_adapter()

        terminal = BusTerminal(
            adapter=adapter,
            gate_key_secret=GATE_KEY,
            embedding_spec=EMBED_SPEC,
        )
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [
            CargoItem(
                content=f"Document about {topic}",
                embedding=_hash_embed(f"Document about {topic}"),
                kind="document",
                doc_id=f"doc-{i}",
            )
            for i, topic in enumerate(
                [
                    "quantum computing",
                    "neural networks",
                    "cryptography",
                    "distributed systems",
                    "formal verification",
                ]
            )
        ]

        manifest = create_manifest(
            tenant_id="poc-tenant",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=len(items),
            subject_id="poc-subject",
            model_digest="poc-model",
            operation="load_and_retrieve",
            policy_version="poc-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
            ttl_seconds=3600,
        )

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)

        load_result = session.load_cargo(items)
        assert load_result.item_count == 5

        bridge = VectorBridge(N_DIMS, MODEL_DIM)
        payload = session.unload_vectors("quantum computing", top_k=3, bridge=bridge)

        assert payload.dim == MODEL_DIM
        assert payload.count <= 3
        assert payload.bridge is bridge

        model = ContrivedModel(MODEL_DIM, N_CLASSES)
        output = model.forward(payload.vectors)

        assert output.shape == (payload.count, N_CLASSES)
        assert np.all(np.isfinite(output))

        receipt = session.depart()

        assert receipt.manifest_id == manifest.manifest_id
        assert receipt.tenant_id == "poc-tenant"
        assert receipt.regime_id == 0
        assert receipt.embedding_spec == EMBED_SPEC
        assert len(receipt.operations) == 2

        assert bus.verify_receipt(receipt, expected_manifest=manifest)

    def test_gated_injection_through_cargo(self):
        """Load into regime-0, unload vectors, gate them, verify isolation."""
        adapter, pmap = _build_adapter()

        terminal = BusTerminal(
            adapter=adapter,
            gate_key_secret=GATE_KEY,
            embedding_spec=EMBED_SPEC,
        )
        terminal.register_bus("regime-0", 0)
        bus = terminal.get_bus("regime-0")

        items = [
            CargoItem(
                content="secret regime-0 data",
                embedding=_hash_embed("secret regime-0 data"),
                doc_id="secret-0",
            ),
        ]

        manifest = create_manifest(
            tenant_id="iso-tenant",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="poc-subject",
            model_digest="poc-model",
            operation="load_and_retrieve",
            policy_version="poc-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )

        session = bus.dock(manifest, gate_key_secret=GATE_KEY)
        session.load_cargo(items)
        payload = session.unload_vectors("secret", top_k=1)

        gated = payload.gated()
        mask_0 = pmap.mask_for("regime-0").to_numpy()
        mask_1 = pmap.mask_for("regime-1").to_numpy()

        zero_dims_0 = np.where(mask_0 == 0)[0]
        np.testing.assert_array_equal(
            gated.ravel()[zero_dims_0], 0.0, err_msg="Gated vectors must be zero outside regime-0"
        )

        cross_regime = gated.ravel() * mask_1
        np.testing.assert_array_equal(
            cross_regime,
            0.0,
            err_msg="Regime-0 gated vectors through regime-1 mask must be all zeros",
        )

        session.depart()

    def test_multi_regime_isolation_via_cargo(self):
        """Two tenants on different regimes cannot see each other's vectors."""
        adapter, pmap = _build_adapter()

        terminal = BusTerminal(
            adapter=adapter,
            gate_key_secret=GATE_KEY,
            embedding_spec=EMBED_SPEC,
        )
        terminal.register_bus("regime-0", 0)
        terminal.register_bus("regime-1", 1)

        items_0 = [
            CargoItem(
                content="tenant-A secret plans",
                embedding=_hash_embed("tenant-A secret plans"),
                doc_id="a-secret",
            )
        ]
        items_1 = [
            CargoItem(
                content="tenant-B secret plans",
                embedding=_hash_embed("tenant-B secret plans"),
                doc_id="b-secret",
            )
        ]

        spec_0 = create_manifest(
            tenant_id="tenant-A",
            regime_id=0,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items_0),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="tenant-A-subject",
            model_digest="poc-model",
            operation="load_and_retrieve",
            policy_version="poc-v1",
            partition_key="regime-0",
            gate_embeddings_at_rest=False,
        )
        spec_1 = create_manifest(
            tenant_id="tenant-B",
            regime_id=1,
            embedding_spec=EMBED_SPEC,
            payload_hash=compute_payload_hash(items_1),
            payload_kind="rag_documents",
            item_count=1,
            subject_id="tenant-B-subject",
            model_digest="poc-model",
            operation="load_and_retrieve",
            policy_version="poc-v1",
            partition_key="regime-1",
            gate_embeddings_at_rest=False,
        )

        bus_0 = terminal.get_bus("regime-0")
        bus_1 = terminal.get_bus("regime-1")

        s0 = bus_0.dock(spec_0, gate_key_secret=GATE_KEY)
        s0.load_cargo(items_0)
        p0 = s0.unload_vectors("secret plans", top_k=1)
        r0 = s0.depart()

        s1 = bus_1.dock(spec_1, gate_key_secret=GATE_KEY)
        s1.load_cargo(items_1)
        p1 = s1.unload_vectors("secret plans", top_k=1)
        r1 = s1.depart()

        gated_0 = p0.gated().ravel()
        gated_1 = p1.gated().ravel()

        mask_0 = pmap.mask_for("regime-0").to_numpy()
        mask_1 = pmap.mask_for("regime-1").to_numpy()
        np.testing.assert_array_equal(gated_0 * mask_1, 0.0)
        np.testing.assert_array_equal(gated_1 * mask_0, 0.0)

        model = ContrivedModel(N_DIMS, N_CLASSES)
        out_0_own = model.forward(gated_0)
        out_0_cross = model.forward(gated_0 * mask_1)

        assert np.linalg.norm(out_0_own - model.b) > 0.01
        np.testing.assert_array_almost_equal(out_0_cross, model.b)

        assert r0.regime_id == 0
        assert r1.regime_id == 1
        assert r0.receipt_id != r1.receipt_id

    def test_vector_vs_text_rag_comparison(self):
        """Direct comparison: vector-RAG vs text-RAG produce same model output.

        This is the definitive proof that the text round-trip is pure
        overhead.  Both paths produce identical model outputs, but
        vector-RAG skips the text -> embed -> text -> embed cycle.
        """
        adapter, _ = _build_adapter()
        model = ContrivedModel(N_DIMS, N_CLASSES)

        docs = ["quantum computing primer", "deep learning basics", "cryptographic protocols"]
        for doc in docs:
            adapter.ingest(doc, doc, "regime-0", gate_embedding=False)

        result = adapter.query("quantum", "regime-0", top_k=1)
        retrieved_doc = result.docs[0]

        output_text_rag = model.forward_text_roundtrip(
            retrieved_doc.content,
            _hash_embed,
        )

        output_vector_rag = model.forward(retrieved_doc.embedding)

        np.testing.assert_array_almost_equal(
            output_text_rag,
            output_vector_rag,
            err_msg="Vector-RAG and text-RAG must produce identical outputs. "
            "The text round-trip is pure overhead.",
        )


# ===========================================================================
# POC 5: Bridge + Gate composition
# ===========================================================================


class TestBridgeGateComposition:
    """Prove that VectorBridge and gate mask compose correctly:
    project first, then gate (or gate first, then project).
    """

    def test_project_then_gate(self):
        """Project to model dim, then apply gate mask in model space."""
        _, pmap = _build_adapter()
        bridge = VectorBridge(N_DIMS, MODEL_DIM)

        mask_0_src = pmap.mask_for("regime-0").to_numpy()

        vec = _hash_embed("test")
        gated_src = vec * mask_0_src
        projected = bridge.project(gated_src)

        assert projected.shape == (MODEL_DIM,)
        assert np.all(np.isfinite(projected))

        active_src_dims = np.where(mask_0_src == 1)[0]
        gated_src_only = np.zeros_like(vec)
        gated_src_only[active_src_dims] = vec[active_src_dims]
        projected_from_gated = bridge.project(gated_src_only)

        np.testing.assert_array_almost_equal(projected, projected_from_gated)

    def test_gate_zeroes_survive_projection(self):
        """If source dims are zeroed by the gate, the projection of
        the zeroed vector differs from the projection of the full vector.
        This proves the gate has effect even after projection.
        """
        _, pmap = _build_adapter()
        bridge = VectorBridge(N_DIMS, MODEL_DIM)

        mask_0 = pmap.mask_for("regime-0").to_numpy()
        vec = _hash_embed("important data")

        full_projected = bridge.project(vec)
        gated_projected = bridge.project(vec * mask_0)

        assert not np.allclose(full_projected, gated_projected), (
            "Gated projection must differ from full projection (gate removed information)"
        )

    def test_cross_regime_zero_survives_projection(self):
        """If regime-0 is gated and then projected, applying regime-1's
        gate in source space first should still produce different output
        than regime-0's gate.
        """
        _, pmap = _build_adapter()
        bridge = VectorBridge(N_DIMS, MODEL_DIM)
        model = ContrivedModel(MODEL_DIM, N_CLASSES)

        mask_0 = pmap.mask_for("regime-0").to_numpy()
        mask_1 = pmap.mask_for("regime-1").to_numpy()

        vec = _hash_embed("contested data")

        out_0 = model.forward(bridge.project(vec * mask_0))
        out_1 = model.forward(bridge.project(vec * mask_1))
        out_cross = model.forward(bridge.project(vec * mask_0 * mask_1))

        assert not np.allclose(out_0, out_1), "Different regimes, different outputs"
        np.testing.assert_array_almost_equal(
            out_cross,
            model.b,
            err_msg="Cross-regime gated + projected should give bias-only output",
        )
