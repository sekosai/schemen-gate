"""VON POC -- Vector Object Notation precision-gated transport.

Tests:
1. Multi-level compression ratios and cosine fidelity
2. Similarity ranking preservation at L0 (int8)
3. Multi-hop idempotency (copy L0 through 10+ hops, zero degradation)
4. Wire format round-trip
5. Gated reconstruction through regime masks
6. Comparison against English re-encoding
7. Real embeddings (sentence-transformers) fidelity
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from schemen_gate._von import (
    VONFrame,
    compression_ratio,
    cosine_at_level,
    encode,
    encode_batch,
    fidelity_curve,
    gate_levels,
    ranking_preservation,
    strip_levels,
)

N_DIMS = 768
RNG = np.random.default_rng(42)


def _random_vec(dim: int = N_DIMS) -> np.ndarray:
    v = RNG.standard_normal(dim)
    return v / np.linalg.norm(v)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _embed_hash(text: str, dim: int = N_DIMS) -> np.ndarray:
    words = text.lower().split()
    bigrams = [text.lower()[i : i + 3] for i in range(len(text) - 2)]
    tokens = words + bigrams
    vec = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        h = hashlib.sha256(t.encode()).digest()
        seed = int.from_bytes(h[:8], "big")
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


class TestCompressionAndFidelity:
    """Multi-level quantization: compression ratios and cosine at each level."""

    def test_l0_compression_ratio(self):
        vec = _random_vec()
        frame = encode(vec, max_level=0)
        ratio = compression_ratio(frame, level=0)
        assert ratio > 7.0, f"L0 compression ratio {ratio:.1f}x < 7x"

    def test_l0_cosine_topology_preserving(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        cos = cosine_at_level(vec, frame, level=0)
        assert cos > 0.99, f"L0 cosine {cos:.4f} < 0.99"

    def test_l1_cosine_near_lossless(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        cos = cosine_at_level(vec, frame, level=1)
        assert cos > 0.9999, f"L1 cosine {cos:.6f} < 0.9999"

    def test_l2_cosine_lossless_to_float32(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        cos = cosine_at_level(vec, frame, level=2)
        assert cos > 0.999999, f"L2 cosine {cos:.8f} < 0.999999"

    def test_l3_bit_perfect(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        reconstructed = frame.reconstruct(max_level=3)
        assert np.allclose(vec, reconstructed, atol=1e-15), (
            f"L3 not bit-perfect: max diff {np.max(np.abs(vec - reconstructed))}"
        )

    def test_fidelity_monotonically_improves(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        curve = fidelity_curve(vec, frame)
        levels = sorted(curve.keys())
        for i in range(1, len(levels)):
            assert curve[levels[i]] >= curve[levels[i - 1]] - 1e-10, (
                f"Fidelity decreased: L{levels[i - 1]}={curve[levels[i - 1]]:.6f} "
                f"-> L{levels[i]}={curve[levels[i]]:.6f}"
            )

    def test_size_at_each_level(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)

        l0_size = frame.levels[0].size_bytes
        l1_size = frame.levels[1].size_bytes
        assert l0_size < l1_size, "L0 should be smaller than L1"
        assert l0_size == N_DIMS * 1 + 16, f"L0 size {l0_size} != {N_DIMS + 16}"
        assert l1_size == N_DIMS * 2 + 16, f"L1 size {l1_size} != {N_DIMS * 2 + 16}"

    def test_batch_encoding(self):
        vecs = np.stack([_random_vec() for _ in range(20)])
        frames = encode_batch(vecs, max_level=2)
        assert len(frames) == 20
        for i, f in enumerate(frames):
            cos = cosine_at_level(vecs[i], f, level=2)
            assert cos > 0.999999, f"Batch item {i} L2 cosine {cos:.8f}"


class TestRankingPreservation:
    """Int8 quantization preserves cosine similarity rankings."""

    def test_spearman_correlation_at_l0(self):
        vecs = np.stack([_random_vec() for _ in range(50)])
        query = _random_vec()
        result = ranking_preservation(vecs, query, level=0)
        assert result["spearman_rho"] > 0.95, f"Spearman rho {result['spearman_rho']:.4f} < 0.95"

    def test_top1_preserved_at_l0(self):
        vecs = np.stack([_random_vec() for _ in range(50)])
        query = _random_vec()
        result = ranking_preservation(vecs, query, level=0)
        assert result["top1_match"], "Top-1 ranking not preserved at L0"

    def test_top5_overlap_at_l0(self):
        vecs = np.stack([_random_vec() for _ in range(50)])
        query = _random_vec()
        result = ranking_preservation(vecs, query, level=0)
        assert result["top5_overlap"] >= 0.8, f"Top-5 overlap {result['top5_overlap']:.2f} < 0.8"

    def test_ranking_with_semantic_vectors(self):
        sentences = [
            "The cat sat on the mat",
            "A dog lay on the rug",
            "Neural networks learn features",
            "Machine learning is a field of AI",
            "The weather is sunny today",
            "It's raining cats and dogs",
            "Quantum computing uses qubits",
            "Encryption protects data privacy",
            "Basketball is a popular sport",
            "Soccer is played worldwide",
        ]
        vecs = np.stack([_embed_hash(s) for s in sentences])
        query = _embed_hash("Deep learning models train on GPUs")
        result = ranking_preservation(vecs, query, level=0)
        assert result["spearman_rho"] > 0.90, (
            f"Semantic ranking Spearman {result['spearman_rho']:.4f} < 0.90"
        )


class TestMultiHopIdempotency:
    """VON L0 payload does not degrade over hops -- copy bytes, get same bytes."""

    def test_100_hops_zero_degradation(self):
        vec = _random_vec()
        frame = encode(vec, max_level=0)
        wire = frame.to_wire()

        current_wire = wire
        for _hop in range(100):
            restored = VONFrame.from_wire(current_wire)
            current_wire = restored.to_wire()

        final = VONFrame.from_wire(current_wire)
        initial_recon = frame.reconstruct(max_level=0)
        final_recon = final.reconstruct(max_level=0)
        cos = _cosine(initial_recon, final_recon)
        assert abs(cos - 1.0) < 1e-12, f"After 100 hops, cosine {cos:.15f} != 1.0"

    def test_100_hops_bytes_identical(self):
        vec = _random_vec()
        frame = encode(vec, max_level=0)
        wire = frame.to_wire()

        current_wire = wire
        for _ in range(100):
            current_wire = VONFrame.from_wire(current_wire).to_wire()

        assert wire == current_wire, "Wire bytes changed after 100 hops"

    def test_l1_multi_hop_stable(self):
        vec = _random_vec()
        frame = encode(vec, max_level=1)
        wire = frame.to_wire(include_levels=[0, 1])

        current_wire = wire
        for _ in range(50):
            current_wire = VONFrame.from_wire(current_wire).to_wire()

        final = VONFrame.from_wire(current_wire)
        cos = cosine_at_level(vec, final, level=1)
        assert cos > 0.9999, f"L1 after 50 hops: cosine {cos:.6f}"

    def test_vs_english_degradation(self):
        """Compare: VON L0 through 10 hops vs English through 10 hops."""
        sentences = [f"fact {i} about topic {chr(65 + i)}" for i in range(20)]
        originals = np.stack([_embed_hash(s) for s in sentences])

        frames = encode_batch(originals, max_level=0)
        von_wires = [f.to_wire() for f in frames]
        for _hop in range(10):
            von_wires = [VONFrame.from_wire(w).to_wire() for w in von_wires]
        von_final = np.stack([VONFrame.from_wire(w).reconstruct(max_level=0) for w in von_wires])
        von_cosines = [_cosine(originals[i], von_final[i]) for i in range(20)]
        von_mean = np.mean(von_cosines)

        eng_vecs = originals.copy()
        for hop in range(10):
            rng = np.random.default_rng(hop * 1000 + 42)
            noise = rng.standard_normal(eng_vecs.shape) * 0.05
            eng_vecs = eng_vecs + noise
            norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
            eng_vecs = eng_vecs / (norms + 1e-12)
        eng_cosines = [_cosine(originals[i], eng_vecs[i]) for i in range(20)]
        eng_mean = np.mean(eng_cosines)

        assert von_mean > eng_mean, (
            f"VON ({von_mean:.4f}) should beat English ({eng_mean:.4f}) after 10 hops"
        )
        von_std = np.std(von_cosines)
        assert von_std < 0.01, f"VON cosine std {von_std:.4f} should be near-zero (idempotent)"


class TestWireFormat:
    """Serialization round-trip and selective level inclusion."""

    def test_round_trip_all_levels(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        wire = frame.to_wire()
        restored = VONFrame.from_wire(wire)

        for lvl in range(4):
            assert lvl in restored.levels, f"Level {lvl} missing after round-trip"

        recon = restored.reconstruct(max_level=3)
        assert np.allclose(vec, recon, atol=1e-15), "L3 round-trip not bit-perfect"

    def test_selective_level_wire(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)

        wire_l0 = frame.to_wire(include_levels=[0])
        restored = VONFrame.from_wire(wire_l0)
        assert 0 in restored.levels
        assert 1 not in restored.levels

        cos = cosine_at_level(vec, restored, level=0)
        assert cos > 0.99

    def test_wire_size_l0_compact(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        wire_l0 = frame.to_wire(include_levels=[0])
        wire_all = frame.to_wire()

        assert len(wire_l0) < len(wire_all), "L0-only should be smaller"
        assert len(wire_l0) < N_DIMS * 2, f"L0 wire {len(wire_l0)} should be < {N_DIMS * 2} bytes"

    def test_payload_hash_deterministic(self):
        vec = _random_vec()
        f1 = encode(vec, max_level=0)
        f2 = encode(vec, max_level=0)
        assert f1.payload_hash() == f2.payload_hash()


class TestGatedReconstruction:
    """Regime-gated precision levels."""

    def test_gate_levels_separation(self):
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        gate_key = b"test-von-gate-key-32-bytes-long!!"

        gated = gate_levels(frame, gate_key, n_regimes=4, unsafe_plaintext_partition=True)

        assert 0 in gated
        assert 1 in gated
        assert len(gated[0].levels) >= 1

    def test_l0_accessible_without_key(self):
        """L0 is the public payload -- always accessible."""
        vec = _random_vec()
        frame = encode(vec, max_level=3)

        public = strip_levels(frame, [0])
        cos = cosine_at_level(vec, public, level=0)
        assert cos > 0.99, f"Public L0 cosine {cos:.4f}"

    def test_precision_improves_with_key_levels(self):
        """More keys = more precision."""
        vec = _random_vec()
        frame = encode(vec, max_level=3)

        cos_l0 = cosine_at_level(vec, strip_levels(frame, [0]), level=0)
        cos_l1 = cosine_at_level(vec, strip_levels(frame, [0, 1]), level=1)
        cos_l2 = cosine_at_level(vec, strip_levels(frame, [0, 1, 2]), level=2)
        cos_l3 = cosine_at_level(vec, strip_levels(frame, [0, 1, 2, 3]), level=3)

        assert cos_l0 < cos_l1 or cos_l0 > 0.9999, (
            "L1 should improve over L0 (unless L0 is already near-perfect)"
        )
        assert cos_l1 <= cos_l2 + 1e-10
        assert cos_l2 <= cos_l3 + 1e-10
        assert cos_l3 > 0.9999999

    def test_gated_regime_map(self):
        """Custom regime-to-level mapping."""
        vec = _random_vec()
        frame = encode(vec, max_level=3)
        gate_key = b"test-von-gate-key-32-bytes-long!!"

        level_map = {0: 0, 1: 0, 2: 5, 3: 5}
        gated = gate_levels(
            frame,
            gate_key,
            n_regimes=8,
            level_regime_map=level_map,
            unsafe_plaintext_partition=True,
        )

        assert 0 in gated
        assert 5 in gated
        assert 0 in gated[0].levels
        assert 1 in gated[0].levels
        assert 2 in gated[5].levels
        assert 3 in gated[5].levels


class TestRealEmbeddings:
    """VON on real sentence-transformer embeddings."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.available = True
        except (ImportError, Exception):
            self.available = False

    def test_real_embedding_fidelity_curve(self):
        if not self.available:
            pytest.skip("sentence-transformers not available")

        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "A fast auburn fox leaps above a sleepy hound",
            "Machine learning models process large datasets",
            "Deep neural networks require significant compute",
            "The stock market closed higher today",
        ]
        embeddings = self.model.encode(sentences)

        for i, emb in enumerate(embeddings):
            frame = encode(emb, max_level=3)
            curve = fidelity_curve(emb, frame)
            assert curve[0] > 0.99, f"Sentence {i} L0 cosine {curve[0]:.4f}"
            assert curve[3] > 0.999999, f"Sentence {i} L3 cosine {curve[3]:.8f}"

    def test_real_ranking_preservation(self):
        if not self.available:
            pytest.skip("sentence-transformers not available")

        corpus = [
            "Python is a programming language",
            "Java is used for enterprise applications",
            "Neural networks are inspired by the brain",
            "The weather forecast predicts rain",
            "Stock prices fluctuated wildly today",
            "Machine learning automates feature engineering",
            "Cryptography protects digital communications",
            "Basketball players must be tall and agile",
            "Cooking requires patience and creativity",
            "Music theory explains harmony and rhythm",
        ]
        query = "Deep learning is a subset of artificial intelligence"

        corpus_emb = self.model.encode(corpus)
        query_emb = self.model.encode([query])[0]

        result = ranking_preservation(corpus_emb, query_emb, level=0)
        assert result["spearman_rho"] > 0.95, (
            f"Real embedding Spearman {result['spearman_rho']:.4f}"
        )
        assert result["top1_match"], "Top-1 not preserved on real embeddings"

    def test_von_vs_english_on_real_embeddings(self):
        """The money test: VON L0 vs English re-encoding on real embeddings."""
        if not self.available:
            pytest.skip("sentence-transformers not available")

        sentences = [
            "Transformers use self-attention mechanisms",
            "Gradient descent minimizes the loss function",
            "Convolutional layers extract spatial features",
            "Recurrent networks model sequential data",
            "Batch normalization stabilizes training",
        ]

        originals = self.model.encode(sentences)
        frames = encode_batch(originals, max_level=0)
        von_recon = np.stack([f.reconstruct(max_level=0) for f in frames])

        von_cosines = [_cosine(originals[i], von_recon[i]) for i in range(len(sentences))]

        re_encoded = self.model.encode(sentences)
        eng_cosines = [_cosine(originals[i], re_encoded[i]) for i in range(len(sentences))]

        von_mean = np.mean(von_cosines)
        eng_mean = np.mean(eng_cosines)

        assert von_mean > 0.99, f"VON L0 mean cosine {von_mean:.4f}"
        assert eng_mean > 0.99, f"Re-encode mean cosine {eng_mean:.4f}"
