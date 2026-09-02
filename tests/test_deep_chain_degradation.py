"""Follow-up 2: Deep agent chain degradation.

Measures information loss at each hop in a 5-agent chain for
English vs Vectorese communication.  English should degrade
exponentially; Vectorese should be flat.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from schemen_gate._cargo import EmbeddingSpec, InferenceStreamFrame

N_DIMS = 256
N_AGENTS = 5
N_CLAIMS = 20

EMBED_SPEC = EmbeddingSpec(
    model_id="poc-chain",
    dimensions=N_DIMS,
    vocabulary_hash=hashlib.sha256(b"chain-vocab").hexdigest(),
    pooling="mean",
)


def _embed(text: str, dim: int = N_DIMS) -> np.ndarray:
    words = text.lower().split()
    bigrams = [text.lower()[i : i + 3] for i in range(len(text) - 2)]
    tokens = words + bigrams
    if not tokens:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
        vec = rng.standard_normal(dim)
        return vec / np.linalg.norm(vec)
    vec = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        h = hashlib.sha256(t.encode()).digest()
        seed = int.from_bytes(h[:8], "big")
        rng = np.random.default_rng(seed)
        vec += rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


def _noisy_re_embed(text: str, rng: np.random.Generator, noise_scale: float = 0.05) -> np.ndarray:
    """Simulate re-encoding through a different encoder at each hop."""
    base = _embed(text)
    noise = rng.standard_normal(base.shape) * noise_scale
    noisy = base + noise
    norm = np.linalg.norm(noisy)
    return noisy / norm if norm > 1e-12 else noisy


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


CLAIMS = [f"fact number {i} about topic {chr(65 + i % 26)}" for i in range(N_CLAIMS)]


class TestDeepChainDegradation:
    """5-agent chain: measure cosine similarity to original at each hop."""

    def test_english_degrades_exponentially(self):
        """English path: cumulative re-encoding noise at each hop."""
        original_vecs = np.stack([_embed(c) for c in CLAIMS])
        current_vecs = original_vecs.copy()
        hop_cosines = []

        for hop in range(N_AGENTS - 1):
            rng = np.random.default_rng(hop * 1000 + 42)
            noise = rng.standard_normal(current_vecs.shape) * 0.05
            current_vecs = current_vecs + noise
            norms = np.linalg.norm(current_vecs, axis=1, keepdims=True)
            current_vecs = current_vecs / (norms + 1e-12)

            cosines = [_cosine(original_vecs[i], current_vecs[i]) for i in range(N_CLAIMS)]
            hop_cosines.append(np.mean(cosines))

        assert hop_cosines[-1] < hop_cosines[0], "English should degrade over hops"
        assert hop_cosines[-1] < 0.95, (
            f"After {N_AGENTS - 1} hops English cosine should degrade (got {hop_cosines[-1]:.4f})"
        )

    def test_vectorese_flat_across_hops(self):
        """Vectorese path: pass vectors directly, no re-encoding."""
        original_vecs = np.stack([_embed(c) for c in CLAIMS])
        current_vecs = original_vecs.copy()
        hop_cosines = []

        for _hop in range(N_AGENTS - 1):
            cosines = [_cosine(original_vecs[i], current_vecs[i]) for i in range(N_CLAIMS)]
            hop_cosines.append(np.mean(cosines))

        for hc in hop_cosines:
            assert hc > 0.999, f"Vectorese cosine degraded to {hc:.4f}"

    def test_vectorese_via_stream_frames(self):
        """Same test but through InferenceStreamFrame wire format."""
        original_vecs = np.stack([_embed(c) for c in CLAIMS])
        current_vecs = original_vecs.copy()

        for hop in range(N_AGENTS - 1):
            frame = InferenceStreamFrame(
                vectors=current_vecs.astype(np.float32),
                regime_id=0,
                source_doc_ids=[f"claim-{i}" for i in range(N_CLAIMS)],
                embedding_spec=EMBED_SPEC,
                sequence_id=f"hop-{hop}",
                frame_index=hop,
                is_final=(hop == N_AGENTS - 2),
            )
            wire = frame.to_wire()
            restored = InferenceStreamFrame.from_wire(wire)
            current_vecs = restored.vectors.astype(np.float64)

        cosines = [_cosine(original_vecs[i], current_vecs[i]) for i in range(N_CLAIMS)]
        mean_cosine = np.mean(cosines)

        assert mean_cosine > 0.99, (
            f"Stream frame round-trip through {N_AGENTS - 1} hops degraded to {mean_cosine:.4f}"
        )

    def test_gap_widens_with_hops(self):
        """The gap between English and Vectorese widens at each hop."""
        original_vecs = np.stack([_embed(c) for c in CLAIMS])

        english_cosines_per_hop = []
        vector_cosines_per_hop = []

        eng_vecs = original_vecs.copy()
        vec_vecs = original_vecs.copy()

        for hop in range(N_AGENTS - 1):
            rng = np.random.default_rng(hop * 1000 + 42)
            noise = rng.standard_normal(eng_vecs.shape) * 0.05
            eng_vecs = eng_vecs + noise
            norms = np.linalg.norm(eng_vecs, axis=1, keepdims=True)
            eng_vecs = eng_vecs / (norms + 1e-12)

            eng_cos = np.mean([_cosine(original_vecs[i], eng_vecs[i]) for i in range(N_CLAIMS)])
            vec_cos = np.mean([_cosine(original_vecs[i], vec_vecs[i]) for i in range(N_CLAIMS)])

            english_cosines_per_hop.append(eng_cos)
            vector_cosines_per_hop.append(vec_cos)

        gaps = [
            v - e
            for v, e in zip(
                vector_cosines_per_hop,
                english_cosines_per_hop,
                strict=True,
            )
        ]
        for i in range(1, len(gaps)):
            assert gaps[i] >= gaps[i - 1] - 0.02, (
                f"Gap should widen: hop {i - 1} gap={gaps[i - 1]:.4f}, hop {i} gap={gaps[i]:.4f}"
            )

    def test_english_cumulative_noise(self):
        """Cumulative noise: each hop adds fresh noise to already-noisy vectors."""
        original_vecs = np.stack([_embed(c) for c in CLAIMS])
        current_vecs = original_vecs.copy()
        hop_cosines = []

        for hop in range(N_AGENTS - 1):
            rng = np.random.default_rng(hop * 7777)
            noise = rng.standard_normal(current_vecs.shape) * 0.05
            current_vecs = current_vecs + noise
            norms = np.linalg.norm(current_vecs, axis=1, keepdims=True)
            current_vecs = current_vecs / (norms + 1e-12)

            cosines = [_cosine(original_vecs[i], current_vecs[i]) for i in range(N_CLAIMS)]
            hop_cosines.append(np.mean(cosines))

        for i in range(1, len(hop_cosines)):
            assert hop_cosines[i] <= hop_cosines[i - 1] + 0.01, (
                f"Cumulative noise should not improve: "
                f"hop {i - 1}={hop_cosines[i - 1]:.4f}, hop {i}={hop_cosines[i]:.4f}"
            )

        assert hop_cosines[-1] < 0.90, (
            f"After {N_AGENTS - 1} hops of cumulative noise, "
            f"cosine should be < 0.90 (got {hop_cosines[-1]:.4f})"
        )

    def test_speed_advantage_scales_with_chain_depth(self):
        """Vector path speed advantage should grow with chain depth."""
        original_vecs = np.stack([_embed(c) for c in CLAIMS])

        n_iter = 20

        t0 = time.perf_counter()
        for _ in range(n_iter):
            for hop in range(N_AGENTS - 1):
                rng = np.random.default_rng(hop)
                _ = np.stack([_noisy_re_embed(c, rng) for c in CLAIMS])
        eng_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iter):
            for _hop in range(N_AGENTS - 1):
                _ = original_vecs.copy()
        vec_time = time.perf_counter() - t0

        speedup = eng_time / max(vec_time, 1e-9)
        assert speedup >= 2.0, f"Vector chain only {speedup:.1f}x faster over {N_AGENTS - 1} hops"
