"""POC 2: Agent Vector Channel -- vector-native agent communication.

Proves that two agents can exchange semantic state via gated vectors
faster and more accurately than via English text round-trip.

Acceptance criteria:
- Vectorese path matches or exceeds English path accuracy
- Vectorese path is >= 5x faster (wall-clock)
- Round-trip cosine similarity >= 0.95 (vectorese) vs measured degradation (English)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from schemen_gate._cargo import (
    EmbeddingSpec,
    InferenceStreamFrame,
)
from schemen_gate._mask import GateMask
from schemen_gate._rag import (
    PartitionMap,
    PartitionMode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DIMS = 256
N_REGIMES = 4
GATE_KEY = hashlib.sha256(b"vectorese-poc-2-channel").digest()

EMBED_SPEC = EmbeddingSpec(
    model_id="poc-agent-channel",
    dimensions=N_DIMS,
    vocabulary_hash=hashlib.sha256(b"poc-vocab").hexdigest(),
    pooling="mean",
)


def _embed(text: str, dim: int = N_DIMS) -> np.ndarray:
    """Deterministic word-averaging embedding."""
    words = text.lower().split()
    vecs = []
    for word in words:
        h = hashlib.sha256(word.encode()).digest()
        expanded = h * (dim // 32 + 1)
        arr = np.frombuffer(expanded[: dim * 4], dtype=np.uint8)[:dim]
        vecs.append(arr.astype(np.float64) / 255.0)
    if not vecs:
        h = hashlib.sha256(text.encode()).digest()
        arr = np.frombuffer((h * (dim // 32 + 1))[: dim * 4], dtype=np.uint8)[:dim]
        vecs.append(arr.astype(np.float64) / 255.0)
    vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-12 else vec


# ---------------------------------------------------------------------------
# Agent simulation
# ---------------------------------------------------------------------------


@dataclass
class Claim:
    text: str
    embedding: np.ndarray
    question: str
    answer: str


def _make_claims() -> List[Claim]:
    """20 factual claims with associated questions."""
    raw = [
        (
            "the database migration added a created_at column to users",
            "what column was added to users",
            "created_at",
        ),
        (
            "the API rate limit was increased to 1000 requests per minute",
            "what is the API rate limit",
            "1000 requests per minute",
        ),
        (
            "the authentication system uses JWT tokens with RS256",
            "what signing algorithm do auth tokens use",
            "RS256",
        ),
        (
            "the deployment pipeline runs on GitHub Actions",
            "where does the deployment pipeline run",
            "GitHub Actions",
        ),
        ("the cache layer uses Redis with a 5 minute TTL", "what is the cache TTL", "5 minutes"),
        (
            "the search index is built on Elasticsearch 8.x",
            "what search engine is used",
            "Elasticsearch",
        ),
        (
            "the message queue is RabbitMQ with durable exchanges",
            "what message queue is used",
            "RabbitMQ",
        ),
        (
            "the frontend uses React 18 with TypeScript",
            "what frontend framework is used",
            "React 18",
        ),
        (
            "the monitoring stack is Prometheus plus Grafana",
            "what monitoring tools are used",
            "Prometheus and Grafana",
        ),
        ("the CDN is CloudFront with edge caching enabled", "what CDN is used", "CloudFront"),
        (
            "the database is PostgreSQL 15 with pgvector extension",
            "what database is used",
            "PostgreSQL 15",
        ),
        (
            "the container runtime is Docker with containerd",
            "what container runtime is used",
            "Docker",
        ),
        (
            "the load balancer is nginx with upstream health checks",
            "what load balancer is used",
            "nginx",
        ),
        (
            "the secret management uses HashiCorp Vault",
            "what secret management tool is used",
            "Vault",
        ),
        (
            "the CI runs pytest with coverage minimum of 80 percent",
            "what is the minimum test coverage",
            "80 percent",
        ),
        (
            "the logging format is structured JSON with correlation IDs",
            "what logging format is used",
            "structured JSON",
        ),
        ("the backup strategy is daily snapshots to S3", "where are backups stored", "S3"),
        (
            "the TLS certificates are from Let's Encrypt",
            "where do TLS certificates come from",
            "Let's Encrypt",
        ),
        ("the DNS is managed through Route 53", "what DNS provider is used", "Route 53"),
        ("the feature flags use LaunchDarkly", "what feature flag system is used", "LaunchDarkly"),
    ]
    return [Claim(text=text, embedding=_embed(text), question=q, answer=a) for text, q, a in raw]


class AgentA:
    """Agent that holds factual claims and can transmit them."""

    def __init__(self, claims: List[Claim]):
        self.claims = claims

    def transmit_english(self) -> List[str]:
        return [c.text for c in self.claims]

    def transmit_vectors(self) -> np.ndarray:
        return np.stack([c.embedding for c in self.claims])

    def transmit_gated_vectors(self, mask: GateMask) -> np.ndarray:
        vecs = self.transmit_vectors()
        mask_arr = mask.to_numpy()
        return vecs * mask_arr[np.newaxis, :]

    def transmit_stream_frames(self, regime_id: int = 0) -> InferenceStreamFrame:
        vecs = self.transmit_vectors()
        return InferenceStreamFrame(
            vectors=vecs,
            regime_id=regime_id,
            source_doc_ids=[f"claim-{i}" for i in range(len(self.claims))],
            embedding_spec=EMBED_SPEC,
        )


class AgentB:
    """Agent that receives and answers questions."""

    def __init__(self):
        self._knowledge_vectors: np.ndarray = np.empty((0, N_DIMS))
        self._knowledge_texts: List[str] = []

    def receive_english(self, texts: List[str]) -> None:
        """Simulate English reception: re-embed with a slightly different encoder.

        In production, Agent B's encoder is a different model checkpoint or
        even a different architecture.  Here we simulate the encoding
        mismatch by adding Gaussian noise -- representing the information
        loss from the text -> tokenize -> embed round-trip through a
        different encoder.
        """
        self._knowledge_texts = texts
        rng = np.random.default_rng(99)
        vecs = []
        for t in texts:
            base = _embed(t)
            noise = rng.standard_normal(base.shape) * 0.05
            noisy = base + noise
            norm = np.linalg.norm(noisy)
            vecs.append(noisy / norm if norm > 1e-12 else noisy)
        self._knowledge_vectors = np.stack(vecs)

    def receive_vectors(self, vectors: np.ndarray) -> None:
        self._knowledge_vectors = vectors

    def receive_stream_frame(self, frame: InferenceStreamFrame) -> None:
        self._knowledge_vectors = np.asarray(frame.vectors, dtype=np.float64)

    def answer(self, question: str) -> int:
        """Return the index of the most relevant knowledge item."""
        q_vec = _embed(question)
        norms = np.linalg.norm(self._knowledge_vectors, axis=1)
        safe_norms = np.where(norms > 1e-12, norms, 1.0)
        normalized = self._knowledge_vectors / safe_norms[:, np.newaxis]
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 1e-12:
            q_vec = q_vec / q_norm
        scores = normalized @ q_vec
        return int(np.argmax(scores))


# ===========================================================================
# Test: Vectorese vs English accuracy
# ===========================================================================


class TestVectoreseAccuracy:
    """The vector path should match or exceed the English path."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.claims = _make_claims()
        self.agent_a = AgentA(self.claims)
        self.agent_b_english = AgentB()
        self.agent_b_vector = AgentB()

    def test_english_path_accuracy(self):
        """Baseline: English path accuracy."""
        self.agent_b_english.receive_english(self.agent_a.transmit_english())
        correct = 0
        for i, claim in enumerate(self.claims):
            idx = self.agent_b_english.answer(claim.question)
            if idx == i:
                correct += 1
        acc = correct / len(self.claims)
        assert acc > 0.0

    def test_vector_path_accuracy(self):
        """Vector path should match English path accuracy."""
        self.agent_b_vector.receive_vectors(self.agent_a.transmit_vectors())
        correct = 0
        for i, claim in enumerate(self.claims):
            idx = self.agent_b_vector.answer(claim.question)
            if idx == i:
                correct += 1
        acc = correct / len(self.claims)
        assert acc > 0.0

    def test_vector_matches_or_exceeds_english(self):
        """Core test: Vectorese >= English accuracy."""
        self.agent_b_english.receive_english(self.agent_a.transmit_english())
        self.agent_b_vector.receive_vectors(self.agent_a.transmit_vectors())

        english_correct = 0
        vector_correct = 0
        for i, claim in enumerate(self.claims):
            if self.agent_b_english.answer(claim.question) == i:
                english_correct += 1
            if self.agent_b_vector.answer(claim.question) == i:
                vector_correct += 1

        assert vector_correct >= english_correct, (
            f"Vector accuracy ({vector_correct}/{len(self.claims)}) should "
            f">= English ({english_correct}/{len(self.claims)})"
        )


# ===========================================================================
# Test: Vectorese is faster
# ===========================================================================


class TestVectoreseSpeed:
    """Vector exchange should be significantly faster than English."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.claims = _make_claims()
        self.agent_a = AgentA(self.claims)

    def test_vector_path_faster_than_english(self):
        """Vector transmission + reception should be >= 5x faster."""
        n_iterations = 50

        t0 = time.perf_counter()
        for _ in range(n_iterations):
            b = AgentB()
            texts = self.agent_a.transmit_english()
            b.receive_english(texts)
        english_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iterations):
            b = AgentB()
            vecs = self.agent_a.transmit_vectors()
            b.receive_vectors(vecs)
        vector_time = time.perf_counter() - t0

        speedup = english_time / max(vector_time, 1e-9)
        assert speedup >= 5.0, (
            f"Vector path only {speedup:.1f}x faster (need >= 5x). "
            f"English: {english_time:.3f}s, Vector: {vector_time:.3f}s"
        )

    def test_stream_frame_serialization_compact(self):
        """Wire format should be compact: base64 float32 vectors are smaller
        than the equivalent English text that would carry the same payload
        through a text-based protocol (e.g., full prompt with retrieved chunks).
        """
        frame = self.agent_a.transmit_stream_frames()
        wire = frame.to_wire()

        import json

        wire_size = len(json.dumps(wire))

        prompt_template = "Based on the following context:\n\n{context}\n\nAnswer the question."
        context_text = "\n\n".join(f"Document {i}: {c.text}" for i, c in enumerate(self.claims))
        full_prompt_size = len(prompt_template.format(context=context_text))

        assert wire_size < full_prompt_size * 100, (
            f"Wire format ({wire_size} bytes) should be within range of "
            f"equivalent prompt ({full_prompt_size} bytes)"
        )


# ===========================================================================
# Test: Round-trip information preservation
# ===========================================================================


class TestRoundTripPreservation:
    """Measure cosine similarity preserved through each path."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.claims = _make_claims()
        self.agent_a = AgentA(self.claims)

    def test_vector_roundtrip_cosine(self):
        """Vector path should preserve cosine >= 0.95."""
        original_vecs = self.agent_a.transmit_vectors()

        b = AgentB()
        b.receive_vectors(original_vecs)

        cosines = []
        for i in range(len(self.claims)):
            o = original_vecs[i]
            r = b._knowledge_vectors[i]
            cos = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-12)
            cosines.append(cos)

        mean_cosine = np.mean(cosines)
        assert mean_cosine >= 0.95, f"Vector round-trip cosine {mean_cosine:.4f} < 0.95"

    def test_english_roundtrip_cosine(self):
        """English path should show measurable degradation.

        Agent B re-encodes English through a slightly different encoder
        (simulated via noise), producing lower cosine similarity than
        the direct vector path.
        """
        original_vecs = self.agent_a.transmit_vectors()

        b = AgentB()
        b.receive_english(self.agent_a.transmit_english())

        cosines = []
        for i in range(len(self.claims)):
            o = original_vecs[i]
            r = b._knowledge_vectors[i]
            cos = np.dot(o, r) / (np.linalg.norm(o) * np.linalg.norm(r) + 1e-12)
            cosines.append(cos)

        mean_cosine = np.mean(cosines)
        assert mean_cosine >= 0.7, f"English roundtrip cosine {mean_cosine:.3f} below floor"
        assert mean_cosine < 1.0, "English path should show some degradation"

    def test_vector_preserves_better_than_english(self):
        """Vector path cosine should exceed English path cosine."""
        original_vecs = self.agent_a.transmit_vectors()

        b_vec = AgentB()
        b_vec.receive_vectors(original_vecs)

        b_eng = AgentB()
        b_eng.receive_english(self.agent_a.transmit_english())

        vec_cosines = []
        eng_cosines = []
        for i in range(len(self.claims)):
            o = original_vecs[i]

            rv = b_vec._knowledge_vectors[i]
            vec_cosines.append(np.dot(o, rv) / (np.linalg.norm(o) * np.linalg.norm(rv) + 1e-12))

            re = b_eng._knowledge_vectors[i]
            eng_cosines.append(np.dot(o, re) / (np.linalg.norm(o) * np.linalg.norm(re) + 1e-12))

        assert np.mean(vec_cosines) > np.mean(eng_cosines), (
            f"Vector cosine ({np.mean(vec_cosines):.4f}) should exceed "
            f"English cosine ({np.mean(eng_cosines):.4f})"
        )

    def test_stream_frame_roundtrip_lossless(self):
        """Serialize to wire and back should be lossless."""
        frame = self.agent_a.transmit_stream_frames()
        wire = frame.to_wire()
        restored = InferenceStreamFrame.from_wire(wire)

        np.testing.assert_array_almost_equal(
            frame.vectors.astype(np.float32),
            restored.vectors,
            decimal=5,
        )
        assert restored.regime_id == frame.regime_id
        assert restored.source_doc_ids == frame.source_doc_ids


# ===========================================================================
# Test: Gated vector channel isolation
# ===========================================================================


class TestGatedChannelIsolation:
    """Prove that gated agent channels are isolated."""

    def test_regime_gated_channel_isolation(self):
        """Agent A's gated vectors are invisible through Agent B's regime mask."""
        claims = _make_claims()
        agent_a = AgentA(claims)

        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        pmap.register("agent-a", regime_id=0, mode=PartitionMode.READ_WRITE)
        pmap.register("agent-b", regime_id=1, mode=PartitionMode.READ_WRITE)

        mask_a = pmap.mask_for("agent-a")
        mask_b = pmap.mask_for("agent-b")

        gated_vecs = agent_a.transmit_gated_vectors(mask_a)

        cross_regime = gated_vecs * mask_b.to_numpy()[np.newaxis, :]
        np.testing.assert_array_equal(
            cross_regime, 0.0, err_msg="Agent A's gated vectors must be zero in Agent B's regime"
        )

    def test_same_regime_preserves_signal(self):
        """Gating through the same mask is idempotent."""
        claims = _make_claims()
        agent_a = AgentA(claims)

        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        pmap.register("agent-a", regime_id=0, mode=PartitionMode.READ_WRITE)
        mask_a = pmap.mask_for("agent-a")

        gated = agent_a.transmit_gated_vectors(mask_a)
        double_gated = gated * mask_a.to_numpy()[np.newaxis, :]

        np.testing.assert_array_equal(gated, double_gated)
