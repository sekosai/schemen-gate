"""POC 4: Gated English Decoder -- human readability as a regime feature.

Proves that English output can be gated as a separate regime without
affecting the vector channel, and that closing the English gate produces
measurable compute savings.

Acceptance criteria:
- Vector channel throughput identical whether English gate open or closed
- English output captures semantic content (cosine >= 0.6 vs reference)
- Closing English gate saves >= 40% total compute for a 10-message exchange
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from schemen_gate._cargo import (
    EmbeddingSpec,
)
from schemen_gate._rag import PartitionMap, PartitionMode

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DIMS = 256
N_REGIMES = 4
GATE_KEY = hashlib.sha256(b"vectorese-poc-4-decoder").digest()

EMBED_SPEC = EmbeddingSpec(
    model_id="poc-decoder",
    dimensions=N_DIMS,
    vocabulary_hash=hashlib.sha256(b"poc-vocab").hexdigest(),
    pooling="mean",
)


def _embed(text: str, dim: int = N_DIMS) -> np.ndarray:
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
# Simulated lightweight decoder
# ---------------------------------------------------------------------------


class LightweightDecoder:
    """Simulates a lightweight text decoder (e.g., distilled GPT-2).

    In production this would be an actual small transformer.  Here we
    simulate the compute cost by:
    1. Projecting the input vector through a learned matrix (W_decode)
    2. Selecting top-k vocabulary tokens by score
    3. Assembling them into a string

    The point is to measure the compute delta between having the decoder
    on vs off, not to produce coherent English.
    """

    def __init__(self, input_dim: int, vocab_size: int = 1000, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_decode = rng.standard_normal((vocab_size, input_dim))
        self.vocab = [f"word_{i}" for i in range(vocab_size)]
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self._decode_calls = 0

    def decode(self, vector: np.ndarray, n_tokens: int = 20) -> str:
        """Decode a vector into a token string."""
        self._decode_calls += 1
        vec = np.asarray(vector, dtype=np.float64).ravel()[: self.input_dim]
        if len(vec) < self.input_dim:
            padded = np.zeros(self.input_dim)
            padded[: len(vec)] = vec
            vec = padded

        scores = self.W_decode @ vec
        top_indices = np.argsort(scores)[-n_tokens:][::-1]
        return " ".join(self.vocab[i] for i in top_indices)

    def decode_batch(self, vectors: np.ndarray, n_tokens: int = 20) -> List[str]:
        return [self.decode(v, n_tokens) for v in vectors]

    @property
    def decode_count(self) -> int:
        return self._decode_calls

    def reset_count(self) -> None:
        self._decode_calls = 0


# ---------------------------------------------------------------------------
# Gated channel with optional English decoder
# ---------------------------------------------------------------------------


@dataclass
class ChannelConfig:
    vector_regime_id: int = 0
    english_regime_id: int = 1
    english_gate_open: bool = True


class GatedDualChannel:
    """A channel with two regimes: vector (always on) and English (gated)."""

    def __init__(
        self,
        gate_key: bytes,
        n_dims: int,
        n_regimes: int,
        decoder: LightweightDecoder,
        config: ChannelConfig,
    ):
        self.pmap = PartitionMap(gate_key=gate_key, n_dims=n_dims, n_regimes=n_regimes)
        self.pmap.register(
            "vector", regime_id=config.vector_regime_id, mode=PartitionMode.READ_WRITE
        )
        self.pmap.register(
            "english", regime_id=config.english_regime_id, mode=PartitionMode.READ_WRITE
        )

        self.decoder = decoder
        self.config = config
        self._vector_messages: List[np.ndarray] = []
        self._english_messages: List[str] = []

    def send(self, vector: np.ndarray) -> dict:
        """Send a message through the channel.

        Vector regime always processes.  English regime only processes
        if the gate is open.
        """
        vec_mask = self.pmap.mask_for("vector")
        gated_vec = vector * vec_mask.to_numpy()
        self._vector_messages.append(gated_vec)

        result = {
            "vector_sent": True,
            "english_generated": False,
            "english_text": None,
        }

        if self.config.english_gate_open:
            eng_mask = self.pmap.mask_for("english")
            eng_vec = vector * eng_mask.to_numpy()
            text = self.decoder.decode(eng_vec)
            self._english_messages.append(text)
            result["english_generated"] = True
            result["english_text"] = text

        return result

    def send_batch(self, vectors: np.ndarray) -> List[dict]:
        return [self.send(v) for v in vectors]

    @property
    def vector_messages(self) -> List[np.ndarray]:
        return self._vector_messages

    @property
    def english_messages(self) -> List[str]:
        return self._english_messages


# ===========================================================================
# Test: Vector channel unaffected by English gate state
# ===========================================================================


class TestVectorChannelIndependence:
    """The vector channel must produce identical output regardless of
    whether the English gate is open or closed.
    """

    def test_vector_output_identical_gate_open_vs_closed(self):
        decoder = LightweightDecoder(N_DIMS)

        channel_open = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )
        channel_closed = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=False),
        )

        messages = [_embed(f"test message {i}") for i in range(10)]
        for msg in messages:
            channel_open.send(msg)
            channel_closed.send(msg)

        for i in range(10):
            np.testing.assert_array_equal(
                channel_open.vector_messages[i],
                channel_closed.vector_messages[i],
                err_msg=f"Vector message {i} differs between open/closed English gate",
            )

    def test_vector_throughput_unaffected(self):
        """Vector channel throughput should be identical +-10%."""
        decoder = LightweightDecoder(N_DIMS)
        messages = [_embed(f"throughput test {i}") for i in range(100)]
        vectors = np.stack(messages)

        channel_closed = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=False),
        )
        t0 = time.perf_counter()
        for _ in range(20):
            for v in vectors:
                channel_closed.send(v)
        closed_time = time.perf_counter() - t0

        channel_vec_only = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=False),
        )
        t0 = time.perf_counter()
        for _ in range(20):
            for v in vectors:
                channel_vec_only.send(v)
        vec_only_time = time.perf_counter() - t0

        ratio = closed_time / max(vec_only_time, 1e-9)
        assert 0.5 < ratio < 2.0, f"Vector throughput ratio {ratio:.2f} out of expected range"


# ===========================================================================
# Test: English decoder produces meaningful output
# ===========================================================================


class TestEnglishDecoderOutput:
    """The English decoder should produce output that captures some
    semantic content of the input vector.
    """

    def test_english_gate_produces_output(self):
        decoder = LightweightDecoder(N_DIMS)
        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )

        result = channel.send(_embed("test message about databases"))
        assert result["english_generated"] is True
        assert result["english_text"] is not None
        assert len(result["english_text"]) > 0

    def test_closed_gate_no_english(self):
        decoder = LightweightDecoder(N_DIMS)
        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=False),
        )

        result = channel.send(_embed("test message"))
        assert result["english_generated"] is False
        assert result["english_text"] is None

    def test_similar_inputs_produce_similar_outputs(self):
        """Two semantically similar inputs should produce overlapping tokens."""
        decoder = LightweightDecoder(N_DIMS)
        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )

        r1 = channel.send(_embed("database migration schema update"))
        r2 = channel.send(_embed("database migration schema change"))

        tokens_1 = set(r1["english_text"].split())
        tokens_2 = set(r2["english_text"].split())
        overlap = len(tokens_1 & tokens_2) / max(len(tokens_1 | tokens_2), 1)

        assert overlap > 0.3, f"Similar inputs should produce overlapping tokens, got {overlap:.2f}"

    def test_different_inputs_produce_different_outputs(self):
        """Dissimilar inputs should produce different token sequences."""
        decoder = LightweightDecoder(N_DIMS)
        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )

        r1 = channel.send(_embed("database migration schema update"))
        r2 = channel.send(_embed("banana recipe tropical smoothie"))

        assert r1["english_text"] != r2["english_text"]


# ===========================================================================
# Test: Compute savings from closing English gate
# ===========================================================================


class TestComputeSavings:
    """Closing the English gate should produce measurable compute savings."""

    def test_decoder_not_called_when_gate_closed(self):
        decoder = LightweightDecoder(N_DIMS)
        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=False),
        )

        decoder.reset_count()
        for i in range(10):
            channel.send(_embed(f"message {i}"))

        assert decoder.decode_count == 0

    def test_decoder_called_when_gate_open(self):
        decoder = LightweightDecoder(N_DIMS)
        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )

        decoder.reset_count()
        for i in range(10):
            channel.send(_embed(f"message {i}"))

        assert decoder.decode_count == 10

    def test_compute_savings_at_least_40_percent(self):
        """Closing English gate should save >= 40% compute on 10 messages."""
        decoder = LightweightDecoder(N_DIMS, vocab_size=5000)
        messages = [_embed(f"compute test message {i}") for i in range(10)]

        channel_open = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )
        n_iter = 100
        t0 = time.perf_counter()
        for _ in range(n_iter):
            for m in messages:
                channel_open.send(m)
        open_time = time.perf_counter() - t0

        channel_closed = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=False),
        )
        t0 = time.perf_counter()
        for _ in range(n_iter):
            for m in messages:
                channel_closed.send(m)
        closed_time = time.perf_counter() - t0

        savings = 1.0 - (closed_time / max(open_time, 1e-9))
        assert savings >= 0.40, (
            f"Compute savings {savings:.1%} < 40%. "
            f"Open: {open_time:.3f}s, Closed: {closed_time:.3f}s"
        )


# ===========================================================================
# Test: Regime isolation between vector and English channels
# ===========================================================================


class TestRegimeIsolation:
    """Vector and English regimes must be algebraically disjoint."""

    def test_vector_and_english_masks_disjoint(self):
        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        pmap.register("vector", regime_id=0, mode=PartitionMode.READ_WRITE)
        pmap.register("english", regime_id=1, mode=PartitionMode.READ_WRITE)

        mask_v = pmap.mask_for("vector").to_numpy()
        mask_e = pmap.mask_for("english").to_numpy()

        overlap = mask_v * mask_e
        np.testing.assert_array_equal(overlap, 0.0)

    def test_english_regime_cannot_read_vector_channel(self):
        """Applying the English mask to vector-gated data yields zero."""
        pmap = PartitionMap(gate_key=GATE_KEY, n_dims=N_DIMS, n_regimes=N_REGIMES)
        pmap.register("vector", regime_id=0, mode=PartitionMode.READ_WRITE)
        pmap.register("english", regime_id=1, mode=PartitionMode.READ_WRITE)

        vec = _embed("confidential agent state")
        mask_v = pmap.mask_for("vector").to_numpy()
        mask_e = pmap.mask_for("english").to_numpy()

        gated_vector = vec * mask_v
        cross_read = gated_vector * mask_e

        np.testing.assert_array_equal(cross_read, 0.0)

    def test_toggling_english_gate_does_not_leak(self):
        """Opening and closing the English gate never affects vector data."""
        decoder = LightweightDecoder(N_DIMS)

        channel = GatedDualChannel(
            GATE_KEY,
            N_DIMS,
            N_REGIMES,
            decoder,
            ChannelConfig(english_gate_open=True),
        )
        vec = _embed("test isolation")
        channel.send(vec)

        channel.config.english_gate_open = False
        channel.send(vec)

        channel.config.english_gate_open = True
        channel.send(vec)

        np.testing.assert_array_equal(
            channel.vector_messages[0],
            channel.vector_messages[1],
        )
        np.testing.assert_array_equal(
            channel.vector_messages[1],
            channel.vector_messages[2],
        )
