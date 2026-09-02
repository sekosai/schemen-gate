"""VON -- Vector Object Notation.

Precision-gated vector transport.  Quantize aggressively for the wire,
keep the reconstruction residual as a gated key.  The dumbest
representation you can afford is still topology-preserving.  The key
per layer controls how sharp you see it.

Multi-level VON:
    L0  int8 payload (784 bytes for 768-dim) -- topology-preserving
    L1  + float16 residual key -- near-lossless
    L2  + float32 residual of residual -- lossless to float32
    L3  + float64 residual -- bit-perfect original

Each level is a residual of the previous level's approximation.
Each residual can be gated through a regime independently.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

HEADER_MAGIC = b"VON1"
HEADER_VERSION = 1
_HEADER_FORMAT = "<BHB"
_LEVEL_HEADER_FORMAT = "<BBlddI"
_MAX_WIRE_BYTES = 2 * 1024 * 1024
_EXPECTED_LEVEL_DTYPES: dict[int, np.dtype[Any]] = {
    0: np.dtype(np.uint8),
    1: np.dtype(np.float16),
    2: np.dtype(np.float32),
    3: np.dtype(np.float64),
}


@dataclass(frozen=True)
class QuantizationLevel:
    """One level of the residual quantization stack."""

    level: int
    dtype: np.dtype
    data: np.ndarray
    scale: float
    offset: float

    @property
    def size_bytes(self) -> int:
        return self.data.nbytes + 16  # 16 bytes for scale+offset header


@dataclass
class VONFrame:
    """A precision-gated vector frame.

    The payload (L0) is always present.  Reconstruction keys (L1-L3)
    are optional and can be gated through regimes.
    """

    n_dims: int
    levels: Dict[int, QuantizationLevel] = field(default_factory=dict)
    source_dtype: np.dtype = field(default=np.dtype(np.float64))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def max_level(self) -> int:
        return max(self.levels.keys()) if self.levels else -1

    @property
    def payload_size(self) -> int:
        """Size of L0 payload only."""
        if 0 not in self.levels:
            return 0
        return self.levels[0].size_bytes

    @property
    def total_size(self) -> int:
        """Total size across all levels."""
        return sum(lv.size_bytes for lv in self.levels.values())

    def reconstruct(self, max_level: Optional[int] = None) -> np.ndarray:
        """Reconstruct the vector up to the specified precision level.

        Without max_level, uses all available levels.
        """
        if not self.levels:
            raise ValueError("VONFrame has no quantization levels")

        if max_level is None:
            max_level = self.max_level

        lv0 = self.levels[0]
        result = lv0.data.astype(np.float64) * lv0.scale + lv0.offset

        for lvl in range(1, max_level + 1):
            if lvl not in self.levels:
                break
            lv = self.levels[lvl]
            residual = lv.data.astype(np.float64) * lv.scale + lv.offset
            result = result + residual

        return result

    def to_wire(self, include_levels: Optional[List[int]] = None) -> bytes:
        """Serialize to wire format.

        include_levels controls which levels are sent.  Default: all.
        """
        if (
            isinstance(self.n_dims, bool)
            or not isinstance(self.n_dims, int)
            or not 0 < self.n_dims <= 0xFFFF
        ):
            raise ValueError("VON n_dims must be an integer in [1, 65535]")
        if include_levels is None:
            include_levels = sorted(self.levels.keys())
        else:
            include_levels = list(include_levels)
        if (
            not include_levels
            or include_levels != list(range(len(include_levels)))
            or any(level not in self.levels for level in include_levels)
        ):
            raise ValueError("VON wire levels must be unique, ordered, contiguous, and start at L0")

        parts = [HEADER_MAGIC]
        parts.append(
            struct.pack(
                _HEADER_FORMAT,
                HEADER_VERSION,
                self.n_dims,
                len(include_levels),
            )
        )

        for lvl in include_levels:
            lv = self.levels[lvl]
            expected_dtype = _EXPECTED_LEVEL_DTYPES[lvl]
            level_data = np.asarray(lv.data)
            if (
                lv.level != lvl
                or np.dtype(lv.dtype) != expected_dtype
                or level_data.dtype != expected_dtype
                or level_data.ndim != 1
                or level_data.shape[0] != self.n_dims
                or not np.all(np.isfinite(level_data))
                or not np.isfinite(lv.scale)
                or not np.isfinite(lv.offset)
            ):
                raise ValueError(f"Invalid VON quantization level L{lvl}")
            dtype_id = _dtype_to_id(expected_dtype)
            data_bytes = np.ascontiguousarray(level_data).tobytes()
            parts.append(
                struct.pack(
                    _LEVEL_HEADER_FORMAT, lvl, dtype_id, 0, lv.scale, lv.offset, len(data_bytes)
                )
            )
            parts.append(data_bytes)

        wire = b"".join(parts)
        if len(wire) > _MAX_WIRE_BYTES:
            raise ValueError("VON wire frame exceeds the maximum size")
        return wire

    @classmethod
    def from_wire(cls, data: bytes) -> VONFrame:
        """Deserialize one complete, bounded VON frame and reject ambiguity."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise ValueError("VON wire frame must be bytes-like")
        raw = bytes(data)
        header_size = struct.calcsize(_HEADER_FORMAT)
        level_header_size = struct.calcsize(_LEVEL_HEADER_FORMAT)
        if len(raw) > _MAX_WIRE_BYTES:
            raise ValueError("VON wire frame exceeds the maximum size")
        if len(raw) < len(HEADER_MAGIC) + header_size:
            raise ValueError("Invalid VON wire format: truncated header")
        if raw[:4] != HEADER_MAGIC:
            raise ValueError("Invalid VON wire format: bad magic")

        offset = 4
        version, n_dims, n_levels = struct.unpack_from(_HEADER_FORMAT, raw, offset)
        offset += header_size

        if version != HEADER_VERSION:
            raise ValueError(f"Unsupported VON version: {version}")
        if n_dims == 0:
            raise ValueError("VON n_dims must be positive")
        if not 1 <= n_levels <= len(_EXPECTED_LEVEL_DTYPES):
            raise ValueError("VON frame must contain one to four levels")

        levels: Dict[int, QuantizationLevel] = {}
        for expected_level in range(n_levels):
            if len(raw) - offset < level_header_size:
                raise ValueError("Invalid VON wire format: truncated level header")
            lvl, dtype_id, pad, scale, off, data_len = struct.unpack_from(
                _LEVEL_HEADER_FORMAT, raw, offset
            )
            offset += level_header_size
            if lvl != expected_level or lvl in levels:
                raise ValueError("VON levels must be unique, ordered, contiguous, and start at L0")
            if pad != 0:
                raise ValueError("VON level reserved field must be zero")
            dtype = _id_to_dtype(dtype_id)
            if dtype != _EXPECTED_LEVEL_DTYPES[lvl]:
                raise ValueError(f"VON L{lvl} has an invalid dtype")
            if not np.isfinite(scale) or not np.isfinite(off):
                raise ValueError("VON scale and offset must be finite")
            expected_data_len = n_dims * dtype.itemsize
            if data_len != expected_data_len:
                raise ValueError(
                    f"VON L{lvl} declares {data_len} bytes; expected "
                    f"{expected_data_len} for {n_dims} dimensions"
                )
            if len(raw) - offset < data_len:
                raise ValueError("Invalid VON wire format: truncated level data")
            arr = np.frombuffer(raw, dtype=dtype, count=n_dims, offset=offset).copy()
            offset += data_len
            if not np.all(np.isfinite(arr)):
                raise ValueError("VON level data must contain only finite values")
            levels[lvl] = QuantizationLevel(
                level=lvl, dtype=dtype, data=arr, scale=scale, offset=off
            )

        if offset != len(raw):
            raise ValueError("Invalid VON wire format: trailing bytes")
        return cls(n_dims=n_dims, levels=levels)

    def payload_hash(self) -> str:
        """SHA-256 of the L0 payload for content-addressable lookup."""
        if 0 not in self.levels:
            raise ValueError("No L0 payload to hash")
        lv0 = self.levels[0]
        h = hashlib.sha256()
        h.update(struct.pack("<dd", lv0.scale, lv0.offset))
        h.update(lv0.data.tobytes())
        return h.hexdigest()


def _dtype_to_id(dtype: np.dtype[Any]) -> int:
    mapping = {
        np.dtype(np.uint8): 0,
        np.dtype(np.float16): 1,
        np.dtype(np.float32): 2,
        np.dtype(np.float64): 3,
    }
    return mapping.get(dtype, 255)


def _id_to_dtype(dtype_id: int) -> np.dtype[Any]:
    mapping: dict[int, np.dtype[Any]] = {
        0: np.dtype(np.uint8),
        1: np.dtype(np.float16),
        2: np.dtype(np.float32),
        3: np.dtype(np.float64),
    }
    if dtype_id not in mapping:
        raise ValueError(f"Unknown dtype id: {dtype_id}")
    return mapping[dtype_id]


def encode(
    vector: np.ndarray,
    *,
    max_level: int = 3,
) -> VONFrame:
    """Encode a vector into a multi-level VONFrame.

    L0: int8 scalar quantization (7.8x compression)
    L1: float16 residual (captures int8 quantization error)
    L2: float32 residual of residual
    L3: float64 residual (bit-perfect reconstruction)
    """
    if isinstance(max_level, bool) or not isinstance(max_level, int) or not 0 <= max_level <= 3:
        raise ValueError("max_level must be an integer in [0, 3]")
    source = np.asarray(vector, dtype=np.float64)
    if source.ndim != 1 or source.size == 0:
        raise ValueError("VON vectors must be non-empty and one-dimensional")
    if source.size > 0xFFFF:
        raise ValueError("VON vectors may contain at most 65535 dimensions")
    if not np.all(np.isfinite(source)):
        raise ValueError("VON vectors must contain only finite values")
    vector = np.ascontiguousarray(source)
    n_dims = vector.shape[0]
    source_dtype = np.dtype(np.float64)

    levels: Dict[int, QuantizationLevel] = {}
    current = vector.copy()

    if max_level >= 0:
        vmin, vmax = float(current.min()), float(current.max())
        span = vmax - vmin
        if not np.isfinite(span):
            raise ValueError("VON vector dynamic range is too large to quantize")
        if span < 1e-15:
            span = 1.0
            vmin -= 0.5
        scale = span / 255.0
        if not np.isfinite(scale) or not np.isfinite(vmin):
            raise ValueError("VON quantization parameters must be finite")
        quantized = np.clip(np.round((current - vmin) / scale), 0, 255).astype(np.uint8)
        levels[0] = QuantizationLevel(
            level=0,
            dtype=np.dtype(np.uint8),
            data=quantized,
            scale=scale,
            offset=vmin,
        )
        approx = quantized.astype(np.float64) * scale + vmin
        current = current - approx
        if not np.all(np.isfinite(current)):
            raise ValueError("VON L0 residual is not finite")

    if max_level >= 1:
        residual_f16 = current.astype(np.float16)
        if not np.all(np.isfinite(residual_f16)):
            raise ValueError("VON L1 residual exceeds float16 range")
        levels[1] = QuantizationLevel(
            level=1,
            dtype=np.dtype(np.float16),
            data=residual_f16,
            scale=1.0,
            offset=0.0,
        )
        current = current - residual_f16.astype(np.float64)

    if max_level >= 2:
        residual_f32 = current.astype(np.float32)
        if not np.all(np.isfinite(residual_f32)):
            raise ValueError("VON L2 residual exceeds float32 range")
        levels[2] = QuantizationLevel(
            level=2,
            dtype=np.dtype(np.float32),
            data=residual_f32,
            scale=1.0,
            offset=0.0,
        )
        current = current - residual_f32.astype(np.float64)

    if max_level >= 3:
        levels[3] = QuantizationLevel(
            level=3,
            dtype=np.dtype(np.float64),
            data=current.copy(),
            scale=1.0,
            offset=0.0,
        )

    return VONFrame(n_dims=n_dims, levels=levels, source_dtype=source_dtype)


def encode_batch(
    matrix: np.ndarray,
    *,
    max_level: int = 3,
) -> List[VONFrame]:
    """Encode each row of a matrix as a separate VONFrame."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        return [encode(matrix, max_level=max_level)]
    return [encode(row, max_level=max_level) for row in matrix]


def cosine_at_level(
    original: np.ndarray,
    frame: VONFrame,
    level: int,
) -> float:
    """Measure cosine similarity between original and reconstruction at a level."""
    reconstructed = frame.reconstruct(max_level=level)
    original = np.asarray(original, dtype=np.float64).ravel()
    no = np.linalg.norm(original)
    nr = np.linalg.norm(reconstructed)
    if no < 1e-15 or nr < 1e-15:
        return 0.0
    return float(np.dot(original, reconstructed) / (no * nr))


def fidelity_curve(
    original: np.ndarray,
    frame: VONFrame,
) -> Dict[int, float]:
    """Compute cosine similarity at each available level."""
    return {lvl: cosine_at_level(original, frame, lvl) for lvl in sorted(frame.levels.keys())}


def ranking_preservation(
    originals: np.ndarray,
    query: np.ndarray,
    level: int = 0,
) -> Dict[str, Any]:
    """Measure whether cosine similarity rankings survive quantization.

    Computes cosine rankings of originals w.r.t. query at full precision
    and at the specified VON level, then reports rank correlation.
    """
    originals = np.asarray(originals, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).ravel()

    norms_orig = np.linalg.norm(originals, axis=1)
    nq = np.linalg.norm(query)
    full_cosines = originals @ query / (norms_orig * nq + 1e-15)
    full_ranking = np.argsort(-full_cosines)

    frames = encode_batch(originals, max_level=level)
    reconstructed = np.stack([f.reconstruct(max_level=level) for f in frames])
    norms_recon = np.linalg.norm(reconstructed, axis=1)

    query_frame = encode(query, max_level=level)
    query_recon = query_frame.reconstruct(max_level=level)
    nqr = np.linalg.norm(query_recon)

    recon_cosines = reconstructed @ query_recon / (norms_recon * nqr + 1e-15)
    recon_ranking = np.argsort(-recon_cosines)

    n = len(full_ranking)
    full_rank_pos = np.empty(n, dtype=int)
    recon_rank_pos = np.empty(n, dtype=int)
    for i, idx in enumerate(full_ranking):
        full_rank_pos[idx] = i
    for i, idx in enumerate(recon_ranking):
        recon_rank_pos[idx] = i

    d_sq = np.sum((full_rank_pos - recon_rank_pos) ** 2)
    spearman = 1.0 - (6.0 * d_sq) / (n * (n * n - 1)) if n > 1 else 1.0

    top1_match = full_ranking[0] == recon_ranking[0]
    top5_overlap = len(set(full_ranking[:5]) & set(recon_ranking[:5])) / min(5, n)

    return {
        "spearman_rho": float(spearman),
        "top1_match": bool(top1_match),
        "top5_overlap": float(top5_overlap),
        "n_items": n,
    }


def strip_levels(frame: VONFrame, keep_levels: List[int]) -> VONFrame:
    """Return a new VONFrame with only the specified levels."""
    return VONFrame(
        n_dims=frame.n_dims,
        levels={k: v for k, v in frame.levels.items() if k in keep_levels},
        source_dtype=frame.source_dtype,
        metadata=frame.metadata,
    )


def gate_levels(
    frame: VONFrame,
    gate_key: bytes,
    n_regimes: int,
    *,
    level_regime_map: Optional[Dict[int, int]] = None,
    unsafe_plaintext_partition: bool = False,
) -> Dict[int, VONFrame]:
    """Gate each reconstruction level through a different regime.

    Returns a dict mapping regime_id -> VONFrame containing only the
    levels accessible to that regime.

    level_regime_map: maps VON level -> regime_id.
    Default: L0 -> regime 0 (public), L1 -> regime 1, L2 -> regime 2, etc.
    """
    if not unsafe_plaintext_partition:
        raise ValueError(
            "gate_levels cannot provide cryptographic access control: it partitions "
            "plaintext and does not use gate_key. Pass unsafe_plaintext_partition=True "
            "only for non-security experiments."
        )
    if isinstance(n_regimes, bool) or not isinstance(n_regimes, int) or n_regimes <= 0:
        raise ValueError("n_regimes must be a positive integer")
    if level_regime_map is None:
        level_regime_map = {lvl: lvl for lvl in frame.levels}

    regime_to_levels: Dict[int, List[int]] = {}
    for lvl, regime_id in level_regime_map.items():
        if lvl in frame.levels:
            if regime_id < 0 or regime_id >= n_regimes:
                raise ValueError(f"regime_id {regime_id} outside [0, {n_regimes})")
            regime_to_levels.setdefault(regime_id, []).append(lvl)

    result: Dict[int, VONFrame] = {}
    for regime_id, lvls in regime_to_levels.items():
        result[regime_id] = strip_levels(frame, lvls)

    return result


def compression_ratio(frame: VONFrame, level: int = 0) -> float:
    """Compression ratio at a given level vs full float64."""
    full_size = frame.n_dims * 8  # float64
    level_size = sum(frame.levels[k].size_bytes for k in sorted(frame.levels.keys()) if k <= level)
    if level_size == 0:
        return float("inf")
    return full_size / level_size
