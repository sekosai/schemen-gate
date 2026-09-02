"""Architecture certification and compression/suggestion cycle for Schemen RAG.

ArchitectureSpec — AAD-bound model certification. The partition doesn't
just know *who* is accessing, it knows *what model architecture* is
accessing. The spec serializes to a topology string compatible with
the existing AdapterToken.topology field.

The compression/suggestion cycle: ingested weight snapshots are analyzed
via SVD (effective rank) and clustering (natural groups). The structure
of the stored representations suggests attention head configurations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ------------------------------------------------------------------
# ArchitectureSpec — AAD-bound model certification
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureSpec:
    """Cryptographically binds a model's architecture into adapter AAD.

    The topology string produced by ``to_topology()`` is compatible with
    ``AdapterToken.topology``.  When the adapter validates a request, it
    verifies the spec matches what's bound in the AAD.  A different model
    architecture means AES-GCM decryption fails — the architecture IS
    the identity.
    """

    backbone: str
    backbone_hash: str
    n_attention_heads: int
    head_dim: int
    n_layers: int
    gate_placements: tuple[str, ...] = ()
    custom: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_model(
        cls,
        model: Any,
        backbone: str = "unknown",
        *,
        gate_placements: Sequence[str] = (),
        custom: Optional[Dict[str, Any]] = None,
    ) -> ArchitectureSpec:
        """Build an ArchitectureSpec by inspecting a PyTorch model.

        Falls back to heuristics when the model doesn't expose standard
        attributes (n_head, num_attention_heads, etc.).
        """
        state_dict = getattr(model, "state_dict", None)
        state = state_dict() if callable(state_dict) else None

        backbone_hash = _hash_state_dict(state) if state else "no_state"

        n_heads = _detect_attr(
            model,
            (
                "n_head",
                "num_attention_heads",
                "num_heads",
                "nhead",
            ),
            default=0,
        )
        head_dim = _detect_attr(
            model,
            (
                "head_dim",
                "d_head",
            ),
            default=0,
        )
        n_layers = _detect_attr(
            model,
            (
                "n_layer",
                "num_layers",
                "num_hidden_layers",
                "n_layers",
            ),
            default=0,
        )

        if n_layers == 0 and state:
            layer_indices = set()
            for k in state.keys():
                parts = k.split(".")
                for p in parts:
                    if p.isdigit():
                        layer_indices.add(int(p))
                        break
            if layer_indices:
                n_layers = max(layer_indices) + 1

        return cls(
            backbone=backbone,
            backbone_hash=backbone_hash,
            n_attention_heads=n_heads,
            head_dim=head_dim,
            n_layers=n_layers,
            gate_placements=tuple(gate_placements),
            custom=custom or {},
        )

    def to_topology(self) -> str:
        """Serialize to a topology string for AdapterToken AAD binding."""
        d = {
            "backbone": self.backbone,
            "backbone_hash": self.backbone_hash,
            "n_attention_heads": self.n_attention_heads,
            "head_dim": self.head_dim,
            "n_layers": self.n_layers,
            "gate_placements": list(self.gate_placements),
        }
        if self.custom:
            d["custom"] = self.custom
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_topology(cls, topology: str) -> ArchitectureSpec:
        """Reconstruct from a topology string."""
        d = json.loads(topology)
        return cls(
            backbone=d["backbone"],
            backbone_hash=d["backbone_hash"],
            n_attention_heads=d["n_attention_heads"],
            head_dim=d["head_dim"],
            n_layers=d["n_layers"],
            gate_placements=tuple(d.get("gate_placements", ())),
            custom=d.get("custom", {}),
        )

    def verify(self, other: ArchitectureSpec) -> bool:
        """Check if another spec matches this one (same architecture)."""
        return self.to_topology() == other.to_topology()

    def fingerprint(self) -> str:
        """SHA-256 of the topology string."""
        return hashlib.sha256(self.to_topology().encode()).hexdigest()


# ------------------------------------------------------------------
# PartitionAnalysis — output of analyze_partition
# ------------------------------------------------------------------


@dataclass
class PartitionAnalysis:
    """Analysis of a partition's representational structure."""

    n_vectors: int
    effective_rank: int
    n_clusters: int
    cluster_dims: List[int]
    compression_ratio: float
    suggested_n_heads: int
    suggested_head_dim: int


@dataclass
class CompressionResult:
    """Output of compress_partition."""

    original_count: int
    compressed_count: int
    compression_ratio: float
    suggested_architecture: PartitionAnalysis
    stored_ids: List[str]


# ------------------------------------------------------------------
# Analysis and compression functions
# ------------------------------------------------------------------


def analyze_vectors(
    vectors: np.ndarray,
    *,
    max_clusters: int = 16,
    rank_threshold: float = 0.01,
) -> PartitionAnalysis:
    """Analyze an array of vectors (rows = vectors).

    Uses SVD for effective rank and k-means for clustering.
    Scikit-learn is a lazy import — only needed for the analysis cycle.

    Parameters
    ----------
    vectors : (N, D) array
    max_clusters : upper bound on cluster search
    rank_threshold : singular value ratio cutoff for effective rank
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n, d = vectors.shape

    if n < 2:
        return PartitionAnalysis(
            n_vectors=n,
            effective_rank=min(n, d),
            n_clusters=1,
            cluster_dims=[d],
            compression_ratio=1.0,
            suggested_n_heads=1,
            suggested_head_dim=d,
        )

    # --- effective rank via SVD ---
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    try:
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        s = np.ones(min(n, d))

    s_ratio = s / (s[0] + 1e-12)
    effective_rank = int(np.sum(s_ratio > rank_threshold))
    effective_rank = max(1, effective_rank)

    # --- clustering ---
    n_clusters, cluster_dims = _cluster_analysis(
        vectors,
        max_clusters=max_clusters,
        rank_threshold=rank_threshold,
    )

    suggested_head_dim = int(np.mean(cluster_dims)) if cluster_dims else d
    suggested_head_dim = max(1, suggested_head_dim)

    compression_ratio = effective_rank / d if d > 0 else 1.0

    return PartitionAnalysis(
        n_vectors=n,
        effective_rank=effective_rank,
        n_clusters=n_clusters,
        cluster_dims=cluster_dims,
        compression_ratio=compression_ratio,
        suggested_n_heads=n_clusters,
        suggested_head_dim=suggested_head_dim,
    )


def compress_vectors(
    vectors: np.ndarray,
    target_ratio: float = 0.5,
) -> tuple[np.ndarray, PartitionAnalysis]:
    """Compress vectors via truncated SVD, preserving gated structure.

    Returns compressed vectors and the analysis.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    n, d = vectors.shape

    analysis = analyze_vectors(vectors)
    target_dims = max(1, int(d * target_ratio))
    target_dims = min(target_dims, min(n, d))

    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    try:
        U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return vectors.copy(), analysis

    U_trunc = U[:, :target_dims]
    s_trunc = s[:target_dims]
    Vt_trunc = Vt[:target_dims, :]

    compressed = U_trunc * s_trunc[np.newaxis, :] @ Vt_trunc + mean
    analysis.compression_ratio = target_dims / d

    return compressed, analysis


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _hash_state_dict(state: dict[str, Any]) -> str:
    """SHA-256 over names, dtype, shape, and exact parameter bytes."""
    h = hashlib.sha256()
    for key in sorted(state.keys()):
        key_bytes = key.encode("utf-8")
        h.update(len(key_bytes).to_bytes(8, "big"))
        h.update(key_bytes)
        param = state[key]
        if not hasattr(param, "shape"):
            raise ValueError(f"state entry {key!r} is not tensor-like")
        shape_bytes = json.dumps(list(param.shape), separators=(",", ":")).encode()
        dtype_bytes = str(getattr(param, "dtype", "unknown")).encode()
        h.update(len(dtype_bytes).to_bytes(8, "big"))
        h.update(dtype_bytes)
        h.update(len(shape_bytes).to_bytes(8, "big"))
        h.update(shape_bytes)
        if hasattr(param, "detach"):
            tensor = param.detach().cpu().contiguous()
            try:
                value_bytes = tensor.numpy().tobytes(order="C")
            except TypeError:
                import torch

                value_bytes = tensor.view(torch.uint8).numpy().tobytes(order="C")
        else:
            value_bytes = np.ascontiguousarray(param).tobytes(order="C")
        h.update(len(value_bytes).to_bytes(8, "big"))
        h.update(value_bytes)
    return h.hexdigest()


def _detect_attr(obj: Any, names: tuple[str, ...], default: int = 0) -> int:
    """Try multiple attribute names on an object or its .config."""
    for name in names:
        val = getattr(obj, name, None)
        if type(val) is int:
            return int(val)
    config = getattr(obj, "config", None)
    if config is not None:
        for name in names:
            val = getattr(config, name, None)
            if type(val) is int:
                return int(val)
    return default


def _cluster_analysis(
    vectors: np.ndarray,
    max_clusters: int = 16,
    rank_threshold: float = 0.01,
) -> tuple[int, list[int]]:
    """Find natural cluster count and per-cluster effective dimensionality.

    Uses k-means with silhouette score from scikit-learn. Falls back
    to simple heuristics if sklearn is unavailable.
    """
    n, d = vectors.shape
    max_k = min(max_clusters, n - 1) if n > 2 else 1

    if max_k < 2:
        return 1, [d]

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        # Fallback: estimate clusters from SVD spectrum
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return 1, [d]
        gaps = np.diff(s / (s[0] + 1e-12))
        if len(gaps) > 0:
            n_clusters = int(np.argmin(gaps) + 1)
            n_clusters = max(1, min(n_clusters, max_k))
        else:
            n_clusters = 1
        return n_clusters, [d] * n_clusters

    best_k = 2
    best_score = -1.0
    best_labels = None

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=3, random_state=42, max_iter=100)
        labels = km.fit_predict(vectors)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(vectors, labels, sample_size=min(n, 500))
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    if best_labels is None:
        return 1, [d]

    cluster_dims = []
    for c in range(best_k):
        mask = best_labels == c
        if mask.sum() < 2:
            cluster_dims.append(d)
            continue
        cluster_vecs = vectors[mask]
        centered = cluster_vecs - cluster_vecs.mean(axis=0, keepdims=True)
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            cluster_dims.append(d)
            continue
        ratio = s / (s[0] + 1e-12)
        eff = int(np.sum(ratio > rank_threshold))
        cluster_dims.append(max(1, eff))

    return best_k, cluster_dims
