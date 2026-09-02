"""Lossless row folding for full-dimensional vector reconstitution.

When a regime has n_dim / R dimensions, it cannot represent the full n_dim
output directly. The folding codec splits the full vector into R chunks of
``n_dim / R`` values and reconstructs it by concatenation.

Folding is a lossless representation transform, not a security boundary. It
does not apply a GateMask, authorize storage, or confine plaintext coordinates.
Callers must enforce those controls separately before persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FoldedRepresentation:
    """A full-dimensional vector folded into R rows of regime-width chunks."""

    rows: np.ndarray
    """Shape (R, dims_per_regime) -- the folded matrix."""

    n_dims: int
    """Original full dimensionality."""

    n_regimes: int
    """Number of regimes (R)."""

    source_regime_id: int
    """Intended storage-regime label; metadata only, not authorization evidence."""

    @property
    def dims_per_regime(self) -> int:
        return self.n_dims // self.n_regimes

    @property
    def n_rows(self) -> int:
        return int(self.rows.shape[0])


def fold_vector(
    vector: np.ndarray,
    n_regimes: int,
) -> FoldedRepresentation:
    """Losslessly encode a vector as R rows of ``n_dim // R`` values.

    This codec is not a security boundary: it does not apply a mask, authorize
    a write, or prove that a store enforces any regime boundary.
    """
    vector = np.asarray(vector, dtype=np.float64).ravel()
    n_dims = vector.shape[0]

    if n_dims % n_regimes != 0:
        raise ValueError(f"n_dims ({n_dims}) must be divisible by n_regimes ({n_regimes})")

    dims_per_regime = n_dims // n_regimes
    rows = vector.reshape(n_regimes, dims_per_regime).copy()

    return FoldedRepresentation(
        rows=rows,
        n_dims=n_dims,
        n_regimes=n_regimes,
        source_regime_id=0,
    )


def unfold_vector(folded: FoldedRepresentation) -> np.ndarray:
    """Reconstruct the full n_dim vector from R folded rows.

    Concatenates the rows back into a single vector of the original
    dimensionality.
    """
    return folded.rows.ravel().copy()


def fold_matrix(
    matrix: np.ndarray,
    n_regimes: int,
) -> list[FoldedRepresentation]:
    """Fold each row of a matrix independently.

    Input shape: (batch, n_dims)
    Returns: list of FoldedRepresentation, one per batch row.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        return [fold_vector(matrix, n_regimes)]
    return [fold_vector(row, n_regimes) for row in matrix]


def unfold_matrix(folded_rows: list[FoldedRepresentation]) -> np.ndarray:
    """Reconstruct a matrix from a list of folded representations."""
    return np.stack([unfold_vector(f) for f in folded_rows])


def reconstruction_quality(
    original: np.ndarray,
    reconstructed: np.ndarray,
) -> dict[str, float]:
    """Measure how well a reconstructed vector matches the original.

    Returns cosine similarity, L2 distance, and relative error.
    """
    original = np.asarray(original, dtype=np.float64).ravel()
    reconstructed = np.asarray(reconstructed, dtype=np.float64).ravel()

    norm_o = np.linalg.norm(original)
    norm_r = np.linalg.norm(reconstructed)

    if norm_o < 1e-12 or norm_r < 1e-12:
        return {
            "cosine_similarity": 0.0,
            "l2_distance": float("inf"),
            "relative_error": float("inf"),
        }

    cosine = float(np.dot(original, reconstructed) / (norm_o * norm_r))
    l2 = float(np.linalg.norm(original - reconstructed))
    relative = float(l2 / norm_o)

    return {
        "cosine_similarity": cosine,
        "l2_distance": l2,
        "relative_error": relative,
    }
