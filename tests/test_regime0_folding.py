"""POC 3: lossless row folding for full-dimensional reconstitution.

Measures whether encoding a full n_dim vector across R compact rows allows
full reconstitution after unfold and preserves downstream task accuracy. This
codec is not a storage authorization or masking boundary.

Acceptance criteria:
- Reconstruction cosine similarity >= 0.98
- Downstream accuracy degradation <= 1% absolute
"""

from __future__ import annotations

import numpy as np
import pytest

from schemen_gate._regime0_fold import (
    fold_matrix,
    fold_vector,
    reconstruction_quality,
    unfold_matrix,
    unfold_vector,
)

N_DIMS = 768
N_REGIMES = 8
DIMS_PER_REGIME = N_DIMS // N_REGIMES  # 96
RNG = np.random.default_rng(42)


def _random_vector(dim: int = N_DIMS) -> np.ndarray:
    vec = RNG.standard_normal(dim)
    return vec / np.linalg.norm(vec)


def _random_matrix(n_rows: int, dim: int = N_DIMS) -> np.ndarray:
    mat = RNG.standard_normal((n_rows, dim))
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-12)


# ===========================================================================
# Test: Basic fold/unfold round-trip
# ===========================================================================


class TestFoldUnfoldRoundTrip:
    """Prove that fold -> unfold is a lossless round-trip."""

    def test_single_vector_roundtrip(self):
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)
        reconstructed = unfold_vector(folded)

        np.testing.assert_array_almost_equal(vec, reconstructed)

    def test_roundtrip_cosine_is_one(self):
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)
        reconstructed = unfold_vector(folded)

        quality = reconstruction_quality(vec, reconstructed)
        assert quality["cosine_similarity"] > 0.9999

    def test_folded_shape(self):
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)

        assert folded.rows.shape == (N_REGIMES, DIMS_PER_REGIME)
        assert folded.n_dims == N_DIMS
        assert folded.n_regimes == N_REGIMES
        assert folded.dims_per_regime == DIMS_PER_REGIME

    def test_matrix_roundtrip(self):
        mat = _random_matrix(10)
        folded_list = fold_matrix(mat, N_REGIMES)
        reconstructed = unfold_matrix(folded_list)

        np.testing.assert_array_almost_equal(mat, reconstructed)
        assert reconstructed.shape == (10, N_DIMS)

    def test_different_n_regimes(self):
        for r in [2, 4, 8, 16]:
            vec = _random_vector()
            folded = fold_vector(vec, r)
            reconstructed = unfold_vector(folded)
            np.testing.assert_array_almost_equal(vec, reconstructed)

    def test_indivisible_dims_raises(self):
        vec = np.zeros(100)
        with pytest.raises(ValueError, match="divisible"):
            fold_vector(vec, 3)


# ===========================================================================
# Test: Reconstruction quality of the encoding
# ===========================================================================


class TestFoldingQuality:
    """Measure the lossless encoding without making a Gate claim."""

    def test_ungated_fold_is_perfect(self):
        """Without gating, fold/unfold is lossless."""
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)
        recon = unfold_vector(folded)

        quality = reconstruction_quality(vec, recon)
        assert abs(quality["cosine_similarity"] - 1.0) < 1e-10
        assert quality["l2_distance"] < 1e-10

    def test_reconstruction_quality_meets_threshold(self):
        """Fold/unfold without inter-regime gating meets cosine >= 0.98."""
        cosines = []
        for _ in range(100):
            vec = _random_vector()
            folded = fold_vector(vec, N_REGIMES)
            recon = unfold_vector(folded)
            q = reconstruction_quality(vec, recon)
            cosines.append(q["cosine_similarity"])

        mean_cosine = np.mean(cosines)
        assert mean_cosine >= 0.98, f"Mean reconstruction cosine {mean_cosine:.4f} < 0.98"


# ===========================================================================
# Test: Downstream task accuracy preservation
# ===========================================================================


class TestDownstreamAccuracy:
    """Simulate a classification task: train on full vectors, evaluate
    on fold/unfolded vectors.  Accuracy degradation should be <= 1%.
    """

    @pytest.fixture(autouse=True)
    def setup_classifier(self):
        n_classes = 4
        n_samples = 200
        self.n_classes = n_classes

        rng = np.random.default_rng(123)
        self.W = rng.standard_normal((n_classes, N_DIMS))
        self.b = rng.standard_normal(n_classes)

        self.X = rng.standard_normal((n_samples, N_DIMS))
        norms = np.linalg.norm(self.X, axis=1, keepdims=True)
        self.X = self.X / (norms + 1e-12)

        logits = self.X @ self.W.T + self.b
        self.y = np.argmax(logits, axis=1)

    def _classify(self, X: np.ndarray) -> np.ndarray:
        logits = X @ self.W.T + self.b
        return np.argmax(logits, axis=1)

    def test_baseline_accuracy(self):
        """Sanity: classifier on original vectors gets 100%."""
        preds = self._classify(self.X)
        acc = np.mean(preds == self.y)
        assert acc == 1.0

    def test_fold_unfold_preserves_accuracy(self):
        """Fold/unfold round-trip should preserve classification accuracy."""
        folded_list = fold_matrix(self.X, N_REGIMES)
        X_recon = unfold_matrix(folded_list)

        preds = self._classify(X_recon)
        acc = np.mean(preds == self.y)

        assert acc >= 0.99, f"Fold/unfold accuracy {acc:.3f} < 0.99 (degradation > 1%)"

    def test_multiple_fold_unfold_is_stable(self):
        """Multiple fold/unfold cycles should not degrade quality."""
        X_current = self.X.copy()
        for _cycle in range(5):
            folded = fold_matrix(X_current, N_REGIMES)
            X_current = unfold_matrix(folded)

        preds = self._classify(X_current)
        acc = np.mean(preds == self.y)
        assert acc >= 0.99, f"Accuracy after 5 fold/unfold cycles: {acc:.3f}"


# ===========================================================================
# Test: Regime0 storage properties
# ===========================================================================


class TestRegime0StorageProperties:
    """Verify properties specific to the Regime0 folding mechanism."""

    def test_each_row_fits_regime_width(self):
        """Each folded row has exactly n_dims / R elements."""
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)

        for i in range(N_REGIMES):
            assert folded.rows[i].shape == (DIMS_PER_REGIME,)

    def test_rows_are_independent_chunks(self):
        """Row i contains dimensions [i*D/R : (i+1)*D/R] of the original."""
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)

        for i in range(N_REGIMES):
            start = i * DIMS_PER_REGIME
            end = (i + 1) * DIMS_PER_REGIME
            np.testing.assert_array_almost_equal(folded.rows[i], vec[start:end])

    def test_folded_rows_sum_to_original_norm_squared(self):
        """Sum of squared norms of rows equals squared norm of original."""
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)

        row_norms_sq = sum(np.linalg.norm(folded.rows[i]) ** 2 for i in range(N_REGIMES))
        orig_norm_sq = np.linalg.norm(vec) ** 2

        np.testing.assert_almost_equal(row_norms_sq, orig_norm_sq)

    def test_partial_reconstruction_from_subset(self):
        """Reconstructing from fewer than R rows gives partial signal."""
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)

        partial = np.zeros(N_DIMS)
        for i in range(N_REGIMES // 2):
            start = i * DIMS_PER_REGIME
            end = (i + 1) * DIMS_PER_REGIME
            partial[start:end] = folded.rows[i]

        quality = reconstruction_quality(vec, partial)
        assert 0.0 < quality["cosine_similarity"] < 1.0
        assert quality["cosine_similarity"] > 0.5

    def test_reconstruction_improves_with_more_rows(self):
        """Adding more rows monotonically improves reconstruction."""
        vec = _random_vector()
        folded = fold_vector(vec, N_REGIMES)

        prev_cosine = 0.0
        for n_rows in range(1, N_REGIMES + 1):
            partial = np.zeros(N_DIMS)
            for i in range(n_rows):
                start = i * DIMS_PER_REGIME
                end = (i + 1) * DIMS_PER_REGIME
                partial[start:end] = folded.rows[i]

            q = reconstruction_quality(vec, partial)
            assert q["cosine_similarity"] >= prev_cosine - 1e-10
            prev_cosine = q["cosine_similarity"]

        assert abs(prev_cosine - 1.0) < 1e-10
