"""Follow-up 3: Real model folding with DistilBERT hidden states.

Tests whether actual model activations (not random vectors) survive
fold/unfold through Regime0, and whether downstream task accuracy
is preserved on real classification data.

Requires: pip install transformers torch datasets
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

try:
    import torch
    from transformers import AutoModel, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

pytestmark = pytest.mark.skipif(not _HAS_TRANSFORMERS, reason="transformers/torch not installed")

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
N_REGIMES = 8  # 768 / 8 = 96 dims per regime

_model_cache = {}


def _get_model_and_tokenizer():
    if "model" not in _model_cache:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
        model.eval()
        _model_cache["model"] = model
        _model_cache["tokenizer"] = tokenizer
    return _model_cache["model"], _model_cache["tokenizer"]


def _get_hidden_states(texts: list[str]) -> np.ndarray:
    """Extract CLS hidden states from DistilBERT."""
    model, tokenizer = _get_model_and_tokenizer()
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_states = outputs.last_hidden_state[:, 0, :].numpy().astype(np.float64)
    return cls_states


CLASSIFICATION_TEXTS = [
    "The movie was absolutely wonderful and I loved every minute",
    "This is the worst film I have ever seen in my entire life",
    "A masterpiece of modern cinema with brilliant performances",
    "Terrible acting and an incoherent plot ruined this movie",
    "I was deeply moved by the emotional depth of this story",
    "What a waste of time, completely boring and predictable",
    "Outstanding direction and a captivating screenplay throughout",
    "The special effects were laughable and the dialogue was awful",
    "A heartwarming tale that restores your faith in humanity",
    "I fell asleep halfway through this tedious disaster",
    "Brilliant character development and superb cinematography",
    "Not even worth the price of admission to see this garbage",
    "One of the best films of the year without a doubt",
    "Painfully slow pacing with absolutely no redeeming qualities",
    "A triumph of storytelling that will stand the test of time",
    "Completely forgettable and devoid of any originality",
    "The performances were electric and the tension was palpable",
    "A cheap cash grab with no artistic merit whatsoever",
    "Beautifully crafted with attention to every small detail",
    "I want those two hours of my life back after watching this",
]

LABELS = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])


class TestRealModelFolding:
    """Fold/unfold real DistilBERT hidden states and measure accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_hidden_states(self):
        hidden = _get_hidden_states(CLASSIFICATION_TEXTS)
        TestRealModelFolding._hidden = hidden
        TestRealModelFolding._n_dim = hidden.shape[1]

    def test_hidden_states_shape(self):
        assert self._hidden.shape == (20, 768)

    def test_fold_unfold_lossless(self):
        """Fold/unfold of real hidden states should be lossless."""
        for i in range(self._hidden.shape[0]):
            vec = self._hidden[i]
            folded = fold_vector(vec, N_REGIMES)
            recon = unfold_vector(folded)
            q = reconstruction_quality(vec, recon)
            assert q["cosine_similarity"] > 0.9999, f"Row {i}: cosine {q['cosine_similarity']:.6f}"

    def test_batch_fold_unfold_lossless(self):
        folded = fold_matrix(self._hidden, N_REGIMES)
        recon = unfold_matrix(folded)
        np.testing.assert_array_almost_equal(self._hidden, recon)

    def test_classification_preserved(self):
        """Train a classifier on original, evaluate on folded/unfolded."""
        X = self._hidden
        y = LABELS

        W = np.linalg.lstsq(X, y.astype(np.float64), rcond=None)[0]

        orig_preds = (X @ W > 0.5).astype(int)
        orig_acc = np.mean(orig_preds == y)

        folded = fold_matrix(X, N_REGIMES)
        X_recon = unfold_matrix(folded)
        recon_preds = (X_recon @ W > 0.5).astype(int)
        recon_acc = np.mean(recon_preds == y)

        assert recon_acc >= orig_acc - 0.01, (
            f"Fold/unfold accuracy {recon_acc:.3f} vs original {orig_acc:.3f}"
        )

    def test_cosine_distribution(self):
        """All reconstruction cosines should be > 0.999."""
        folded = fold_matrix(self._hidden, N_REGIMES)
        recon = unfold_matrix(folded)

        cosines = []
        for i in range(self._hidden.shape[0]):
            q = reconstruction_quality(self._hidden[i], recon[i])
            cosines.append(q["cosine_similarity"])

        assert min(cosines) > 0.999
        assert np.mean(cosines) > 0.9999

    def test_different_regime_counts(self):
        """Fold/unfold works for R = 2, 4, 8, 16."""
        for r in [2, 4, 8, 16]:
            if 768 % r != 0:
                continue
            folded = fold_matrix(self._hidden, r)
            recon = unfold_matrix(folded)
            np.testing.assert_array_almost_equal(self._hidden, recon)

    def test_partial_reconstruction_quality(self):
        """Partial reconstruction from k < R rows on real activations."""
        vec = self._hidden[0]
        folded = fold_vector(vec, N_REGIMES)
        dims_per = 768 // N_REGIMES

        prev_cos = 0.0
        for k in range(1, N_REGIMES + 1):
            partial = np.zeros(768)
            for i in range(k):
                partial[i * dims_per : (i + 1) * dims_per] = folded.rows[i]
            q = reconstruction_quality(vec, partial)
            assert q["cosine_similarity"] >= prev_cos - 1e-10
            prev_cos = q["cosine_similarity"]

    def test_hidden_state_energy_conservation(self):
        """Energy (squared norm) is conserved across folded rows."""
        for idx in range(5):
            vec = self._hidden[idx]
            folded = fold_vector(vec, N_REGIMES)
            row_energy = sum(np.linalg.norm(folded.rows[i]) ** 2 for i in range(N_REGIMES))
            orig_energy = np.linalg.norm(vec) ** 2
            np.testing.assert_almost_equal(row_energy, orig_energy, decimal=8)
