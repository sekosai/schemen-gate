"""Executable evidence for the retrieval-architecture analysis helpers."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from schemen_gate import analyze_vectors


def _two_separated_clusters(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first = rng.standard_normal((40, 8)) * 0.05 + np.array([10.0] + [0.0] * 7)
    second = rng.standard_normal((40, 8)) * 0.05 + np.array([-10.0] + [0.0] * 7)
    return np.vstack([first, second])


def test_analysis_finds_two_well_separated_clusters() -> None:
    analysis = analyze_vectors(_two_separated_clusters(), max_clusters=4)

    assert analysis.n_vectors == 80
    assert analysis.n_clusters == 2
    assert len(analysis.cluster_dims) == 2
    assert all(1 <= dims <= 8 for dims in analysis.cluster_dims)
    assert 1 <= analysis.effective_rank <= 8
    assert analysis.suggested_n_heads == 2
    assert 1 <= analysis.suggested_head_dim <= 8
    assert 0.0 < analysis.compression_ratio <= 1.0


def test_analysis_of_fewer_than_two_vectors_is_trivial() -> None:
    analysis = analyze_vectors(np.ones((1, 6)))

    assert analysis.n_vectors == 1
    assert analysis.n_clusters == 1
    assert analysis.cluster_dims == [6]
    assert analysis.effective_rank == 1
    assert analysis.compression_ratio == 1.0


def test_analysis_falls_back_to_the_spectral_heuristic_without_scikit_learn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sklearn", None)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", None)
    monkeypatch.setitem(sys.modules, "sklearn.metrics", None)

    analysis = analyze_vectors(_two_separated_clusters(), max_clusters=4)

    assert 1 <= analysis.n_clusters <= 4
    assert len(analysis.cluster_dims) == analysis.n_clusters
    assert analysis.cluster_dims == [8] * analysis.n_clusters
