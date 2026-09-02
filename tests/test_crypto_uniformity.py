"""Uniformity fixtures for ``schemen_gate._crypto``.

Maps to the Lean development ``GateSecurity.lean`` §10 and §6:

- ``rejection_sampling_count`` / ``rejection_unbiased`` (proven): residue
  classes mod ``bound`` are exactly equinumerous, so per-draw sampling is
  uniform. The chi-square check below is a STATISTICAL FIXTURE replicating
  the proven fact, with a deterministic key schedule (HMAC of a fixed
  master key); it is labeled as such — it adds empirical confidence, and
  the proof stands without it.
- ``ValidPartition.disjoint/exhaustive/equal_size`` (proven structurally):
  checked here EXACTLY (not statistically) on the production entry point.

No secrets, no network, deterministic seeds. Fail-closed: any deviation
raises rather than warns.
"""

from __future__ import annotations

import hashlib
import math

import pytest

crypto = pytest.importorskip("cryptography")

from schemen_gate._crypto import derive_partition  # noqa: E402


def _fixture_key(i: int) -> bytes:
    """Deterministic per-trial keys from a fixed master key (FIXTURE)."""
    return hashlib.sha256(b"schemen-uniformity-fixture-v1:" + i.to_bytes(8, "big")).digest()


def test_permutation_is_a_bijection():
    """Exact check: Fisher-Yates output is a permutation of [0, n)."""
    from schemen_gate._crypto import _csprng_permutation

    n = 257
    perm = _csprng_permutation(_fixture_key(0), n)
    assert sorted(perm) == list(range(n))


def test_partition_groups_disjoint_exhaustive_equal_sized():
    """Exact check mirroring the ValidPartition structure fields."""
    groups = derive_partition(_fixture_key(1), 768, 2)
    flat = [d for g in groups for d in g]
    assert len(flat) == 768  # exhaustive + no double counting
    assert sorted(flat) == list(range(768))  # disjoint and exhaustive
    assert all(len(g) == 384 for g in groups)  # equal_size


def test_partition_derivation_is_deterministic():
    """Derivation is a pure function of the key (Facet 2, dual-phase)."""
    key = _fixture_key(2)
    assert derive_partition(key, 768, 4) == derive_partition(key, 768, 4)


@pytest.mark.parametrize(("n", "trials"), [(8, 4096)])
def test_first_position_uniformity_statistical_fixture(n: int, trials: int):
    """STATISTICAL FIXTURE — replicates ``rejection_unbiased`` empirically.

    Distribution of the element landing at position 0 across many keyed
    shuffles should be uniform over n symbols. Chi-square statistic against
    the uniform expectation, with a generous bound (mean df + 5 sigma) so the
    check fails only on a genuinely broken sampler, never on noise.

    If this test ever flakes, treat it as a security signal, not a test
    bug: the proven statement is exact, so a broken sampler means a
    regression in the draw path (e.g. someone removed rejection sampling).
    """
    from schemen_gate._crypto import _csprng_permutation

    counts = [0] * n
    for i in range(trials):
        perm = _csprng_permutation(_fixture_key(1000 + i), n)
        counts[perm[0]] += 1

    expected = trials / n
    chi2 = sum((c - expected) ** 2 / expected for c in counts)
    df = n - 1
    bound = df + 5.0 * math.sqrt(2.0 * df)
    assert chi2 < bound, (
        f"first-position distribution deviates from uniform: "
        f"chi2={chi2:.2f} > {bound:.2f} (df={df}). counts={counts}"
    )
