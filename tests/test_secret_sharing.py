"""Executable evidence for GF(256) Shamir sharing of Gate keys."""

from __future__ import annotations

import itertools
import os

import pytest

from schemen_gate import GateKey, reconstitute_gate_key, split_gate_key
from schemen_gate._lockbox import _eval_poly, _gf256_inv, _gf256_mul


def test_gf256_inverse_is_exact_for_every_nonzero_element() -> None:
    for element in range(1, 256):
        assert _gf256_mul(element, _gf256_inv(element)) == 1
    with pytest.raises(ValueError, match="invert zero"):
        _gf256_inv(0)


def test_gf256_multiplication_is_commutative_with_identity() -> None:
    for a in (1, 2, 0x53, 0xCA, 0xFF):
        assert _gf256_mul(a, 1) == a
        for b in (1, 3, 0x8E, 0x80):
            assert _gf256_mul(a, b) == _gf256_mul(b, a)
    # The AES field: 0x53 and 0xCA are multiplicative inverses.
    assert _gf256_mul(0x53, 0xCA) == 1


def test_polynomial_evaluation_at_zero_returns_the_secret_byte() -> None:
    assert _eval_poly([0x2A, 7, 9], 0) == 0x2A
    assert _eval_poly([0x2A], 200) == 0x2A


@pytest.mark.parametrize(("threshold", "num_shares"), [(2, 2), (2, 3), (3, 5), (5, 5), (4, 9)])
def test_any_threshold_subset_reconstitutes_the_exact_key(threshold: int, num_shares: int) -> None:
    key = GateKey(os.urandom(32))
    shares = split_gate_key(key, threshold, num_shares)

    assert [index for index, _ in shares] == list(range(1, num_shares + 1))
    assert all(len(share) == 32 for _, share in shares)
    for subset in itertools.islice(itertools.combinations(shares, threshold), 12):
        assert reconstitute_gate_key(list(subset)).secret == key.secret
    assert reconstitute_gate_key(list(reversed(shares))).secret == key.secret


def test_fewer_than_threshold_shares_do_not_yield_the_key() -> None:
    key = GateKey(b"\x11" * 32)
    shares = split_gate_key(key, 3, 5)
    assert reconstitute_gate_key(shares[:2]).secret != key.secret


def test_each_split_uses_fresh_randomness() -> None:
    key = GateKey(b"\x22" * 32)
    assert split_gate_key(key, 2, 3) != split_gate_key(key, 2, 3)


@pytest.mark.parametrize(("threshold", "num_shares"), [(1, 3), (3, 2), (2, 256), (0, 0)])
def test_invalid_split_parameters_are_rejected(threshold: int, num_shares: int) -> None:
    with pytest.raises(ValueError, match="threshold"):
        split_gate_key(GateKey(b"k" * 32), threshold, num_shares)


def test_reconstitution_rejects_malformed_share_sets() -> None:
    shares = split_gate_key(GateKey(b"k" * 32), 2, 3)
    with pytest.raises(ValueError, match="at least 2"):
        reconstitute_gate_key(shares[:1])
    with pytest.raises(ValueError, match="same length"):
        reconstitute_gate_key([shares[0], (2, shares[1][1][:31])])
    with pytest.raises(ValueError, match="Duplicate"):
        reconstitute_gate_key([shares[0], shares[0]])
    for bad_index in (0, 256, True):
        with pytest.raises(ValueError, match="indices"):
            reconstitute_gate_key([(bad_index, shares[0][1]), shares[1]])
