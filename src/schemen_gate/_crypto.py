"""Keyed mask derivation for Schemen Gate.

This module uses HMAC-SHA256 from the Python standard library and NumPy. The
``crypto`` package extra is required by token, wrapping, and signature APIs,
not by deterministic mask derivation itself.

The byte-level algorithm is part of the public Gate contract and is locked by
immutable test vectors. Other implementations are interoperable only when they
reproduce those vectors exactly.
"""

from __future__ import annotations

import hashlib
import hmac as _hmac

import numpy as np


def _csprng_permutation(key_material: bytes, n: int) -> list[int]:
    """Fisher-Yates shuffle of [0, n) driven by HMAC-SHA256 in counter mode.

    Each 32-byte HMAC block yields eight 4-byte draws.  Rejection sampling
    eliminates modulo bias.  The full 256-bit key_material is used.
    """
    indices = list(range(n))
    buf = b""
    ctr = 0

    for i in range(n - 1, 0, -1):
        bound = i + 1
        limit = (0x1_0000_0000 // bound) * bound
        while True:
            if len(buf) < 4:
                ctr += 1
                buf += _hmac.new(key_material, ctr.to_bytes(4, "big"), hashlib.sha256).digest()
            val = int.from_bytes(buf[:4], "big")
            buf = buf[4:]
            if val < limit:
                break
        j = val % bound
        indices[i], indices[j] = indices[j], indices[i]

    return indices


def derive_partition(key: bytes, n_dims: int, n_regimes: int) -> list[list[int]]:
    """Derive disjoint dimension groups from a 32-byte key.

    The fixed vectors in ``tests/test_mask.py`` lock exact output ordering.
    """
    if len(key) != 32:
        raise ValueError("Key must be exactly 32 bytes")
    if n_dims <= 0 or n_regimes <= 0:
        raise ValueError("n_dims and n_regimes must be positive")
    if n_dims % n_regimes != 0:
        raise ValueError(f"n_dims ({n_dims}) must be divisible by n_regimes ({n_regimes})")

    tag = _hmac.new(key, b"partition", hashlib.sha256).digest()
    perm = _csprng_permutation(tag, n_dims)
    group_size = n_dims // n_regimes
    return [perm[i * group_size : (i + 1) * group_size] for i in range(n_regimes)]


def derive_gate_mask_raw(
    key: bytes,
    regime_id: int,
    n_dims: int,
    n_regimes: int = 2,
) -> np.ndarray:
    """Return a binary float64 mask for ``regime_id``.

    Given the same validated inputs, the output is bit-identical across Gate
    versions that retain this protocol contract.
    """
    if regime_id < 0 or regime_id >= n_regimes:
        raise ValueError(f"regime_id {regime_id} out of range [0, {n_regimes})")
    groups = derive_partition(key, n_dims, n_regimes)
    mask = np.zeros(n_dims, dtype=np.float64)
    mask[groups[regime_id]] = 1.0
    return mask
