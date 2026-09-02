"""Public, research-only mask construction for CDP experiments.

This module intentionally contains no production key custody, token handling,
lockbox integration, or runtime authorization code.  It creates deterministic,
balanced partitions so the paper's model experiments can be reproduced without
publishing the production enforcement implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch


def balanced_partition(
    dimensions: int,
    regimes: int,
    *,
    experiment_seed: int,
) -> list[list[int]]:
    """Return deterministic, exhaustive, equal-width coordinate supports."""
    if dimensions <= 0 or regimes <= 0:
        raise ValueError("dimensions and regimes must be positive")
    if dimensions % regimes:
        raise ValueError(
            f"dimensions ({dimensions}) must be divisible by regimes ({regimes})"
        )

    if experiment_seed < 0 or experiment_seed >= 2**256:
        raise ValueError("experiment_seed must fit in 32 unsigned bytes")
    key_bytes = experiment_seed.to_bytes(32, "big")
    digest = hashlib.sha256(key_bytes).digest()
    rng_seed = int.from_bytes(digest[:8], "big") % (2**63)
    indices = np.random.default_rng(rng_seed).permutation(dimensions)
    width = dimensions // regimes
    return [
        sorted(indices[r * width : (r + 1) * width].tolist())
        for r in range(regimes)
    ]


def binary_masks(
    dimensions: int,
    regimes: int,
    *,
    experiment_seed: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    """Materialize one binary tensor for each support."""
    masks = []
    for support in balanced_partition(
        dimensions,
        regimes,
        experiment_seed=experiment_seed,
    ):
        mask = torch.zeros(dimensions, dtype=dtype, device=device)
        mask[support] = 1
        masks.append(mask)
    return masks
