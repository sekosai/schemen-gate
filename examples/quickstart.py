"""Smallest inspectable Schemen Gate example.

Run after ``python -m pip install -e .`` from the repository root.
"""

from __future__ import annotations

import numpy as np

from schemen_gate import GateMask


def main() -> None:
    hidden = np.arange(8, dtype=np.float64)
    gate = GateMask.from_indices([0, 2, 4, 6], n_dims=hidden.size)
    gated = gate.apply(hidden)

    if not np.all(gated[gate.mask == 0] == 0):
        raise RuntimeError("inactive dimensions were not zeroed")
    if not np.array_equal(gated[gate.mask == 1], hidden[gate.mask == 1]):
        raise RuntimeError("active dimensions changed")

    print(f"input:  {hidden.tolist()}")
    print(f"mask:   {gate.mask.tolist()}")
    print(f"gated:  {gated.tolist()}")
    print("PASS: inactive dimensions are zero and active dimensions are unchanged")


if __name__ == "__main__":
    main()
