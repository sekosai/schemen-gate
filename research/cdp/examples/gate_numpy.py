"""Small, no-network CDP gate example using the locked Schemen Gate wheel."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from schemen_gate import GateMask


def main() -> None:
    # This key stands in for authority-held key material. It is deterministic so
    # the example is replayable, and it is never accepted from a request.
    authority_key = hashlib.sha256(b"cdp-paper-public-example-v1").digest()
    masks = [
        GateMask.derive(
            authority_key,
            regime_id,
            n_dims=8,
            n_regimes=2,
        ).to_numpy()
        for regime_id in range(2)
    ]
    hidden = np.arange(1, 9, dtype=np.float64)
    outputs = [hidden * mask for mask in masks]

    assert np.array_equal(masks[0] + masks[1], np.ones(8))
    assert np.count_nonzero(masks[0] * masks[1]) == 0
    assert all(
        np.count_nonzero(output[mask == 0]) == 0
        for output, mask in zip(outputs, masks, strict=True)
    )

    print(
        json.dumps(
            {
                "hidden": hidden.tolist(),
                "masks": [mask.astype(int).tolist() for mask in masks],
                "gated_outputs": [output.tolist() for output in outputs],
                "partition_complete": True,
                "partition_disjoint": True,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
