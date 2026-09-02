"""Free canary: verify gated/full and extracted submodel equivalence.

This isolates the algebra used by the paper's exact-extraction claim without
downloading or training a backbone.  Every run writes a timestamped JSON
artifact so local and cloud evidence follow the same collection discipline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
from library_provenance import collect_library_provenance, collect_source_provenance

from schemen_gate import GateMask


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_case(
    *,
    dtype: torch.dtype,
    seed: int,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    regimes: int,
) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    absolute_tolerance = 1e-6 if dtype == torch.float32 else 2e-2
    relative_tolerance = 0.0 if dtype == torch.float32 else 1e-3
    x = torch.randn(batch_size, input_dim, dtype=dtype, generator=generator)
    weight = torch.randn(
        output_dim, input_dim, dtype=dtype, generator=generator
    )
    bias = torch.randn(output_dim, dtype=dtype, generator=generator)

    gate_key = hashlib.sha256(f"cdp-exact-extraction:{seed}".encode()).digest()
    masks = [
        GateMask.derive(
            gate_key,
            regime,
            n_dims=input_dim,
            n_regimes=regimes,
        ).to_torch(dtype=dtype)
        for regime in range(regimes)
    ]
    chunks = [torch.nonzero(mask, as_tuple=False).flatten() for mask in masks]
    regime_results = []

    def forward_error_bound(active: torch.Tensor) -> float:
        """Conservative bound for two differently shaped floating matmuls."""

        epsilon = torch.finfo(dtype).eps
        magnitude = (
            x[:, active].abs() @ weight[:, active].abs().T
            + bias.abs()
        ).max()
        return float(4 * epsilon * active.numel() * magnitude.item())

    for regime in range(regimes):
        active = chunks[regime].sort().values
        mask = masks[regime]

        gated = (x * mask) @ weight.T + bias
        extracted = x[:, active] @ weight[:, active].T + bias
        diff = (gated - extracted).abs()
        error_bound = forward_error_bound(active)

        regime_results.append(
            {
                "regime": regime,
                "active_dimensions": int(active.numel()),
                "max_absolute_difference": float(diff.max().item()),
                "forward_error_bound": error_bound,
                "bit_equal": bool(torch.equal(gated, extracted)),
                "allclose": bool(torch.allclose(gated, extracted, rtol=0, atol=0)),
                "within_dtype_tolerance": bool(diff.max().item() <= error_bound),
            }
        )

    compound_results = []
    for count in range(1, regimes + 1):
        active = torch.cat([chunks[r] for r in range(count)]).sort().values
        mask = torch.zeros(input_dim, dtype=dtype)
        mask[active] = 1

        gated = (x * mask) @ weight.T + bias
        extracted = x[:, active] @ weight[:, active].T + bias
        diff = (gated - extracted).abs()
        error_bound = forward_error_bound(active)
        compound_results.append(
            {
                "authorized_regimes": count,
                "active_dimensions": int(active.numel()),
                "max_absolute_difference": float(diff.max().item()),
                "forward_error_bound": error_bound,
                "bit_equal": bool(torch.equal(gated, extracted)),
                "allclose": bool(torch.allclose(gated, extracted, rtol=0, atol=0)),
                "within_dtype_tolerance": bool(diff.max().item() <= error_bound),
            }
        )

    max_diff = max(
        row["max_absolute_difference"]
        for row in regime_results + compound_results
    )
    max_bound = max(
        row["forward_error_bound"]
        for row in regime_results + compound_results
    )
    return {
        "dtype": str(dtype).removeprefix("torch."),
        "seed": seed,
        "absolute_tolerance": absolute_tolerance,
        "relative_tolerance": relative_tolerance,
        "max_absolute_difference": max_diff,
        "maximum_forward_error_bound": max_bound,
        "maximum_observed_to_bound_ratio": max(
            row["max_absolute_difference"] / row["forward_error_bound"]
            if row["forward_error_bound"]
            else 0.0
            for row in regime_results + compound_results
        ),
        "within_tolerance": all(
            row["within_dtype_tolerance"]
            for row in regime_results + compound_results
        ),
        "all_cases_bit_equal": all(
            row["bit_equal"] for row in regime_results + compound_results
        ),
        "all_cases_exact_allclose": all(
            row["allclose"] for row in regime_results + compound_results
        ),
        "regimes": regime_results,
        "compound_unions": compound_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--output-dim", type=int, default=4)
    parser.add_argument("--regimes", type=int, default=4)
    args = parser.parse_args()

    if args.input_dim % args.regimes != 0:
        raise ValueError("input_dim must be divisible by regimes")

    started = time.time()
    cases = [
        run_case(
            dtype=dtype,
            seed=seed,
            batch_size=args.batch_size,
            input_dim=args.input_dim,
            output_dim=args.output_dim,
            regimes=args.regimes,
        )
        for dtype in (torch.float32, torch.float16)
        for seed in range(args.seeds)
    ]

    artifact = {
        "schema_version": 2,
        "experiment": "local_exact_extraction",
        "status": "pass"
        if all(case["within_tolerance"] for case in cases)
        else "tolerance_failure",
        "started_at_unix": started,
        "elapsed_seconds": time.time() - started,
        "git_revision": git_revision(),
        "source": collect_source_provenance(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "libraries": collect_library_provenance(),
        },
        "mask_contract": {
            "implementation": "schemen_gate.GateMask.derive",
            "key_derivation": "SHA-256('cdp-exact-extraction:' + seed)",
            "n_dims": args.input_dim,
            "n_regimes": args.regimes,
        },
        "config": vars(args),
        "summary": {
            "cases": len(cases),
            "regime_checks": len(cases) * args.regimes,
            "compound_checks": len(cases) * args.regimes,
            "all_cases_bit_equal": all(
                case["all_cases_bit_equal"] for case in cases
            ),
            "all_cases_within_dtype_tolerance": all(
                case["within_tolerance"] for case in cases
            ),
            "maximum_absolute_difference": max(
                case["max_absolute_difference"] for case in cases
            ),
            "maximum_forward_error_bound": max(
                case["maximum_forward_error_bound"] for case in cases
            ),
            "maximum_observed_to_bound_ratio": max(
                case["maximum_observed_to_bound_ratio"] for case in cases
            ),
        },
        "cases": cases,
    }

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"local_exact_extraction_{timestamp}.json"
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")

    print(json.dumps(artifact["summary"], indent=2))
    print(f"Results saved to {output_path}")
    if artifact["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
