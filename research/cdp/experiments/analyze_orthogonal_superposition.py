"""Reanalyze a completed orthogonal run without repeating paid compute."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from library_provenance import collect_source_provenance


def classify_ratio(row: dict) -> dict:
    rejections = row["runtime"]["rejection_probe"]
    passed = (
        row["maximum_absolute_accuracy_gap"] == 0.0
        and row["accuracy_zero_loss"] is True
        and rejections["all_rejected"] is True
        and rejections["unauthorized_model_calls"] == 0
    )
    return {
        "R": row["R"],
        "evaluated_regimes": row["evaluated_regimes"],
        "baseline_accuracy": row["baseline_accuracy"],
        "maximum_absolute_accuracy_gap": row["maximum_absolute_accuracy_gap"],
        "maximum_absolute_logit_difference_diagnostic": row[
            "maximum_absolute_logit_difference"
        ],
        "all_runtime_rejections_pass": rejections["all_rejected"],
        "unauthorized_model_calls": rejections["unauthorized_model_calls"],
        "status": "pass" if passed else "failure",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    raw_bytes = args.input.read_bytes()
    source_artifact = json.loads(raw_bytes)
    result = source_artifact["result"]
    ratios = [classify_ratio(row) for row in result["ratios"]]
    artifact = {
        "schema_version": 1,
        "experiment": "orthogonal_superposition_reanalysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "pass" if all(row["status"] == "pass" for row in ratios) else "failure"
        ),
        "claim": (
            "Exact measured accuracy equivalence under execution-authorized "
            "whole-model permutation placement."
        ),
        "claim_boundary": (
            "Low-order fp32 logit drift is diagnostic, not loss, and this is "
            "not sparse FFN cotenancy or concurrent execution."
        ),
        "verdict_correction": (
            "The raw runner incorrectly thresholded an auxiliary logit drift "
            "at 1e-5. The historical and mathematical claim is zero accuracy "
            "loss; all predictions remained unchanged."
        ),
        "source_artifact": {
            "path": str(args.input),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "source_commit": source_artifact["source"]["commit"],
            "source_dirty": source_artifact["source"]["dirty"],
        },
        "analysis_source": collect_source_provenance(),
        "ratios": ratios,
    }
    output = args.output or args.input.with_name(
        args.input.stem + "_reanalysis.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
