"""Aggregate strict Transformer cotenancy artifacts.

The script ignores smoke runs, verifies the expected three-seed matrix, and
writes the compact values used by the paper.
"""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"
EXPECTED_SEEDS = {42, 123, 256}


def load_results(pattern: str) -> list[dict[str, Any]]:
    rows = []
    for path in RESULTS.glob(pattern):
        artifact = json.loads(path.read_text())
        rows.extend(artifact.get("results", []))
    return rows


def mean_sd(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_sd": statistics.stdev(values),
    }


def dense_summary() -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_results("dense_ffn_cotenancy_*.json")
        if not row["config"]["smoke"]
    ]
    by_ratio: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_ratio.setdefault(int(row["config"]["R"]), []).append(row)

    if set(by_ratio) != {1, 2, 4, 8, 16}:
        raise RuntimeError(f"incomplete dense ratio matrix: {sorted(by_ratio)}")
    for ratio, ratio_rows in by_ratio.items():
        seeds = {int(row["config"]["seed"]) for row in ratio_rows}
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"R={ratio} has seeds {sorted(seeds)}")

    baseline = statistics.mean(
        [row["mean_owning_accuracy"] for row in by_ratio[1]]
    )
    summary = []
    for ratio in sorted(by_ratio):
        ratio_rows = by_ratio[ratio]
        owning = [row["mean_owning_accuracy"] for row in ratio_rows]
        wrong = [
            row["mean_wrong_key_accuracy"]
            for row in ratio_rows
            if row["mean_wrong_key_accuracy"] is not None
        ]
        summary.append(
            {
                "R": ratio,
                "dimensions_per_regime_per_layer": ratio_rows[0][
                    "dimensions_per_regime_per_layer"
                ],
                "owning_accuracy": mean_sd(owning),
                "drop_from_R1_percentage_points": 100
                * (baseline - statistics.mean(owning)),
                "wrong_key_accuracy": mean_sd(wrong) if wrong else None,
                "maximum_frozen_shared_parameter_delta": max(
                    row["maximum_frozen_shared_parameter_delta"]
                    for row in ratio_rows
                ),
                "maximum_off_partition_parameter_delta": max(
                    row["maximum_off_partition_parameter_delta"]
                    for row in ratio_rows
                ),
                "maximum_off_partition_optimizer_moment": max(
                    row["maximum_off_partition_optimizer_moment"]
                    for row in ratio_rows
                ),
                "maximum_inactive_classifier_delta": max(
                    row["maximum_inactive_classifier_delta"]
                    for row in ratio_rows
                ),
            }
        )
    return summary


def lane_summary() -> list[dict[str, Any]]:
    rows = [
        row
        for row in load_results("private_transformer_lanes_*.json")
        if not row["config"]["smoke"]
    ]
    summary = []
    for design in ("adapter", "expert"):
        selected = [
            row for row in rows if row["config"]["design"] == design
        ]
        seeds = {int(row["config"]["seed"]) for row in selected}
        if seeds != EXPECTED_SEEDS:
            raise RuntimeError(f"{design} has seeds {sorted(seeds)}")
        summary.append(
            {
                "design": design,
                "owning_accuracy": mean_sd(
                    [row["mean_owning_accuracy"] for row in selected]
                ),
                "wrong_key_accuracy": mean_sd(
                    [row["mean_wrong_key_accuracy"] for row in selected]
                ),
                "maximum_shared_backbone_delta": max(
                    row["maximum_shared_backbone_delta"] for row in selected
                ),
                "maximum_inactive_lane_delta": max(
                    row["maximum_inactive_lane_delta"] for row in selected
                ),
            }
        )
    return summary


def cargo_summary() -> dict[str, Any]:
    paths = sorted(RESULTS.glob("cargo_transformer_*.json"))
    if not paths:
        raise RuntimeError("missing Cargo Transformer artifact")
    result = json.loads(paths[-1].read_text())["result"]
    return {
        "status": result["status"],
        "owning_exact_recall": result["owning_exact_recall"],
        "all_unauthorized_docks_rejected": result[
            "all_unauthorized_docks_rejected"
        ],
        "unauthorized_model_calls": result["unauthorized_model_calls"],
        "load_receipts_valid": result["load_receipts_valid"],
        "retrieval_receipts_valid": result["retrieval_receipts_valid"],
        "owning_queries": len(result["owning_rows"]),
        "unauthorized_attempts": len(result["rejection_rows"]),
        "threat_boundary": result["threat_boundary"],
    }


def main() -> None:
    artifact = {
        "schema_version": 1,
        "experiment": "transformer_cotenancy_analysis",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": sorted(EXPECTED_SEEDS),
        "dense_partition_algorithm": "sha256_seed_numpy_balanced_research",
        "dense_ffn": dense_summary(),
        "private_lanes": lane_summary(),
        "cargo": cargo_summary(),
    }
    output = RESULTS / "transformer_cotenancy_summary.json"
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
