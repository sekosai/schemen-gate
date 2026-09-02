"""Reproducible microbenchmark for the symbolic identity VectorBridge path."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc

import numpy as np

from schemen_gate import VectorBridge


def _average_seconds(operation, repetitions: int) -> float:
    started = time.perf_counter()
    for _ in range(repetitions):
        operation()
    return (time.perf_counter() - started) / repetitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--construct-repetitions", type=int, default=1000)
    parser.add_argument("--project-repetitions", type=int, default=100)
    args = parser.parse_args()

    dimension = args.dimension
    vector = np.ones(dimension, dtype=np.float64)

    tracemalloc.start()
    implicit_construct_s = _average_seconds(
        lambda: VectorBridge(dimension, dimension),
        args.construct_repetitions,
    )
    _, implicit_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()

    explicit_matrix = np.eye(dimension, dtype=np.float64)
    explicit_construct_s = _average_seconds(
        lambda: VectorBridge(dimension, dimension, explicit_matrix),
        max(1, args.construct_repetitions // 100),
    )
    _, explicit_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    implicit = VectorBridge(dimension, dimension)
    explicit = VectorBridge(dimension, dimension, explicit_matrix)
    implicit_project_s = _average_seconds(
        lambda: implicit.project(vector), args.project_repetitions
    )
    explicit_project_s = _average_seconds(
        lambda: explicit.project(vector), args.project_repetitions
    )

    print(
        json.dumps(
            {
                "dimension": dimension,
                "implicit_construct_us": implicit_construct_s * 1e6,
                "explicit_construct_ms": explicit_construct_s * 1e3,
                "implicit_peak_bytes": implicit_peak_bytes,
                "explicit_peak_bytes": explicit_peak_bytes,
                "implicit_project_us": implicit_project_s * 1e6,
                "explicit_project_ms": explicit_project_s * 1e3,
                "explicit_matrix_bytes": explicit_matrix.nbytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
