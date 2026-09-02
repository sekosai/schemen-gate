"""Pure helpers for the execution service-consolidation benchmarks."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def percentile(samples: Iterable[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty samples."""

    values = sorted(float(sample) for sample in samples)
    if not values:
        raise ValueError("at least one sample is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def tensor_bytes(tensors: Iterable[Any]) -> int:
    """Count unique tensor storage bytes without double-counting tied weights."""

    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        identity = (str(tensor.device), int(storage.data_ptr()))
        if identity in seen:
            continue
        seen.add(identity)
        total += int(storage.nbytes())
    return total


def state_dict_bytes(state: Mapping[str, Any]) -> int:
    return tensor_bytes(state.values())


def model_parameter_bytes(model: Any) -> int:
    return tensor_bytes(model.parameters())


def summarize_timings(
    latencies_seconds: Iterable[float], *, samples_per_request: int
) -> dict[str, float | int]:
    values = [float(value) for value in latencies_seconds]
    if samples_per_request <= 0:
        raise ValueError("samples_per_request must be positive")
    total = sum(values)
    return {
        "requests": len(values),
        "samples": len(values) * samples_per_request,
        "p50_latency_ms": percentile(values, 0.50) * 1000.0,
        "p95_latency_ms": percentile(values, 0.95) * 1000.0,
        "mean_latency_ms": (total / len(values)) * 1000.0,
        "throughput_samples_per_second": (
            len(values) * samples_per_request / total
        ),
    }


def require_complete_result(result: Mapping[str, Any]) -> None:
    """Fail closed when a remote result lacks required proof fields."""

    required_conditions = {
        "separate_services",
        "shared_backbone_private_adapters",
        "shared_ffn_authorized_slices",
        "physically_extracted_authorized_slice",
    }
    conditions = result.get("conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("conditions are required")
    missing = required_conditions - set(conditions)
    if missing:
        raise ValueError(f"missing conditions: {sorted(missing)}")
    runtime = result.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime evidence is required")
    rejection = runtime.get("rejection_probe")
    if not isinstance(rejection, Mapping):
        raise ValueError("runtime rejection evidence is required")
    if rejection.get("all_rejected") is not True:
        raise ValueError("runtime malformed-authority probes did not all reject")
    if rejection.get("unauthorized_model_calls") != 0:
        raise ValueError("an unauthorized model callback was observed")
    extraction = result.get("extraction_equivalence")
    if not isinstance(extraction, Mapping) or extraction.get("within_tolerance") is not True:
        raise ValueError("physical extraction equivalence was not proven")
