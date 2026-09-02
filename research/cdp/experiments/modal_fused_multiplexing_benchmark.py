"""Fused multiplexing microbenchmark for Modal GPUs (PLAN Priority 2).

This benchmark measures one representative Transformer MLP boundary:

    inverse-permute residual -> up projection -> GELU -> down projection
    -> forward-permute residual

It compares equal-work implementations over the same R regimes and tokens:

* sequential_dense: R calls using materialized conjugated weights;
* batched_dense: regime-indexed batched GEMMs using the same weights;
* eager_masked_multiplexing: eager gathers around shared dense weights;
* compiled_exact_multiplexing: torch.compile/Inductor applied to that exact
  eager expression.

The compiled condition is semantically valid because its output must pass the
same numerical comparison as every other condition.  It is not described as a
custom Triton gather-GEMM: Inductor decides which pointwise/gather operations
it can fuse around vendor GEMMs.  Compilation time is measured separately.

No fallback is allowed for the compiled condition.  If CUDA, Triton, or the
Inductor backend is unavailable, the remote job persists ``unsupported`` and
returns without timing a substitute implementation.

Examples (documentation only; importing this file does not launch anything):

    modal run experiments/modal_fused_multiplexing_benchmark.py --smoke
    modal run experiments/modal_fused_multiplexing_benchmark.py \
        --gpu-types H100,A100,L40S,L4
    modal run experiments/modal_fused_multiplexing_benchmark.py \
        --gpu-types A100 --r-values 1,4,8,16,32,64,128
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cdp-fused-multiplexing-benchmark"
VOLUME_NAME = "cdp-fused-multiplexing-results"
VOLUME_PATH = "/results"
SUPPORTED_GPUS = ("H100", "A100", "L40S", "L4")
DEFAULT_R_VALUES = (1, 4, 8, 16, 32, 64, 128)

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.13.0")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _parse_csv(value: str, *, name: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _parse_positive_int_csv(value: str, *, name: str) -> list[int]:
    try:
        values = [int(item) for item in _parse_csv(value, name=name)]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated positive integers") from exc
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} values must be positive")
    return values


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


def _write_local_json(payload: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return output_path


@app.function(
    max_containers=3,
    image=gpu_image,
    gpu="H100",
    timeout=4 * 60 * 60,
    volumes={VOLUME_PATH: results_volume},
)
def run_gpu_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run one GPU's complete R sweep and persist after every condition."""
    import gc
    import importlib.metadata
    import math
    import os
    import socket
    import traceback

    import torch
    import torch.nn.functional as F

    run_id = str(config["run_id"])
    requested_gpu = str(config["gpu_type"])
    artifact_path = (
        Path(VOLUME_PATH)
        / f"fused_multiplexing_{requested_gpu}_{run_id}_{_timestamp()}.json"
    )
    started = time.perf_counter()
    dtype_name = str(config["dtype"])
    dtype_by_name = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fused_multiplexing_kernel_benchmark",
        "run_id": run_id,
        "requested_gpu": requested_gpu,
        "status": "running",
        "claim_eligible": False,
        "started_at": _utc_now(),
        "updated_at": _utc_now(),
        "config": config,
        "environment": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "modal_gpu_type_env": os.environ.get("MODAL_GPU_TYPE"),
        },
        "protocol": {
            "timing": (
                "CUDA events around one invocation; synchronize after every "
                "sample; warmups excluded; p50/p95 over repeated samples"
            ),
            "equal_work": (
                "every method evaluates R regimes times tokens_per_regime "
                "through one d_model -> d_ff -> d_model GELU MLP"
            ),
            "correctness_reference": (
                "sequential execution with materialized permutation-conjugated "
                "up-projection columns and down-projection rows"
            ),
            "compiled_scope": (
                "torch.compile(fullgraph=True, backend='inductor') of the exact "
                "gather/shared-GEMM/GELU/shared-GEMM/gather expression"
            ),
        },
        "limitations": [
            "This is an isolated MLP-boundary microbenchmark, not end-to-end Transformer latency.",
            "torch.compile/Inductor may fuse gathers and pointwise epilogues around GEMMs; "
            "this script does not assert that it emitted a custom fused gather-GEMM.",
            "Materialized dense baselines include their resident per-regime weight memory "
            "but exclude one-time weight construction from latency.",
            "CUDA allocator reserved memory and compiler caches can outlive a condition; "
            "allocated and reserved peaks are both reported.",
            "Results from different GPU types are separate jobs and are not simultaneous.",
        ],
        "cases": [],
    }

    def persist() -> None:
        result["updated_at"] = _utc_now()
        result["elapsed_seconds"] = time.perf_counter() - started
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        temporary.replace(artifact_path)
        results_volume.commit()

    def unsupported(reason: str, detail: str | None = None) -> dict[str, Any]:
        result["status"] = "unsupported"
        result["unsupported_reason"] = reason
        if detail is not None:
            result["unsupported_detail"] = detail
        result["completed_at"] = _utc_now()
        persist()
        return json.loads(json.dumps(result, default=str))

    if not torch.cuda.is_available():
        return unsupported("cuda_unavailable")
    try:
        triton_version = importlib.metadata.version("triton")
        import triton  # noqa: F401
        from torch._inductor import config as _inductor_config  # noqa: F401
    except Exception as exc:
        return unsupported("triton_or_inductor_unavailable", repr(exc))
    if dtype_name not in dtype_by_name:
        return unsupported("unsupported_dtype", dtype_name)

    device = torch.device("cuda")
    dtype = dtype_by_name[dtype_name]
    properties = torch.cuda.get_device_properties(0)
    result["environment"].update(
        {
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "cuda_runtime": torch.version.cuda,
            "triton": triton_version,
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
        }
    )
    # Modal's environment variable is not guaranteed on all runtime versions;
    # retain both the requested resource and the actual CUDA device identity.
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["allow_tf32"])

    warmups = int(config["warmups"])
    repetitions = int(config["repetitions"])
    d_model = int(config["d_model"])
    d_ff = int(config["d_ff"])
    tokens = int(config["tokens_per_regime"])
    tolerance = (
        {"atol": 1e-4, "rtol": 1e-4}
        if dtype == torch.float32
        else {"atol": 2e-2, "rtol": 2e-2}
    )

    def percentile(samples: list[float], quantile: float) -> float:
        ordered = sorted(samples)
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    def synchronize() -> None:
        torch.cuda.synchronize()

    def benchmark(
        function: Any,
        *,
        static_allocated_bytes: int,
        compile_seconds: float | None = None,
        compile_peak_allocated_bytes: int | None = None,
    ) -> tuple[dict[str, Any], Any]:
        for _ in range(warmups):
            warm_output = function()
        synchronize()
        del warm_output
        torch.cuda.reset_peak_memory_stats()
        samples_ms: list[float] = []
        output = None
        for _ in range(repetitions):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = function()
            end_event.record()
            synchronize()
            samples_ms.append(float(start_event.elapsed_time(end_event)))
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        record: dict[str, Any] = {
            "status": "ok",
            "warmups": warmups,
            "repetitions": repetitions,
            "samples_ms": samples_ms,
            "p50_ms": percentile(samples_ms, 0.50),
            "p95_ms": percentile(samples_ms, 0.95),
            "mean_ms": sum(samples_ms) / len(samples_ms),
            "min_ms": min(samples_ms),
            "max_ms": max(samples_ms),
            "tokens_per_invocation": int(output.shape[0] * output.shape[1]),
            "tokens_per_second_p50": (
                int(output.shape[0] * output.shape[1])
                / (percentile(samples_ms, 0.50) / 1000.0)
            ),
            "static_allocated_bytes": static_allocated_bytes,
            "peak_allocated_bytes": peak_allocated,
            "peak_incremental_allocated_bytes": max(
                0, peak_allocated - static_allocated_bytes
            ),
            "peak_reserved_bytes": peak_reserved,
        }
        if compile_seconds is not None:
            record["compile_seconds"] = compile_seconds
            record["compile_peak_allocated_bytes"] = compile_peak_allocated_bytes
        return record, output

    def correctness(candidate: Any, reference: Any) -> dict[str, Any]:
        candidate_float = candidate.float()
        reference_float = reference.float()
        difference = (candidate_float - reference_float).abs()
        passed = bool(
            torch.allclose(
                candidate_float,
                reference_float,
                atol=tolerance["atol"],
                rtol=tolerance["rtol"],
            )
        )
        return {
            "passed": passed,
            "atol": tolerance["atol"],
            "rtol": tolerance["rtol"],
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
        }

    def make_permutations(R: int) -> tuple[Any, Any]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config["seed"]) + R)
        forward_cpu = torch.stack(
            [torch.randperm(d_model, generator=generator) for _ in range(R)]
        )
        inverse_cpu = torch.empty_like(forward_cpu)
        source = torch.arange(d_model).expand(R, -1)
        inverse_cpu.scatter_(1, forward_cpu, source)
        return forward_cpu.to(device), inverse_cpu.to(device)

    try:
        base_up = (
            torch.randn(d_ff, d_model, device=device, dtype=dtype)
            * (1.0 / math.sqrt(d_model))
        )
        base_down = (
            torch.randn(d_model, d_ff, device=device, dtype=dtype)
            * (1.0 / math.sqrt(d_ff))
        )
        synchronize()
    except Exception as exc:
        return unsupported("base_tensor_setup_failed", repr(exc))

    for R in [int(value) for value in config["r_values"]]:
        case: dict[str, Any] = {
            "R": R,
            "shape": {
                "d_model": d_model,
                "d_ff": d_ff,
                "tokens_per_regime": tokens,
                "total_tokens": R * tokens,
            },
            "status": "running",
            "methods": {},
        }
        result["cases"].append(case)
        persist()
        reference_output = None
        try:
            forward, inverse = make_permutations(R)
            canonical_input = torch.randn(
                R, tokens, d_model, device=device, dtype=dtype
            )
            gather_index = forward[:, None, :].expand(R, tokens, d_model)
            regime_input = torch.gather(canonical_input, 2, gather_index)
            del canonical_input, gather_index

            dense_weight_bytes = (
                R * (d_ff * d_model + d_model * d_ff) * dtype.itemsize
            )
            free_bytes, _ = torch.cuda.mem_get_info()
            case["dense_weight_bytes"] = dense_weight_bytes
            case["free_bytes_before_dense_setup"] = free_bytes

            # Keep enough headroom for activations, allocator fragmentation, and
            # later compiler work. "Where applicable" is explicit in the JSON.
            dense_applicable = dense_weight_bytes <= int(free_bytes * 0.55)
            if not dense_applicable:
                skipped = {
                    "status": "skipped",
                    "reason": "materialized_regime_weights_exceed_55pct_free_memory",
                    "required_weight_bytes": dense_weight_bytes,
                    "free_bytes": free_bytes,
                }
                case["methods"]["sequential_dense"] = skipped
                case["methods"]["batched_dense"] = dict(skipped)
            else:
                regime_up = base_up[None, :, :].expand(R, -1, -1)
                regime_up = torch.gather(
                    regime_up,
                    2,
                    forward[:, None, :].expand(R, d_ff, d_model),
                ).contiguous()
                regime_down = base_down[None, :, :].expand(R, -1, -1)
                regime_down = torch.gather(
                    regime_down,
                    1,
                    forward[:, :, None].expand(R, d_model, d_ff),
                ).contiguous()
                synchronize()

                def sequential_dense(
                    regime_input=regime_input,
                    regime_up=regime_up,
                    regime_down=regime_down,
                    regime_count=R,
                ) -> Any:
                    outputs = []
                    for regime in range(regime_count):
                        hidden = F.gelu(
                            F.linear(regime_input[regime], regime_up[regime])
                        )
                        outputs.append(
                            F.linear(hidden, regime_down[regime])
                        )
                    return torch.stack(outputs)

                static = torch.cuda.memory_allocated()
                sequential_record, reference_output = benchmark(
                    sequential_dense, static_allocated_bytes=static
                )
                sequential_record["correctness"] = {
                    "passed": True,
                    "reference": True,
                }
                case["methods"]["sequential_dense"] = sequential_record
                persist()

                def batched_dense(
                    regime_input=regime_input,
                    regime_up=regime_up,
                    regime_down=regime_down,
                ) -> Any:
                    hidden = F.gelu(
                        torch.bmm(regime_input, regime_up.transpose(1, 2))
                    )
                    return torch.bmm(hidden, regime_down.transpose(1, 2))

                static = torch.cuda.memory_allocated()
                batched_record, batched_output = benchmark(
                    batched_dense, static_allocated_bytes=static
                )
                batched_record["correctness"] = correctness(
                    batched_output, reference_output
                )
                batched_record["speedup_vs_sequential_p50"] = (
                    sequential_record["p50_ms"] / batched_record["p50_ms"]
                )
                case["methods"]["batched_dense"] = batched_record
                del batched_output, regime_up, regime_down
                gc.collect()
                torch.cuda.empty_cache()
                persist()

            input_index = inverse[:, None, :].expand(R, tokens, d_model)
            output_index = forward[:, None, :].expand(R, tokens, d_model)

            def eager_multiplexing(
                regime_input=regime_input,
                input_index=input_index,
                output_index=output_index,
            ) -> Any:
                original_basis = torch.gather(regime_input, 2, input_index)
                hidden = F.gelu(F.linear(original_basis, base_up))
                original_output = F.linear(hidden, base_down)
                return torch.gather(original_output, 2, output_index)

            static = torch.cuda.memory_allocated()
            eager_record, eager_output = benchmark(
                eager_multiplexing, static_allocated_bytes=static
            )
            if reference_output is None:
                # A direct canonical formulation is a low-memory correctness
                # oracle when materialized dense baselines are inapplicable.
                canonical = torch.gather(regime_input, 2, input_index)
                canonical = F.linear(F.gelu(F.linear(canonical, base_up)), base_down)
                reference_output = torch.gather(canonical, 2, output_index)
                eager_record["correctness"] = correctness(
                    eager_output, reference_output
                )
                eager_record["correctness"]["reference"] = "canonical_low_memory"
                del canonical
            else:
                eager_record["correctness"] = correctness(
                    eager_output, reference_output
                )
                eager_record["speedup_vs_sequential_p50"] = (
                    case["methods"]["sequential_dense"]["p50_ms"]
                    / eager_record["p50_ms"]
                )
            case["methods"]["eager_masked_multiplexing"] = eager_record
            persist()

            # Compile a fresh exact-shape graph for each R. The first invocation
            # includes tracing/code generation and is never included in latency.
            compiled = torch.compile(
                eager_multiplexing,
                backend="inductor",
                fullgraph=True,
                dynamic=False,
            )
            torch.cuda.reset_peak_memory_stats()
            compile_start = time.perf_counter()
            compiled_first_output = compiled()
            synchronize()
            compile_seconds = time.perf_counter() - compile_start
            compile_peak = torch.cuda.max_memory_allocated()
            compiled_first_correctness = correctness(
                compiled_first_output, reference_output
            )
            if not compiled_first_correctness["passed"]:
                case["methods"]["compiled_exact_multiplexing"] = {
                    "status": "failed_closed",
                    "reason": "compiled_first_output_failed_correctness",
                    "compile_seconds": compile_seconds,
                    "correctness": compiled_first_correctness,
                }
                case["status"] = "failed"
            else:
                static = torch.cuda.memory_allocated()
                compiled_record, compiled_output = benchmark(
                    compiled,
                    static_allocated_bytes=static,
                    compile_seconds=compile_seconds,
                    compile_peak_allocated_bytes=compile_peak,
                )
                compiled_record["correctness"] = correctness(
                    compiled_output, reference_output
                )
                compiled_record["first_call_correctness"] = compiled_first_correctness
                compiled_record["speedup_vs_eager_p50"] = (
                    eager_record["p50_ms"] / compiled_record["p50_ms"]
                )
                if case["methods"]["sequential_dense"]["status"] == "ok":
                    compiled_record["speedup_vs_sequential_p50"] = (
                        case["methods"]["sequential_dense"]["p50_ms"]
                        / compiled_record["p50_ms"]
                    )
                case["methods"]["compiled_exact_multiplexing"] = compiled_record
                all_correct = all(
                    method.get("correctness", {}).get("passed", True)
                    for method in case["methods"].values()
                    if method.get("status") == "ok"
                )
                if not all_correct:
                    case["status"] = "failed"
                elif not dense_applicable:
                    case["status"] = "partial"
                else:
                    case["status"] = "ok"
                del compiled_output

            del (
                compiled,
                compiled_first_output,
                eager_output,
                reference_output,
                regime_input,
                forward,
                inverse,
                input_index,
                output_index,
            )
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as exc:
            case["status"] = "oom"
            case["error"] = repr(exc)
            case["traceback"] = traceback.format_exc()
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            # A compiler failure is retained as failure, never replaced by eager.
            case["status"] = "failed"
            case["error"] = repr(exc)
            case["traceback"] = traceback.format_exc()
            gc.collect()
            torch.cuda.empty_cache()
        persist()

    statuses = [case["status"] for case in result["cases"]]
    result["status"] = "completed" if all(item == "ok" for item in statuses) else "partial"
    result["claim_eligible"] = result["status"] == "completed"
    result["completed_at"] = _utc_now()
    persist()
    return json.loads(json.dumps(result, default=str))


@app.local_entrypoint()
def main(
    gpu_types: str = "H100",
    r_values: str = "1,4,8,16,32,64,128",
    d_model: int = 768,
    d_ff: int = 3072,
    tokens_per_regime: int = 32,
    warmups: int = 10,
    repetitions: int = 50,
    dtype: str = "float16",
    seed: int = 1947,
    allow_tf32: bool = False,
    smoke: bool = False,
) -> None:
    """Run selected GPU jobs and archive their returned records locally."""
    selected_gpus = [item.upper() for item in _parse_csv(gpu_types, name="gpu_types")]
    invalid_gpus = [item for item in selected_gpus if item not in SUPPORTED_GPUS]
    if invalid_gpus:
        raise ValueError(
            f"unsupported GPU type(s): {invalid_gpus}; choose from {SUPPORTED_GPUS}"
        )
    selected_r_values = _parse_positive_int_csv(r_values, name="r_values")
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16, or float32")
    for name, value in (
        ("d_model", d_model),
        ("d_ff", d_ff),
        ("tokens_per_regime", tokens_per_regime),
        ("warmups", warmups),
        ("repetitions", repetitions),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    if smoke:
        selected_r_values = [1, 4]
        tokens_per_regime = min(tokens_per_regime, 4)
        warmups = 2
        repetitions = 5

    run_id = _timestamp()
    common = {
        "run_id": run_id,
        "r_values": selected_r_values,
        "d_model": d_model,
        "d_ff": d_ff,
        "tokens_per_regime": tokens_per_regime,
        "warmups": warmups,
        "repetitions": repetitions,
        "dtype": dtype,
        "seed": seed,
        "allow_tf32": allow_tf32,
        "smoke": smoke,
        "git": _git_metadata(),
        "local_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modal": getattr(modal, "__version__", "unknown"),
        },
    }
    combined: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fused_multiplexing_kernel_benchmark_combined",
        "run_id": run_id,
        "started_at": _utc_now(),
        "config": common,
        "requested_gpus": selected_gpus,
        "gpu_results": [],
    }
    output_path = (
        Path(__file__).resolve().parent
        / "results"
        / f"fused_multiplexing_{run_id}.json"
    )
    wall_start = time.perf_counter()
    for gpu_type in selected_gpus:
        config = {**common, "gpu_type": gpu_type}
        gpu_result = run_gpu_benchmark.with_options(gpu=gpu_type).remote(config)
        combined["gpu_results"].append(gpu_result)
        # Preserve completed GPU data locally even if a later GPU job fails.
        combined["updated_at"] = _utc_now()
        combined["local_wall_seconds"] = time.perf_counter() - wall_start
        _write_local_json(combined, output_path)
        print(
            f"{gpu_type}: {gpu_result.get('status')} "
            f"({len(gpu_result.get('cases', []))} cases)"
        )
        print(f"Combined local result: {output_path}")
    combined["completed_at"] = _utc_now()
    combined["local_wall_seconds"] = time.perf_counter() - wall_start
    _write_local_json(combined, output_path)
    print(f"Final combined local result: {output_path}")
