"""Instrumented vLLM cross-adapter prefix-cache reuse experiment.

This is a serving-cache experiment, not a sequential Hugging Face surrogate.
It deliberately aliases two different LoRA adapters onto one vLLM LoRA integer
ID (the cache namespace used by vLLM releases that support this protocol),
unloading the first before loading the second.  The dirty and clean pairs are
identical except that the clean pair resets the prefix cache between requests.

The run is claim-eligible only when vLLM reports:
  * a same-adapter cache hit (positive control),
  * no hit after an explicit reset (clean control),
  * a cross-adapter hit in the deliberately shared namespace (dirty condition),
  * no hit from an engine with prefix caching disabled.

If the installed vLLM cannot switch adapters, reset the cache, or expose
per-request cached-token evidence, the script records ``status=unsupported``
and stops.  It never substitutes ordinary Transformers generation.

Usage (do not run implicitly):
    modal run experiments/modal_vllm_kv_reuse.py --smoke
    modal run experiments/modal_vllm_kv_reuse.py
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cdp-vllm-kv-reuse"
BASE_MODEL = "Qwen/Qwen2.5-0.5B"
BASE_MODEL_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
RESULTS_MOUNT = "/results"
SHARED_CACHE_NAMESPACE = 730_001

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name("cdp-vllm-kv-reuse-results", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.26.0",
    "torch==2.13.0",
    "transformers==5.14.1",
    "peft==0.20.0",
    "safetensors==0.7.0",
).env({"VLLM_USE_FLASHINFER_SAMPLER": "0"})


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _jsonable(v)
            for k, v in vars(value).items()
            if not str(k).startswith("_")
        }
    return repr(value)


@app.function(
    max_containers=3,
    image=image,
    gpu="A10",
    cpu=4,
    memory=32768,
    timeout=3600,
    volumes={RESULTS_MOUNT: results_volume},
)
def run_experiment(smoke: bool = False) -> dict[str, Any]:
    """Run on one isolated vLLM engine and persist every condition immediately."""
    import gc
    import importlib.metadata
    import os
    import platform
    import tempfile

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    started = datetime.now(timezone.utc).isoformat()
    run_id = _utc_stamp()
    remote_dir = Path(RESULTS_MOUNT) / run_id
    remote_dir.mkdir(parents=True, exist_ok=False)

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "instrumented_vllm_cross_adapter_prefix_cache_reuse",
        "run_id": run_id,
        "started_at": started,
        "status": "running",
        "claim_eligible": False,
        "smoke": smoke,
        "config": {
            "model": BASE_MODEL,
            "model_revision": BASE_MODEL_REVISION,
            "shared_cache_namespace": SHARED_CACHE_NAMESPACE,
            "seed": 1947,
            "gpu": os.environ.get("MODAL_GPU_TYPE", "A10"),
            "protocol": (
                "Unload adapter B and load distinct adapter A under the same LoRA "
                "integer ID; clean differs from dirty only by prefix-cache reset."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": importlib.metadata.version("vllm"),
            "transformers": importlib.metadata.version("transformers"),
            "peft": importlib.metadata.version("peft"),
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "conditions": {},
        "limitations": [
            "The shared namespace is a deliberate ID collision in an isolated process, "
            "not a claim that a correctly configured production router aliases tenant IDs.",
            "vLLM does not expose physical KV block IDs through its stable RequestOutput API "
            "in all releases; absence is recorded, never inferred.",
            "Random synthetic LoRAs test cache identity/reuse mechanics, not trained-secret recall.",
        ],
    }

    def persist(name: str, payload: dict[str, Any]) -> None:
        envelope = {
            "run_id": run_id,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "condition": name,
            "payload": payload,
        }
        path = remote_dir / f"{_utc_stamp()}_{name}.json"
        path.write_text(json.dumps(envelope, indent=2, sort_keys=True, default=str))
        results_volume.commit()

    def unsupported(reason: str, details: Any = None) -> dict[str, Any]:
        result["status"] = "unsupported"
        result["unsupported_reason"] = reason
        if details is not None:
            result["unsupported_details"] = _jsonable(details)
        result["completed_at"] = datetime.now(timezone.utc).isoformat()
        persist("unsupported", result)
        return json.loads(json.dumps(result, default=str))

    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def hash_tree(path: Path) -> str:
        digest = hashlib.sha256()
        for child in sorted(p for p in path.rglob("*") if p.is_file()):
            digest.update(str(child.relative_to(path)).encode())
            digest.update(bytes.fromhex(hash_file(child)))
        return digest.hexdigest()

    def make_adapters() -> dict[str, dict[str, Any]]:
        root = Path(tempfile.mkdtemp(prefix="cdp-kv-reuse-adapters-"))
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            revision=BASE_MODEL_REVISION,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(base, config)
        records: dict[str, dict[str, Any]] = {}
        # Nonzero, opposite-sign B matrices make the two adapter artifacts distinct.
        for tenant, sign in (("tenant_b", 1.0), ("tenant_a", -1.0)):
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if "lora_A" in name:
                        parameter.fill_(0.025)
                    elif "lora_B" in name:
                        parameter.fill_(sign * 0.025)
            path = root / tenant
            model.save_pretrained(path, safe_serialization=True)
            records[tenant] = {
                "tenant": tenant,
                "path": str(path),
                "artifact_sha256": hash_tree(path),
                "cache_namespace": SHARED_CACHE_NAMESPACE,
            }
        del model, base
        gc.collect()
        if records["tenant_a"]["artifact_sha256"] == records["tenant_b"]["artifact_sha256"]:
            raise RuntimeError("synthetic adapter artifacts unexpectedly have identical hashes")
        return records

    try:
        adapters = make_adapters()
        result["adapters"] = adapters
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            revision=BASE_MODEL_REVISION,
        )
        repetitions = 24 if smoke else 96
        prefix = (
            "Cache isolation audit record. Tenant-neutral fixed prefix. "
            "Every token in this record is intentionally identical across conditions. "
        ) * repetitions
        prompt = prefix + "\nAudit response:"
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        result["prompt"] = {
            "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()
            ).hexdigest(),
            "token_count": len(token_ids),
            "text_preview": prompt[:160],
        }
        persist("setup", {"adapters": adapters, "prompt": result["prompt"]})
    except Exception as exc:
        return unsupported("adapter_or_prompt_setup_failed", repr(exc))

    sampling = SamplingParams(temperature=0.0, max_tokens=1 if smoke else 4, seed=1947)

    def new_engine(prefix_caching: bool) -> Any:
        return LLM(
            model=BASE_MODEL,
            revision=BASE_MODEL_REVISION,
            enable_lora=True,
            enable_prefix_caching=prefix_caching,
            max_lora_rank=8,
            max_model_len=max(1024, len(token_ids) + 32),
            gpu_memory_utilization=0.72,
            seed=1947,
            enforce_eager=True,
            disable_log_stats=False,
        )

    def engine_method(engine: Any, method: str) -> Any:
        for owner in (engine, getattr(engine, "llm_engine", None)):
            candidate = getattr(owner, method, None) if owner is not None else None
            if callable(candidate):
                return candidate
        return None

    def reset_cache(engine: Any) -> tuple[bool, Any]:
        method = engine_method(engine, "reset_prefix_cache")
        if method is None:
            return False, "reset_prefix_cache is not exposed"
        response = method()
        # Some releases return None on success; False is the only explicit failure.
        return response is not False, _jsonable(response)

    def list_loras(engine: Any) -> Any:
        method = engine_method(engine, "list_loras")
        return _jsonable(method()) if method is not None else None

    def namespace_registered(snapshot: Any) -> bool:
        if isinstance(snapshot, dict):
            return any(namespace_registered(value) for value in snapshot.values())
        if isinstance(snapshot, list):
            return SHARED_CACHE_NAMESPACE in snapshot
        return snapshot == SHARED_CACHE_NAMESPACE

    def engine_cache_metadata(engine: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        owners = {
            "llm": engine,
            "llm_engine": getattr(engine, "llm_engine", None),
        }
        for owner_name, owner in owners.items():
            if owner is None:
                continue
            for attribute in ("cache_config", "model_config", "scheduler_config"):
                value = getattr(owner, attribute, None)
                if value is not None:
                    metadata[f"{owner_name}.{attribute}"] = _jsonable(value)
            for attribute in ("cache_engine", "block_manager", "kv_cache_manager"):
                value = getattr(owner, attribute, None)
                if value is not None:
                    metadata[f"{owner_name}.{attribute}"] = _jsonable(value)
        return metadata

    def add_lora(engine: Any, request: Any) -> tuple[bool, Any]:
        method = engine_method(engine, "add_lora")
        if method is None:
            return False, "add_lora is not exposed"
        response = method(request)
        return response is not False, _jsonable(response)

    def remove_lora(engine: Any, adapter_id: int) -> tuple[bool, Any]:
        method = engine_method(engine, "remove_lora")
        if method is None:
            return False, "remove_lora is not exposed"
        response = method(adapter_id)
        return response is not False, _jsonable(response)

    def request_for(tenant: str) -> Any:
        record = adapters[tenant]
        return LoRARequest(
            lora_name=f"{tenant}-{record['artifact_sha256'][:12]}",
            lora_int_id=SHARED_CACHE_NAMESPACE,
            lora_path=record["path"],
        )

    def output_evidence(output: Any) -> dict[str, Any]:
        evidence: dict[str, Any] = {
            "request_id": getattr(output, "request_id", None),
            "finished": getattr(output, "finished", None),
            "num_cached_tokens": getattr(output, "num_cached_tokens", None),
            "metrics": _jsonable(getattr(output, "metrics", None)),
            "kv_transfer_params": _jsonable(getattr(output, "kv_transfer_params", None)),
            "physical_block_ids": None,
            "physical_block_ids_exposed": False,
        }
        metrics = getattr(output, "metrics", None)
        if evidence["num_cached_tokens"] is None and metrics is not None:
            evidence["num_cached_tokens"] = getattr(metrics, "num_cached_tokens", None)
        for source_name, source in (("output", output), ("metrics", metrics)):
            if source is None:
                continue
            for name in dir(source):
                lowered = name.lower()
                if ("cache" in lowered or "block" in lowered) and not name.startswith("_"):
                    try:
                        evidence[f"{source_name}.{name}"] = _jsonable(getattr(source, name))
                    except Exception as exc:
                        evidence[f"{source_name}.{name}.inspection_error"] = type(exc).__name__
        outputs = getattr(output, "outputs", [])
        evidence["generated_text"] = outputs[0].text if outputs else None
        return evidence

    def generate(engine: Any, request: Any) -> dict[str, Any]:
        output = engine.generate([prompt], sampling, lora_request=request, use_tqdm=False)[0]
        return output_evidence(output)

    def run_pair(engine: Any, name: str, reset_between: bool) -> dict[str, Any]:
        reset_before_ok, reset_before = reset_cache(engine)
        if not reset_before_ok:
            raise RuntimeError(f"cannot reset before {name}: {reset_before}")
        initial_loras = list_loras(engine)
        initial_remove = None
        if namespace_registered(initial_loras):
            initial_remove_ok, initial_remove = remove_lora(
                engine, SHARED_CACHE_NAMESPACE
            )
            if not initial_remove_ok:
                raise RuntimeError(
                    f"cannot remove prior adapter before {name}: {initial_remove}"
                )
        b_request = request_for("tenant_b")
        a_request = request_for("tenant_a")
        add_b_ok, add_b = add_lora(engine, b_request)
        if not add_b_ok:
            raise RuntimeError(f"cannot add tenant B: {add_b}")
        before_prime = list_loras(engine)
        if not namespace_registered(before_prime):
            raise RuntimeError("tenant B namespace not registered before prime")
        prime = generate(engine, b_request)
        remove_ok, remove_result = remove_lora(engine, SHARED_CACHE_NAMESPACE)
        after_remove = list_loras(engine)
        if not remove_ok:
            raise RuntimeError(f"cannot remove tenant B: {remove_result}")
        if namespace_registered(after_remove):
            raise RuntimeError("tenant B namespace remained registered after remove_lora")
        add_a_ok, add_a = add_lora(engine, a_request)
        if not add_a_ok:
            raise RuntimeError(f"cannot add tenant A: {add_a}")
        before_probe = list_loras(engine)
        if not namespace_registered(before_probe):
            raise RuntimeError("tenant A namespace not registered before probe")
        between_reset = None
        if reset_between:
            reset_ok, between_reset = reset_cache(engine)
            if not reset_ok:
                raise RuntimeError(f"cannot reset between requests: {between_reset}")
        probe = generate(engine, a_request)
        record = {
            "name": name,
            "prefix_caching_enabled": True,
            "reset_between_requests": reset_between,
            "cache_namespace": SHARED_CACHE_NAMESPACE,
            "engine_cache_metadata": engine_cache_metadata(engine),
            "prompt": result["prompt"],
            "prime_adapter": adapters["tenant_b"],
            "probe_adapter": adapters["tenant_a"],
            "runtime": {
                "initial_loras": initial_loras,
                "initial_remove": initial_remove,
                "before_prime_loras": before_prime,
                "after_remove_loras": after_remove,
                "before_probe_loras": before_probe,
                "add_b": add_b,
                "remove_b": remove_result,
                "add_a": add_a,
                "reset_before": reset_before,
                "reset_between": between_reset,
            },
            "prime": prime,
            "probe": probe,
        }
        persist(name, record)
        return record

    try:
        engine = new_engine(prefix_caching=True)
        if engine_method(engine, "add_lora") is None or engine_method(engine, "remove_lora") is None:
            return unsupported(
                "installed_vllm_cannot_explicitly_switch_loras",
                {"vllm": result["environment"]["vllm"]},
            )
        if engine_method(engine, "list_loras") is None:
            return unsupported(
                "installed_vllm_cannot_confirm_active_lora_identity",
                {"vllm": result["environment"]["vllm"]},
            )
        if engine_method(engine, "reset_prefix_cache") is None:
            return unsupported(
                "installed_vllm_cannot_reset_prefix_cache",
                {"vllm": result["environment"]["vllm"]},
            )

        clean = run_pair(engine, "cross_tenant_clean", reset_between=True)
        result["conditions"]["cross_tenant_clean"] = clean
        dirty = run_pair(engine, "cross_tenant_dirty", reset_between=False)
        result["conditions"]["cross_tenant_dirty"] = dirty

        # Positive control uses the same adapter and namespace for both requests.
        ok, reset_detail = reset_cache(engine)
        if not ok:
            return unsupported("same_tenant_control_reset_failed", reset_detail)
        a_request = request_for("tenant_a")
        remove_lora(engine, SHARED_CACHE_NAMESPACE)
        add_ok, add_detail = add_lora(engine, a_request)
        if not add_ok:
            return unsupported("same_tenant_control_adapter_load_failed", add_detail)
        same_prime = generate(engine, a_request)
        same_probe = generate(engine, a_request)
        same = {
            "name": "same_tenant_positive",
            "prefix_caching_enabled": True,
            "reset_between_requests": False,
            "cache_namespace": SHARED_CACHE_NAMESPACE,
            "prompt": result["prompt"],
            "prime_adapter": adapters["tenant_a"],
            "probe_adapter": adapters["tenant_a"],
            "reset_before": reset_detail,
            "prime": same_prime,
            "probe": same_probe,
        }
        persist("same_tenant_positive", same)
        result["conditions"]["same_tenant_positive"] = same
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        return unsupported("prefix_cache_protocol_could_not_be_executed", repr(exc))

    try:
        disabled_engine = new_engine(prefix_caching=False)
        a_request = request_for("tenant_a")
        add_ok, add_detail = add_lora(disabled_engine, a_request)
        if not add_ok:
            return unsupported("cache_disabled_adapter_load_failed", add_detail)
        disabled_prime = generate(disabled_engine, a_request)
        disabled_probe = generate(disabled_engine, a_request)
        disabled = {
            "name": "cache_disabled",
            "prefix_caching_enabled": False,
            "reset_between_requests": False,
            "cache_namespace": SHARED_CACHE_NAMESPACE,
            "engine_cache_metadata": engine_cache_metadata(disabled_engine),
            "prompt": result["prompt"],
            "prime_adapter": adapters["tenant_a"],
            "probe_adapter": adapters["tenant_a"],
            "prime": disabled_prime,
            "probe": disabled_probe,
        }
        result["conditions"]["cache_disabled"] = disabled
        persist("cache_disabled", disabled)
        del disabled_engine
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as exc:
        return unsupported("cache_disabled_control_could_not_be_executed", repr(exc))

    def cached_tokens(condition: str) -> int | None:
        value = result["conditions"][condition]["probe"].get("num_cached_tokens")
        return value if isinstance(value, int) and not isinstance(value, bool) else None

    observed = {
        name: cached_tokens(name)
        for name in (
            "cross_tenant_clean",
            "cross_tenant_dirty",
            "same_tenant_positive",
            "cache_disabled",
        )
    }
    result["cache_hit_summary"] = observed
    if any(value is None for value in observed.values()):
        return unsupported(
            "vllm_did_not_expose_per_request_num_cached_tokens_for_all_conditions",
            observed,
        )
    if observed["same_tenant_positive"] <= 0:
        return unsupported("same_tenant_positive_control_did_not_hit_cache", observed)
    if observed["cross_tenant_clean"] != 0:
        return unsupported("clean_reset_control_reported_cached_tokens", observed)
    if observed["cache_disabled"] != 0:
        return unsupported("cache_disabled_control_reported_cached_tokens", observed)
    if observed["cross_tenant_dirty"] <= 0:
        return unsupported(
            "cross_adapter_reuse_could_not_be_forced_or_observed_safely",
            observed,
        )

    result["status"] = "supported"
    result["claim_eligible"] = True
    result["completed_at"] = datetime.now(timezone.utc).isoformat()
    persist("combined", result)
    return json.loads(json.dumps(result, default=str))


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    """Launch explicitly via Modal and archive the combined return locally."""
    started = time.time()
    result = run_experiment.remote(smoke=smoke)
    result["local_wall_seconds"] = time.time() - started
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"vllm_kv_reuse_{_utc_stamp()}.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(f"Status: {result.get('status')}")
    if result.get("unsupported_reason"):
        print(f"Unsupported: {result['unsupported_reason']}")
    print(f"Combined local result: {output_path}")
