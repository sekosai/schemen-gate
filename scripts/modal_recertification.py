"""Deterministic planning and receipt validation for Modal re-certification."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "research" / "cdp" / "experiments" / "modal-recertification.json"
RESULTS_DIR = ROOT / "research" / "cdp" / "experiments" / "results"
RELEASE_CONTRACT = ROOT / "release-contract.json"
RELEASE_MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"
MAX_CONFIG_BYTES = 256 * 1024
MAX_PLAN_BYTES = 2 * 1024 * 1024
MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MODAL_CLIENT_VERSION = "1.5.4"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SAFE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")


class RecertificationError(RuntimeError):
    """A fail-closed planning or receipt-validation error."""


def _reject_constant(value: str) -> None:
    raise RecertificationError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RecertificationError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def load_json(path: Path, *, maximum_bytes: int) -> Any:
    """Load bounded strict JSON, rejecting duplicate keys and NaN values."""

    if not path.is_file() or path.is_symlink():
        raise RecertificationError(f"expected one regular JSON file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > maximum_bytes:
        raise RecertificationError(f"JSON file size {size} is outside 1..{maximum_bytes}: {path}")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecertificationError(f"invalid JSON: {path}") from exc


def canonical_json(value: Any) -> bytes:
    """Return the single byte representation used for plan and ledger seals."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(*arguments: str) -> str:
    try:
        return subprocess.run(  # nosec B603
            ("git", *arguments),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecertificationError(f"Git inspection failed: {' '.join(arguments)}") from exc


def require_clean_head() -> str:
    commit = _git("rev-parse", "HEAD")
    if _COMMIT.fullmatch(commit) is None:
        raise RecertificationError("Git HEAD is not a canonical SHA-1 commit id")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RecertificationError("re-certification requires a completely clean checkout")
    return commit


@dataclass(frozen=True)
class Job:
    """One allowlisted local entrypoint and its bounded evidence contract."""

    id: str
    app_name: str
    launcher: str
    artifact_prefix: str | None
    expected_experiment: str | None
    gpu: str | None
    historical_seconds: Mapping[str, int]
    max_wall_seconds: Mapping[str, int]
    remote_invocations: Mapping[str, int]
    artifact_records: Mapping[str, int]
    arguments: Mapping[str, tuple[str, ...]]

    def variant(self, name: str) -> tuple[str, ...]:
        try:
            return self.arguments[name]
        except KeyError as exc:
            raise RecertificationError(f"job {self.id!r} has no {name!r} variant") from exc


@dataclass(frozen=True)
class Config:
    release_version: str
    pricing: Mapping[str, Any]
    campaigns: Mapping[str, Mapping[str, Any]]
    jobs: tuple[Job, ...]
    sha256: str


def _string_map(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise RecertificationError(f"{field} must be a nonempty object")
    result: dict[str, int] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, int) or item <= 0:
            raise RecertificationError(f"{field} values must be positive integers")
        result[key] = item
    return result


def load_config() -> Config:
    raw = load_json(CONFIG_PATH, maximum_bytes=MAX_CONFIG_BYTES)
    if not isinstance(raw, dict):
        raise RecertificationError("re-certification config must be an object")
    expected = {
        "schema",
        "release_version",
        "pricing",
        "campaigns",
        "jobs",
    }
    if set(raw) != expected or raw["schema"] != "schemen-gate/modal-recertification-config-v1":
        raise RecertificationError("unexpected re-certification config schema")
    if raw["release_version"] != "1.0.2":
        raise RecertificationError("re-certification config must target Gate 1.0.2")
    if not isinstance(raw["pricing"], dict) or not isinstance(raw["campaigns"], dict):
        raise RecertificationError("pricing and campaigns must be objects")
    raw_jobs = raw["jobs"]
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise RecertificationError("jobs must be a nonempty array")
    jobs: list[Job] = []
    seen: set[str] = set()
    for raw_job in raw_jobs:
        if not isinstance(raw_job, dict):
            raise RecertificationError("each job must be an object")
        required = {
            "id",
            "app_name",
            "launcher",
            "artifact_prefix",
            "expected_experiment",
            "gpu",
            "historical_seconds",
            "max_wall_seconds",
            "remote_invocations",
            "arguments",
        }
        allowed = required | {"artifact_records"}
        if not required.issubset(raw_job) or not set(raw_job).issubset(allowed):
            raise RecertificationError("job fields do not match the v1 schema")
        job_id = raw_job["id"]
        app_name = raw_job["app_name"]
        launcher = raw_job["launcher"]
        if not isinstance(job_id, str) or _SAFE_ID.fullmatch(job_id) is None or job_id in seen:
            raise RecertificationError(f"invalid or duplicate job id: {job_id!r}")
        seen.add(job_id)
        if not isinstance(app_name, str) or _SAFE_ID.fullmatch(app_name) is None:
            raise RecertificationError(f"job {job_id!r} has an invalid Modal app name")
        if not isinstance(launcher, str):
            raise RecertificationError(f"job {job_id!r} launcher must be a string")
        launcher_path = Path(launcher)
        if launcher_path.is_absolute() or ".." in launcher_path.parts:
            raise RecertificationError(f"job {job_id!r} launcher escapes the repository")
        absolute_launcher = ROOT / launcher_path
        if not absolute_launcher.is_file() or absolute_launcher.is_symlink():
            raise RecertificationError(f"job {job_id!r} launcher is not a regular file")
        raw_arguments = raw_job["arguments"]
        if not isinstance(raw_arguments, dict) or not raw_arguments:
            raise RecertificationError(f"job {job_id!r} arguments must be an object")
        arguments: dict[str, tuple[str, ...]] = {}
        for variant, items in raw_arguments.items():
            if (
                not isinstance(variant, str)
                or not isinstance(items, list)
                or any(not isinstance(item, str) or "\x00" in item for item in items)
            ):
                raise RecertificationError(f"job {job_id!r} has invalid arguments")
            arguments[variant] = tuple(items)
        prefix = raw_job["artifact_prefix"]
        experiment = raw_job["expected_experiment"]
        gpu = raw_job["gpu"]
        if prefix is not None and (
            not isinstance(prefix, str) or not prefix or "/" in prefix or ".." in prefix
        ):
            raise RecertificationError(f"job {job_id!r} has an invalid artifact prefix")
        if experiment is not None and not isinstance(experiment, str):
            raise RecertificationError(f"job {job_id!r} has an invalid experiment name")
        if gpu not in {None, "A100", "T4"}:
            raise RecertificationError(f"job {job_id!r} uses an unpriced GPU")
        historical_seconds = _string_map(raw_job["historical_seconds"], field="historical_seconds")
        max_wall_seconds = _string_map(raw_job["max_wall_seconds"], field="max_wall_seconds")
        remote_invocations = _string_map(raw_job["remote_invocations"], field="remote_invocations")
        artifact_records = (
            _string_map(raw_job["artifact_records"], field="artifact_records")
            if "artifact_records" in raw_job
            else {}
        )
        variants = set(arguments)
        if (
            set(historical_seconds) != variants
            or set(max_wall_seconds) != variants
            or set(remote_invocations) != variants
            or (prefix is None and artifact_records)
            or (prefix is not None and set(artifact_records) != variants)
        ):
            raise RecertificationError(
                f"job {job_id!r} resource and artifact maps must match its variants"
            )
        jobs.append(
            Job(
                id=job_id,
                app_name=app_name,
                launcher=launcher,
                artifact_prefix=prefix,
                expected_experiment=experiment,
                gpu=gpu,
                historical_seconds=historical_seconds,
                max_wall_seconds=max_wall_seconds,
                remote_invocations=remote_invocations,
                artifact_records=artifact_records,
                arguments=arguments,
            )
        )
    return Config(
        release_version=raw["release_version"],
        pricing=raw["pricing"],
        campaigns=raw["campaigns"],
        jobs=tuple(jobs),
        sha256=sha256_file(CONFIG_PATH),
    )


def _require_release(config: Config) -> Mapping[str, Any]:
    contract = load_json(RELEASE_CONTRACT, maximum_bytes=64 * 1024)
    if not isinstance(contract, dict):
        raise RecertificationError("release contract must be an object")
    if (
        contract.get("schema") != "schemen/gate-release-contract-v1"
        or contract.get("package") != "schemen-gate"
        or contract.get("version") != config.release_version
        or contract.get("tag") != f"v{config.release_version}"
        or contract.get("repository") != "https://github.com/sekosai/schemen-gate"
    ):
        raise RecertificationError("release contract does not match the campaign")
    return contract


def _plan_step(job: Job, variant: str) -> dict[str, Any]:
    launcher = ROOT / job.launcher
    records = job.artifact_records.get(variant)
    return {
        "id": job.id,
        "app_name": job.app_name,
        "launcher": job.launcher,
        "launcher_sha256": sha256_file(launcher),
        "arguments": list(job.variant(variant)),
        "artifact_prefix": job.artifact_prefix,
        "expected_experiment": job.expected_experiment,
        "expected_artifact_records": records,
        "gpu": job.gpu,
        "historical_seconds": job.historical_seconds[variant],
        "max_wall_seconds": job.max_wall_seconds[variant],
        "remote_invocations": job.remote_invocations[variant],
        "variant": variant,
    }


def build_plan(campaign: str) -> dict[str, Any]:
    """Build a deterministic exact-HEAD plan without importing Modal launchers."""

    config = load_config()
    try:
        campaign_config = config.campaigns[campaign]
    except KeyError as exc:
        raise RecertificationError(f"unknown campaign: {campaign!r}") from exc
    if not isinstance(campaign_config, dict):
        raise RecertificationError("campaign config must be an object")
    variant = campaign_config.get("full_variant")
    if not isinstance(variant, str):
        raise RecertificationError("campaign full_variant must be a string")
    commit = require_clean_head()
    contract = _require_release(config)

    canary_steps = [_plan_step(job, "canary") for job in config.jobs if "canary" in job.arguments]
    full_steps = [_plan_step(job, variant) for job in config.jobs if variant in job.arguments]
    base = {
        "schema": "schemen-gate/modal-recertification-plan-v1",
        "campaign": campaign,
        "description": campaign_config.get("description"),
        "release": {
            "version": config.release_version,
            "repository": contract["repository"],
            "source_commit": commit,
            "release_contract_sha256": sha256_file(RELEASE_CONTRACT),
            "release_manifest_sha256": sha256_file(RELEASE_MANIFEST),
            "config_sha256": config.sha256,
        },
        "pricing": config.pricing,
        "execution": {
            "modal_client_version": MODAL_CLIENT_VERSION,
            "sequential_launchers": True,
            "automatic_retries": 0,
        },
        "estimate": {
            "expected_gross_usd": campaign_config.get("expected_gross_usd"),
            "campaign_approval_ceiling_usd": campaign_config.get("campaign_approval_ceiling_usd"),
            "credits_included": False,
        },
        "stages": {
            "canary": canary_steps,
            "full": full_steps,
        },
    }
    plan_sha256 = sha256_bytes(canonical_json(base))
    return {**base, "plan_sha256": plan_sha256}


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict) or plan.get("schema") != (
        "schemen-gate/modal-recertification-plan-v1"
    ):
        raise RecertificationError("unexpected plan schema")
    supplied_hash = plan.get("plan_sha256")
    if not isinstance(supplied_hash, str) or _SHA256.fullmatch(supplied_hash) is None:
        raise RecertificationError("plan has no canonical SHA-256 seal")
    base = dict(plan)
    del base["plan_sha256"]
    if sha256_bytes(canonical_json(base)) != supplied_hash:
        raise RecertificationError("plan SHA-256 does not match its contents")
    campaign = plan.get("campaign")
    if not isinstance(campaign, str):
        raise RecertificationError("plan campaign is invalid")
    expected = build_plan(campaign)
    if canonical_json(plan) != canonical_json(expected):
        raise RecertificationError("plan differs from the current clean checkout")
    return expected


def load_plan(path: Path) -> dict[str, Any]:
    return validate_plan(load_json(path, maximum_bytes=MAX_PLAN_BYTES))


def jobs_for_stage(plan: Mapping[str, Any], stage: str) -> tuple[Mapping[str, Any], ...]:
    stages = plan.get("stages")
    if not isinstance(stages, dict) or stage not in {"canary", "full"}:
        raise RecertificationError("stage must be canary or full")
    jobs = stages.get(stage)
    if not isinstance(jobs, list) or not jobs:
        raise RecertificationError(f"plan has no jobs for stage {stage!r}")
    if any(not isinstance(job, dict) for job in jobs):
        raise RecertificationError("plan stage contains an invalid job")
    return tuple(jobs)


def estimate_historical_gpu_cost(plan: Mapping[str, Any], stage: str) -> float:
    pricing = plan.get("pricing")
    if not isinstance(pricing, dict) or not isinstance(pricing.get("rates_per_second"), dict):
        raise RecertificationError("plan pricing is invalid")
    rates = pricing["rates_per_second"]
    total = 0.0
    for job in jobs_for_stage(plan, stage):
        gpu = job.get("gpu")
        seconds = job.get("historical_seconds")
        if not isinstance(seconds, int) or seconds <= 0:
            raise RecertificationError("job historical seconds are invalid")
        if gpu is None:
            continue
        rate = rates.get(gpu)
        if not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise RecertificationError(f"missing price for GPU {gpu!r}")
        total += seconds * float(rate)
    return total


def _require_runtime_rejections(record: Mapping[str, Any]) -> None:
    runtime = record.get("runtime")
    probe = runtime.get("rejection_probe") if isinstance(runtime, dict) else None
    if (
        not isinstance(probe, dict)
        or probe.get("all_rejected") is not True
        or probe.get("unauthorized_model_calls") != 0
    ):
        raise RecertificationError("runtime refusal evidence did not pass")


def _require_zeroes(record: Mapping[str, Any], names: Iterable[str]) -> None:
    for name in names:
        value = record.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != 0:
            raise RecertificationError(f"exact-zero assertion failed: {name}")


_EMPTY_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
_AG_MODEL = "distilbert-base-uncased"
_AG_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
_AG_DATASET = "fancyzhx/ag_news"
_AG_DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
_GATE_PLACEMENT = "post-activation-pre-down-projection"


def _require_exact_json(actual: Any, expected: Any, *, field: str) -> None:
    try:
        matches = canonical_json(actual) == canonical_json(expected)
    except (TypeError, ValueError) as exc:
        raise RecertificationError(f"{field} is not finite canonical JSON") from exc
    if not matches:
        raise RecertificationError(f"{field} does not match the planned value")


def _require_finite_number(value: Any, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RecertificationError(f"{field} must be a finite JSON number")
    return float(value)


def _require_integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise RecertificationError(f"{field} must be an integer >= {minimum}")
    return value


def _arguments(job: Mapping[str, Any]) -> dict[str, str | bool]:
    raw = job.get("arguments")
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise RecertificationError("planned launcher arguments are invalid")
    result: dict[str, str | bool] = {}
    index = 0
    while index < len(raw):
        name = raw[index]
        if not name.startswith("--") or name in result:
            raise RecertificationError("planned launcher arguments are not canonical")
        if index + 1 < len(raw) and not raw[index + 1].startswith("--"):
            result[name] = raw[index + 1]
            index += 2
        else:
            result[name] = True
            index += 1
    return result


def _require_options(options: Mapping[str, str | bool], allowed: set[str]) -> None:
    if not set(options).issubset(allowed):
        raise RecertificationError("planned launcher arguments contain an unknown option")


def _option_text(options: Mapping[str, str | bool], name: str, default: str) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or not value:
        raise RecertificationError(f"planned option {name} requires a value")
    return value


def _option_int(options: Mapping[str, str | bool], name: str, default: int) -> int:
    try:
        return int(_option_text(options, name, str(default)))
    except ValueError as exc:
        raise RecertificationError(f"planned option {name} must be an integer") from exc


def _option_float(options: Mapping[str, str | bool], name: str, default: float) -> float:
    try:
        value = float(_option_text(options, name, str(default)))
    except ValueError as exc:
        raise RecertificationError(f"planned option {name} must be numeric") from exc
    if not math.isfinite(value):
        raise RecertificationError(f"planned option {name} must be finite")
    return value


def _csv_integers(value: str, *, field: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise RecertificationError(f"planned {field} must be comma-separated integers") from exc
    if not result or len(result) != len(set(result)):
        raise RecertificationError(f"planned {field} must be nonempty and unique")
    return result


def _csv_text(value: str, *, field: str) -> list[str]:
    result = [item for item in value.split(",") if item]
    if not result or len(result) != len(set(result)):
        raise RecertificationError(f"planned {field} must be nonempty and unique")
    return result


def _smoke(options: Mapping[str, str | bool]) -> bool:
    value = options.get("--smoke", False)
    if not isinstance(value, bool):
        raise RecertificationError("--smoke does not accept a value")
    return value


def _require_config_multiset(
    records: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> None:
    try:
        actual_values = sorted(canonical_json(record.get("config")) for record in records)
        expected_values = sorted(canonical_json(value) for value in expected)
    except (TypeError, ValueError) as exc:
        raise RecertificationError("execution config is not finite canonical JSON") from exc
    if actual_values != expected_values:
        raise RecertificationError("artifact execution configs differ from the plan")


def _expected_dense_configs(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    options = _arguments(job)
    _require_options(options, {"--r-values", "--seeds", "--smoke"})
    smoke = _smoke(options)
    ratios = (
        [2]
        if smoke
        else _csv_integers(_option_text(options, "--r-values", "2,4,8,16"), field="R values")
    )
    seeds = (
        [42]
        if smoke
        else _csv_integers(_option_text(options, "--seeds", "42,123,256"), field="seeds")
    )
    return [
        {
            "R": ratio,
            "seed": seed,
            "train_examples": 256 if smoke else 8000,
            "test_examples": 256 if smoke else 2000,
            "max_length": 64 if smoke else 128,
            "batch_size": 32,
            "epochs": 1 if smoke else 2,
            "ffn_learning_rate": 2e-5,
            "head_learning_rate": 1e-3,
            "smoke": smoke,
        }
        for ratio in ratios
        for seed in seeds
    ]


def _expected_private_configs(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    options = _arguments(job)
    _require_options(options, {"--designs", "--seeds", "--smoke"})
    smoke = _smoke(options)
    designs = _csv_text(_option_text(options, "--designs", "adapter,expert"), field="designs")
    seeds = _csv_integers(_option_text(options, "--seeds", "42,123,256"), field="seeds")
    if smoke:
        designs = designs[:1]
        seeds = seeds[:1]
    if any(design not in {"adapter", "expert"} for design in designs):
        raise RecertificationError("planned private-lane design is unsupported")
    return [
        {
            "design": design,
            "seed": seed,
            "regimes": 4,
            "train_examples": 256 if smoke else 8000,
            "test_examples": 256 if smoke else 2000,
            "max_length": 64 if smoke else 128,
            "batch_size": 32,
            "epochs": 1 if smoke else 2,
            "learning_rate": 5e-4 if design == "adapter" else 1e-4,
            "smoke": smoke,
        }
        for design in designs
        for seed in seeds
    ]


def _factorial_config(seed: int, *, reduced: bool) -> dict[str, Any]:
    return {
        "seed": seed,
        "R": 8,
        "public_examples": 256 if reduced else 8000,
        "tenant_examples": 256 if reduced else 8000,
        "test_examples": 256 if reduced else 2000,
        "max_length": 64 if reduced else 128,
        "batch_size": 32,
        "public_epochs": 1 if reduced else 2,
        "adaptation_epochs": 1,
        "tenant_epochs": 1 if reduced else 2,
        "public_encoder_lr": 2e-5,
        "public_head_lr": 1e-3,
        "adaptation_encoder_lr": 2e-5,
        "adaptation_head_lr": 5e-4,
        "tenant_ffn_lr": 2e-5,
        "tenant_head_lr": 1e-3,
        "temperature": 2.0,
        "hard_loss_weight": 0.5,
        "smoke": reduced,
    }


def _factorial_seeds(job: Mapping[str, Any]) -> tuple[list[int], bool]:
    options = _arguments(job)
    _require_options(options, {"--seeds", "--smoke"})
    smoke = _smoke(options)
    seeds = _csv_integers(_option_text(options, "--seeds", "42,123,256,512,1024"), field="seeds")
    return ([42] if smoke else seeds), smoke


def _expected_orthogonal_config(job: Mapping[str, Any]) -> dict[str, Any]:
    options = _arguments(job)
    _require_options(options, {"--ratios", "--smoke"})
    smoke = _smoke(options)
    ratios = (
        [8] if smoke else _csv_integers(_option_text(options, "--ratios", "8,128"), field="ratios")
    )
    return {
        "seed": 0,
        "ratios": ratios,
        "train_examples": 256 if smoke else 8000,
        "test_examples": 256 if smoke else 7600,
        "max_length": 64 if smoke else 128,
        "batch_size": 64,
        "epochs": 1 if smoke else 2,
        "learning_rate": 2e-5,
        "smoke": smoke,
    }


def _expected_generative_configs(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    options = _arguments(job)
    allowed = {
        "--seeds",
        "--r",
        "--epochs",
        "--lr",
        "--seq-len",
        "--batch-size",
        "--eval-batch-size",
        "--max-train-texts",
        "--max-eval-texts",
        "--max-train-blocks",
        "--max-eval-blocks",
        "--max-steps",
        "--canaries-per-regime",
        "--canary-repeats",
        "--canary-max-new-tokens",
        "--max-grad-norm",
        "--smoke",
    }
    _require_options(options, allowed)
    smoke = _smoke(options)
    seeds = _csv_integers(_option_text(options, "--seeds", "11,22,33,44,55"), field="seeds")
    common: dict[str, Any] = {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "r": _option_int(options, "--r", 4),
        "epochs": _option_int(options, "--epochs", 3),
        "lr": _option_float(options, "--lr", 2e-5),
        "seq_len": _option_int(options, "--seq-len", 256),
        "batch_size": _option_int(options, "--batch-size", 2),
        "eval_batch_size": _option_int(options, "--eval-batch-size", 4),
        "max_train_texts": _option_int(options, "--max-train-texts", 5000),
        "max_eval_texts": _option_int(options, "--max-eval-texts", 1000),
        "max_train_blocks": _option_int(options, "--max-train-blocks", 512),
        "max_eval_blocks": _option_int(options, "--max-eval-blocks", 128),
        "max_steps": _option_int(options, "--max-steps", 0),
        "canaries_per_regime": _option_int(options, "--canaries-per-regime", 2),
        "canary_repeats": _option_int(options, "--canary-repeats", 50),
        "canary_max_new_tokens": _option_int(options, "--canary-max-new-tokens", 24),
        "max_grad_norm": _option_float(options, "--max-grad-norm", 1.0),
        "smoke": smoke,
    }
    if smoke:
        common.update(
            {
                "epochs": 1,
                "max_train_texts": min(common["max_train_texts"], 128),
                "max_eval_texts": min(common["max_eval_texts"], 64),
                "max_train_blocks": min(common["max_train_blocks"], 8),
                "max_eval_blocks": min(common["max_eval_blocks"], 4),
                "max_steps": 2,
                "canaries_per_regime": min(common["canaries_per_regime"], 1),
                "canary_repeats": min(common["canary_repeats"], 2),
                "canary_max_new_tokens": min(common["canary_max_new_tokens"], 12),
            }
        )
    combined = {**common, "seeds": seeds}
    return combined, [{**common, "seed": seed} for seed in seeds]


def _expected_service_config(job: Mapping[str, Any]) -> dict[str, Any]:
    options = _arguments(job)
    _require_options(options, {"--smoke"})
    smoke = _smoke(options)
    return {
        "seed": 42,
        "R": 8,
        "examples": 64 if smoke else 872,
        "max_length": 32 if smoke else 128,
        "batch_size": 8 if smoke else 32,
        "warmups": 2 if smoke else 10,
        "repetitions": 4 if smoke else 50,
        "smoke": smoke,
    }


def _execution_records(
    artifact: Mapping[str, Any], job: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    job_id = job.get("id")
    expected = job.get("expected_artifact_records")
    remote_invocations = job.get("remote_invocations")
    if (
        not isinstance(expected, int)
        or isinstance(expected, bool)
        or expected <= 0
        or not isinstance(remote_invocations, int)
        or isinstance(remote_invocations, bool)
        or remote_invocations <= 0
    ):
        raise RecertificationError("plan result and invocation counts are invalid")
    if job_id in {
        "dense-ffn-cotenancy",
        "private-transformer-lanes",
        "public-gate-adaptation-factorial",
    }:
        results = artifact.get("results")
        if (
            not isinstance(results, list)
            or len(results) != expected
            or any(not isinstance(result, dict) for result in results)
        ):
            raise RecertificationError("artifact has the wrong result records")
        records: list[Mapping[str, Any]] = list(results)
        if job_id == "public-gate-adaptation-factorial":
            pilot = artifact.get("preflight_pilot")
            if job.get("variant") == "canary":
                if pilot is not None:
                    raise RecertificationError("canary factorial artifact has an unexpected pilot")
            elif not isinstance(pilot, dict):
                raise RecertificationError("full factorial artifact lacks its preflight pilot")
            else:
                records.insert(0, pilot)
    elif job_id == "generative-intermediate":
        jobs = artifact.get("jobs")
        if (
            not isinstance(jobs, list)
            or len(jobs) != expected
            or any(not isinstance(item, dict) for item in jobs)
        ):
            raise RecertificationError("generative artifact has the wrong job records")
        records = list(jobs)
    elif job_id in {
        "cargo-transformer-authorization",
        "runtime-orthogonal-superposition",
        "distilbert-service-consolidation",
    }:
        result = artifact.get("result")
        if expected != 1 or not isinstance(result, dict):
            raise RecertificationError("single-job artifact has the wrong result record")
        records = [result]
    else:
        raise RecertificationError(f"no result schema exists for job {job_id!r}")
    if len(records) != remote_invocations:
        raise RecertificationError("artifact remote execution count differs from the plan")
    return tuple(records)


def _require_local_provenance(
    source: Mapping[str, Any],
    *,
    commit: str,
    launcher_sha256: str,
    version: str,
    repository: str,
) -> Mapping[str, Any]:
    dependency = source.get("dependency_bundle")
    gate_source = dependency.get("gate_source") if isinstance(dependency, dict) else None
    package_files = gate_source.get("package_files") if isinstance(gate_source, dict) else None
    tree_sha256 = gate_source.get("tree_sha256") if isinstance(gate_source, dict) else None
    if (
        source.get("commit") != commit
        or source.get("dirty") is not False
        or source.get("status_sha256") != _EMPTY_STATUS_SHA256
        or source.get("script_sha256") != launcher_sha256
        or not isinstance(dependency, dict)
        or dependency.get("mismatches") != []
        or not isinstance(gate_source, dict)
        or gate_source.get("repository") != repository
        or gate_source.get("version") != version
        or gate_source.get("commit") != commit
        or gate_source.get("dirty") is not False
        or not isinstance(tree_sha256, str)
        or _SHA256.fullmatch(tree_sha256) is None
        or not isinstance(package_files, list)
        or not package_files
        or len(package_files) != len(set(package_files))
        or any(not isinstance(name, str) or Path(name).name != name for name in package_files)
    ):
        raise RecertificationError("artifact local Gate provenance does not match the plan")
    lock_sha256 = dependency.get("lock_sha256")
    if not isinstance(lock_sha256, str) or _SHA256.fullmatch(lock_sha256) is None:
        raise RecertificationError("artifact library-lock provenance is invalid")
    return gate_source


def _require_provenance(
    artifact: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    commit: str,
    launcher_sha256: str,
    version: str,
    repository: str,
    require_top_source: bool,
) -> None:
    verified_sources: list[Mapping[str, Any]] = []
    for record in records:
        source = record.get("source")
        if not isinstance(source, dict):
            raise RecertificationError("remote result lacks its own source provenance")
        gate_source = _require_local_provenance(
            source,
            commit=commit,
            launcher_sha256=launcher_sha256,
            version=version,
            repository=repository,
        )
        remote = source.get("remote_verification")
        if (
            not isinstance(remote, dict)
            or remote.get("gate_version") != version
            or remote.get("gate_repository") != repository
            or remote.get("gate_source_commit") != commit
            or remote.get("gate_tree_sha256") != gate_source.get("tree_sha256")
            or remote.get("script_sha256") != launcher_sha256
        ):
            raise RecertificationError("remote Gate provenance does not match local custody")
        verified_sources.append(source)
    baseline = canonical_json(verified_sources[0])
    if any(canonical_json(source) != baseline for source in verified_sources[1:]):
        raise RecertificationError("remote result provenance records are inconsistent")
    top_source = artifact.get("source")
    if require_top_source and not isinstance(top_source, dict):
        raise RecertificationError("combined artifact lacks local source provenance")
    if isinstance(top_source, dict):
        _require_local_provenance(
            top_source,
            commit=commit,
            launcher_sha256=launcher_sha256,
            version=version,
            repository=repository,
        )
        expected_top = dict(verified_sources[0])
        expected_top.pop("remote_verification", None)
        _require_exact_json(top_source, expected_top, field="combined source provenance")


def _require_child_identity(
    record: Mapping[str, Any], *, experiment: str, schema_version: int
) -> None:
    if record.get("experiment") != experiment or record.get("schema_version") != schema_version:
        raise RecertificationError("remote result schema identity is invalid")


def _require_ag_assets(record: Mapping[str, Any]) -> None:
    assets = record.get("assets")
    expected = {
        "model": _AG_MODEL,
        "model_revision": _AG_MODEL_REVISION,
        "dataset": _AG_DATASET,
        "dataset_revision": _AG_DATASET_REVISION,
    }
    _require_exact_json(assets, expected, field="pinned AG News assets")


def _require_cargo_assets(record: Mapping[str, Any]) -> None:
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_revision = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    generation_model = "google/flan-t5-small"
    generation_revision = "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
    generation_digest = hashlib.sha256(
        f"{generation_model}@{generation_revision}".encode()
    ).hexdigest()
    model_digest = hashlib.sha256(
        (
            f"{embedding_model}@{embedding_revision}|{generation_model}@{generation_revision}"
        ).encode()
    ).hexdigest()
    _require_exact_json(
        record.get("assets"),
        {
            "embedding_model": embedding_model,
            "embedding_model_revision": embedding_revision,
            "generation_model": generation_model,
            "generation_model_revision": generation_revision,
            "generation_model_digest": generation_digest,
            "model_digest": model_digest,
        },
        field="pinned Cargo assets",
    )


def _require_generative_assets(record: Mapping[str, Any]) -> None:
    _require_exact_json(
        record.get("assets"),
        {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_revision": "77e23968eed12d195bd46c519aa679cc22a27ddc",
            "dataset": "Salesforce/wikitext",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        },
        field="pinned generative assets",
    )


def _require_service_assets(record: Mapping[str, Any]) -> None:
    _require_exact_json(
        record.get("assets"),
        {
            "model": ("distilbert/distilbert-base-uncased-finetuned-sst-2-english"),
            "model_revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
            "dataset": "nyu-mll/glue",
            "dataset_config": "sst2",
            "dataset_revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        },
        field="pinned service assets",
    )


def _validate_dense(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    results = records
    _require_config_multiset(results, _expected_dense_configs(job))
    zero_fields = (
        "maximum_frozen_shared_parameter_delta",
        "maximum_off_partition_parameter_delta",
        "maximum_off_partition_optimizer_moment",
        "maximum_inactive_classifier_delta",
    )
    for result in results:
        _require_child_identity(result, experiment="dense_ffn_cotenancy", schema_version=1)
        if result.get("status") != "pass":
            raise RecertificationError("dense separation result did not pass")
        _require_ag_assets(result)
        if (
            result.get("gate_placement") != _GATE_PLACEMENT
            or result.get("partition_algorithm") != "schemen_gate.GateMask.derive@1.0.2"
        ):
            raise RecertificationError("dense Gate identity or placement is invalid")
        _require_zeroes(result, zero_fields)
        _require_runtime_rejections(result)


def _validate_private(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    del artifact
    results = records
    _require_config_multiset(results, _expected_private_configs(job))
    for result in results:
        _require_child_identity(result, experiment="private_transformer_lanes", schema_version=1)
        if result.get("status") != "pass":
            raise RecertificationError("private-lane result did not pass")
        _require_ag_assets(result)
        if result.get("gate_placement") != ("gate-authorized-hard-route-to-complete-private-lane"):
            raise RecertificationError("private-lane Gate placement is invalid")
        _require_zeroes(
            result,
            ("maximum_shared_backbone_delta", "maximum_inactive_lane_delta"),
        )
        _require_runtime_rejections(result)


def _validate_factorial_result(result: Mapping[str, Any]) -> None:
    _require_child_identity(result, experiment="public_gate_adaptation_factorial", schema_version=2)
    if result.get("status") != "pass":
        raise RecertificationError("public-factorial result did not pass")
    _require_ag_assets(result)
    if (
        result.get("gate_placement") != _GATE_PLACEMENT
        or result.get("partition_algorithm") != "schemen_gate.GateMask.derive@1.0.2"
        or result.get("public_and_tenant_splits_disjoint") is not True
    ):
        raise RecertificationError("public-factorial Gate or split evidence is invalid")
    equal_compute = result.get("equal_compute_adapted_conditions")
    equal_flags = (
        "same_public_examples",
        "same_batch_order",
        "same_dropout_seed_schedule",
        "same_optimizer_steps",
        "same_teacher_forward_passes",
        "same_forward_backward_passes",
    )
    config = result.get("config")
    if (
        not isinstance(equal_compute, dict)
        or any(equal_compute.get(name) is not True for name in equal_flags)
        or not isinstance(config, dict)
        or equal_compute.get("ungated_repetitions_per_batch") != config.get("R")
    ):
        raise RecertificationError("public-factorial equal-compute evidence is invalid")
    conditions = result.get("conditions")
    expected_conditions = {
        "no_extra_public_adaptation",
        "ungated_hard_label",
        "all_mask_hard_label",
        "all_mask_distillation",
    }
    if not isinstance(conditions, dict) or set(conditions) != expected_conditions:
        raise RecertificationError("public-factorial conditions are incomplete")
    zero_fields = (
        "maximum_frozen_shared_parameter_delta",
        "maximum_off_partition_parameter_delta",
        "maximum_off_partition_optimizer_moment",
        "maximum_inactive_classifier_delta",
    )
    for name, condition in conditions.items():
        if (
            not isinstance(condition, dict)
            or condition.get("condition") != name
            or condition.get("separation_pass") is not True
        ):
            raise RecertificationError("public-factorial condition did not pass")
        _require_zeroes(condition, zero_fields)
    _require_runtime_rejections(result)


def _validate_factorial(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    results = artifact.get("results")
    summary = artifact.get("summary")
    seeds, smoke = _factorial_seeds(job)
    if (
        not isinstance(results, list)
        or not isinstance(summary, dict)
        or summary.get("all_separation_checks_pass") is not True
        or summary.get("seeds") != seeds
    ):
        raise RecertificationError("public-factorial summary did not pass")
    pilot = artifact.get("preflight_pilot")
    if smoke:
        _require_config_multiset(results, [_factorial_config(42, reduced=True)])
    else:
        if not isinstance(pilot, dict):
            raise RecertificationError("public-factorial pilot is missing")
        _require_exact_json(
            pilot.get("config"),
            _factorial_config(seeds[0], reduced=True),
            field="public-factorial pilot config",
        )
        _validate_factorial_result(pilot)
        _require_config_multiset(
            results,
            [_factorial_config(seed, reduced=False) for seed in seeds],
        )
    for result in results:
        if not isinstance(result, dict):
            raise RecertificationError("public-factorial result is invalid")
        _validate_factorial_result(result)
    if len(records) != len(results) + (0 if smoke else 1):
        raise RecertificationError("public-factorial execution records are inconsistent")


def _validate_cargo(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    del artifact, job
    result = records[0]
    _require_child_identity(
        result, experiment="cargo_transformer_corpus_authorization", schema_version=3
    )
    recall = result.get("owning_exact_recall")
    unauthorized_calls = result.get("unauthorized_model_calls")
    if (
        result.get("status") != "pass"
        or not isinstance(recall, float)
        or recall != 1.0
        or result.get("all_unauthorized_docks_rejected") is not True
        or not isinstance(unauthorized_calls, int)
        or isinstance(unauthorized_calls, bool)
        or unauthorized_calls != 0
        or result.get("load_receipts_valid") is not True
        or result.get("retrieval_receipts_valid") is not True
    ):
        raise RecertificationError("Cargo acceptance contract did not pass")
    _require_cargo_assets(result)
    owning_rows = result.get("owning_rows")
    if not isinstance(owning_rows, list) or len(owning_rows) != 4:
        raise RecertificationError("Cargo owning rows are incomplete")
    expected_owners = {
        ("customer-alpha", "claims-team"): 2,
        ("customer-beta", "renewals-team"): 2,
    }
    actual_owners: dict[tuple[str, str], int] = {}
    for row in owning_rows:
        if (
            not isinstance(row, dict)
            or row.get("correct") is not True
            or row.get("retrieved_partition") != row.get("tenant_id")
            or not isinstance(row.get("tenant_id"), str)
            or not isinstance(row.get("sub_id"), str)
        ):
            raise RecertificationError("Cargo owning-row evidence is invalid")
        key = (row["tenant_id"], row["sub_id"])
        actual_owners[key] = actual_owners.get(key, 0) + 1
    if actual_owners != expected_owners:
        raise RecertificationError("Cargo owning-row coverage is invalid")
    rejection_rows = result.get("rejection_rows")
    if not isinstance(rejection_rows, list) or len(rejection_rows) != 23:
        raise RecertificationError("Cargo rejection rows are incomplete")
    common_attacks = {
        "random",
        "other_operation_same_scope",
        "other_tenant",
        "other_sub_id_same_tenant",
        "other_regime_same_tenant",
        "other_model_same_identity",
        "other_policy_same_identity",
        "runtime_other_model_digest",
        "runtime_other_regime",
        "runtime_wrong_dimensions",
        "runtime_empty_authority",
    }
    expected_attacks = {
        (tenant, attack)
        for tenant in ("customer-alpha", "customer-beta")
        for attack in common_attacks
    } | {("customer-alpha-same-regime-alias", "other_partition_same_regime_valid_origin_key")}
    actual_attacks: set[tuple[str, str]] = set()
    expected_subjects = {
        "customer-alpha": "claims-team",
        "customer-beta": "renewals-team",
        "customer-alpha-same-regime-alias": "claims-team",
    }
    for row in rejection_rows:
        target = row.get("target") if isinstance(row, dict) else None
        attack = row.get("attack") if isinstance(row, dict) else None
        if (
            not isinstance(row, dict)
            or row.get("rejected") is not True
            or not isinstance(row.get("model_calls"), int)
            or isinstance(row.get("model_calls"), bool)
            or row.get("model_calls") != 0
            or not isinstance(target, str)
            or not isinstance(attack, str)
            or row.get("target_sub_id") != expected_subjects.get(target)
        ):
            raise RecertificationError("Cargo rejection-row evidence is invalid")
        actual_attacks.add((target, attack))
    if actual_attacks != expected_attacks or len(actual_attacks) != len(rejection_rows):
        raise RecertificationError("Cargo rejection attack coverage is invalid")


def _validate_orthogonal(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    del artifact
    result = records[0]
    expected_config = _expected_orthogonal_config(job)
    _require_exact_json(result.get("config"), expected_config, field="orthogonal config")
    _require_child_identity(result, experiment="orthogonal_superposition", schema_version=1)
    _require_ag_assets(result)
    ratios = result.get("ratios") if isinstance(result, dict) else None
    if (
        result.get("status") != "pass"
        or result.get("gate_placement") != "gate-authorized-whole-model-activation"
        or result.get("scope")
        != (
            "Whole-model permutation conjugation and serial addressed use; "
            "not sparse FFN capacity partitioning or concurrent cotenancy."
        )
        or not isinstance(ratios, list)
        or len(ratios) != len(expected_config["ratios"])
    ):
        raise RecertificationError("orthogonal-placement result did not pass")
    if [row.get("R") if isinstance(row, dict) else None for row in ratios] != (
        expected_config["ratios"]
    ):
        raise RecertificationError("orthogonal-placement ratios differ from the plan")
    for expected_ratio, ratio in zip(expected_config["ratios"], ratios, strict=True):
        if (
            not isinstance(ratio, dict)
            or ratio.get("status") != "pass"
            or ratio.get("accuracy_zero_loss") is not True
            or ratio.get("evaluated_regimes") != expected_ratio
        ):
            raise RecertificationError("orthogonal-placement ratio did not pass")
        _require_zeroes(ratio, ("maximum_absolute_accuracy_gap",))
        for field in (
            "baseline_accuracy",
            "mean_accuracy",
            "minimum_accuracy",
            "maximum_accuracy",
            "maximum_absolute_logit_difference",
        ):
            value = _require_finite_number(ratio.get(field), field=field)
            if field != "maximum_absolute_logit_difference" and not 0 <= value <= 1:
                raise RecertificationError(f"{field} is outside 0..1")
        _require_runtime_rejections(ratio)


def _validate_generative(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    jobs = artifact.get("jobs")
    combined_config, expected_configs = _expected_generative_configs(job)
    if artifact.get("all_jobs_ok") is not True or not isinstance(jobs, list):
        raise RecertificationError("generative artifact did not pass")
    _require_exact_json(artifact.get("config"), combined_config, field="generative config")
    _require_config_multiset(records, expected_configs)
    for job in jobs:
        summary = job.get("summary") if isinstance(job, dict) else None
        if (
            not isinstance(job, dict)
            or job.get("status") != "pass"
            or not isinstance(summary, dict)
            or summary.get("confinement_probe_passed") is not True
        ):
            raise RecertificationError("generative confinement did not pass")
        _require_child_identity(job, experiment="tinyllama_intermediate_ffn_gate", schema_version=1)
        _require_generative_assets(job)
        environment = job.get("environment")
        determinism = environment.get("determinism") if isinstance(environment, dict) else None
        if not isinstance(determinism, dict):
            raise RecertificationError("generative deterministic execution evidence is missing")
        expected_determinism = {
            "deterministic_algorithms_enabled": True,
            "deterministic_algorithms_warn_only": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "sdpa_policy": "pytorch_strict_deterministic_with_math_fallback",
        }
        _require_exact_json(
            {key: determinism.get(key) for key in expected_determinism},
            expected_determinism,
            field="generative deterministic execution evidence",
        )
        sdpa_backends = determinism.get("sdpa_backends")
        if (
            set(determinism) != {*expected_determinism, "sdpa_backends"}
            or not isinstance(sdpa_backends, dict)
            or set(sdpa_backends)
            != {
                "math",
                "flash",
                "memory_efficient",
                "cudnn",
            }
            or sdpa_backends.get("math") is not True
            or any(not isinstance(value, bool) for value in sdpa_backends.values())
        ):
            raise RecertificationError(
                "generative deterministic execution SDPA evidence is invalid"
            )
        gate = job.get("gate")
        config = job.get("config")
        if (
            not isinstance(gate, dict)
            or not isinstance(config, dict)
            or gate.get("partition_disjoint") is not True
            or gate.get("partition_complete") is not True
            or gate.get("semantic_location")
            != ("post-SwiGLU expanded activation, pre-down-projection")
        ):
            raise RecertificationError("generative Gate partition evidence is invalid")
        mask_key_sha256 = gate.get("mask_key_sha256")
        if not isinstance(mask_key_sha256, str) or _SHA256.fullmatch(mask_key_sha256) is None:
            raise RecertificationError("generative Gate key digest is invalid")
        intermediate = _require_integer(
            gate.get("intermediate_size"), field="intermediate_size", minimum=1
        )
        active = _require_integer(
            gate.get("active_dimensions_per_regime"),
            field="active_dimensions_per_regime",
            minimum=1,
        )
        regimes = _require_integer(config.get("r"), field="R", minimum=2)
        if active * regimes != intermediate:
            raise RecertificationError("generative Gate partition dimensions are invalid")
        matched = job.get("matched_initialization")
        if not isinstance(matched, dict) or matched.get("exact_match") is not True:
            raise RecertificationError("generative matched initialization did not pass")
        gated_fingerprint = matched.get("gated_fingerprint")
        control_fingerprint = matched.get("ungated_control_fingerprint")
        if (
            not isinstance(gated_fingerprint, str)
            or _SHA256.fullmatch(gated_fingerprint) is None
            or gated_fingerprint != control_fingerprint
        ):
            raise RecertificationError("generative initialization fingerprints differ")
        conditions = job.get("conditions")
        if not isinstance(conditions, dict) or set(conditions) != {
            "gated",
            "ungated_control",
        }:
            raise RecertificationError("generative matched conditions are incomplete")
        trainings: dict[str, Mapping[str, Any]] = {}
        for name, condition in conditions.items():
            training = condition.get("training") if isinstance(condition, dict) else None
            if not isinstance(training, dict):
                raise RecertificationError("generative training evidence is incomplete")
            if training.get("initialization_fingerprint") != gated_fingerprint:
                raise RecertificationError("generative training initialization drifted")
            trainings[name] = training
        gated_training = trainings["gated"]
        control_training = trainings["ungated_control"]
        gated_steps = _require_integer(
            gated_training.get("optimizer_steps"), field="gated optimizer steps", minimum=1
        )
        if control_training.get("optimizer_steps") != gated_steps:
            raise RecertificationError("generative optimizer-step counts are unmatched")
        gated_history = gated_training.get("history")
        control_history = control_training.get("history")
        if not isinstance(gated_history, list) or not isinstance(control_history, list):
            raise RecertificationError("generative training histories are incomplete")
        if [item.get("optimizer_steps") for item in gated_history if isinstance(item, dict)] != [
            item.get("optimizer_steps") for item in control_history if isinstance(item, dict)
        ] or len(gated_history) != len(control_history):
            raise RecertificationError("generative per-epoch step schedules are unmatched")
        probe = gated_training.get("confinement_probe")
        if (
            not isinstance(probe, dict)
            or probe.get("passed") is not True
            or probe.get("all_inactive_exact_zero") is not True
            or probe.get("any_active_delta_nonzero") is not True
            or control_training.get("confinement_probe") is not None
        ):
            raise RecertificationError("generative raw confinement probe did not pass")
        gradients = probe.get("gradient_checks")
        deltas = probe.get("parameter_delta_checks")
        if not isinstance(gradients, dict) or not isinstance(deltas, dict):
            raise RecertificationError("generative confinement measurements are missing")
        _require_zeroes(gradients, tuple(gradients))
        inactive_names = tuple(name for name in deltas if "inactive" in name)
        _require_zeroes(deltas, inactive_names)
        active_values = [
            _require_finite_number(value, field=name)
            for name, value in deltas.items()
            if "active" in name and "inactive" not in name
        ]
        if not active_values or not any(value > 0 for value in active_values):
            raise RecertificationError("generative confinement changed no active parameter")
        _require_runtime_rejections(job)


def _validate_service(
    artifact: Mapping[str, Any], job: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    del artifact
    result = records[0]
    expected_config = _expected_service_config(job)
    _require_exact_json(result.get("config"), expected_config, field="service config")
    _require_child_identity(
        result, experiment="distilbert_runtime_service_consolidation", schema_version=1
    )
    _require_service_assets(result)
    extraction = result.get("extraction_equivalence") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("status") != "pass"
        or not isinstance(extraction, dict)
        or extraction.get("within_tolerance") is not True
    ):
        raise RecertificationError("service-consolidation result did not pass")
    difference = _require_finite_number(
        extraction.get("max_absolute_logit_difference"),
        field="max_absolute_logit_difference",
    )
    tolerance = _require_finite_number(extraction.get("tolerance"), field="tolerance")
    if difference < 0 or tolerance != 0.02 or difference > tolerance:
        raise RecertificationError("service extraction measurement exceeds tolerance")
    conditions = result.get("conditions")
    required_conditions = {
        "separate_services",
        "shared_backbone_private_adapters",
        "shared_ffn_authorized_slices",
        "physically_extracted_authorized_slice",
    }
    if not isinstance(conditions, dict) or set(conditions) != required_conditions:
        raise RecertificationError("service-consolidation conditions are incomplete")
    for name, condition in conditions.items():
        if not isinstance(condition, dict):
            raise RecertificationError(f"service condition {name} is invalid")
        for field in (
            "checkpoint_tensor_bytes",
            "resident_cuda_parameter_and_buffer_bytes",
            "peak_cuda_allocated_bytes_during_load",
            "active_parameter_bytes_per_request",
        ):
            _require_integer(condition.get(field), field=f"{name}.{field}", minimum=1)
        ready = _require_finite_number(
            condition.get("warm_cache_serial_construction_to_ready_seconds"),
            field=f"{name}.ready_seconds",
        )
        accuracy = _require_finite_number(
            condition.get("utility_accuracy"), field=f"{name}.utility_accuracy"
        )
        correct = _require_integer(
            condition.get("utility_correct"), field=f"{name}.utility_correct"
        )
        total = _require_integer(
            condition.get("utility_total"), field=f"{name}.utility_total", minimum=1
        )
        if ready < 0 or not 0 <= accuracy <= 1 or correct > total:
            raise RecertificationError(f"service condition {name} has invalid measurements")
        if not math.isclose(accuracy, correct / total, rel_tol=0.0, abs_tol=1e-15):
            raise RecertificationError(f"service condition {name} utility is inconsistent")
        timing = condition.get("timing")
        if not isinstance(timing, dict):
            raise RecertificationError(f"service condition {name} lacks timing evidence")
        if (
            timing.get("requests") != expected_config["repetitions"]
            or timing.get("samples")
            != expected_config["repetitions"] * expected_config["batch_size"]
        ):
            raise RecertificationError(f"service condition {name} timing count is invalid")
        for field in (
            "p50_latency_ms",
            "p95_latency_ms",
            "mean_latency_ms",
            "throughput_samples_per_second",
        ):
            if _require_finite_number(timing.get(field), field=f"{name}.timing.{field}") <= 0:
                raise RecertificationError(f"service condition {name} timing is nonpositive")
    _require_runtime_rejections(result)


_VALIDATORS: Mapping[
    str,
    Callable[
        [Mapping[str, Any], Mapping[str, Any], Sequence[Mapping[str, Any]]],
        None,
    ],
] = {
    "dense-ffn-cotenancy": _validate_dense,
    "private-transformer-lanes": _validate_private,
    "public-gate-adaptation-factorial": _validate_factorial,
    "cargo-transformer-authorization": _validate_cargo,
    "runtime-orthogonal-superposition": _validate_orthogonal,
    "generative-intermediate": _validate_generative,
    "distilbert-service-consolidation": _validate_service,
}


def validate_artifact(path: Path, job: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    """Validate one newly returned local artifact and return its SHA-256."""

    artifact = load_json(path, maximum_bytes=MAX_ARTIFACT_BYTES)
    if not isinstance(artifact, dict):
        raise RecertificationError("experiment artifact must be an object")
    if artifact.get("experiment") != job.get("expected_experiment"):
        raise RecertificationError("experiment artifact identity does not match the plan")
    release = plan.get("release")
    if not isinstance(release, dict) or not isinstance(release.get("source_commit"), str):
        raise RecertificationError("plan release identity is invalid")
    launcher_sha256 = job.get("launcher_sha256")
    if not isinstance(launcher_sha256, str):
        raise RecertificationError("plan launcher digest is invalid")
    job_id = job.get("id")
    validator = _VALIDATORS.get(job_id) if isinstance(job_id, str) else None
    if validator is None:
        raise RecertificationError(f"no validator exists for job {job_id!r}")
    version = release.get("version")
    repository = release.get("repository")
    if not isinstance(version, str) or not isinstance(repository, str):
        raise RecertificationError("plan Gate release version or repository is invalid")
    records = _execution_records(artifact, job)
    _require_provenance(
        artifact,
        records,
        commit=release["source_commit"],
        launcher_sha256=launcher_sha256,
        version=version,
        repository=repository,
        require_top_source=job_id
        in {
            "private-transformer-lanes",
            "public-gate-adaptation-factorial",
            "runtime-orthogonal-superposition",
            "generative-intermediate",
            "distilbert-service-consolidation",
        },
    )
    validator(artifact, job, records)
    return sha256_file(path)


__all__ = [
    "CONFIG_PATH",
    "MAX_ARTIFACT_BYTES",
    "MAX_PLAN_BYTES",
    "MODAL_CLIENT_VERSION",
    "RESULTS_DIR",
    "ROOT",
    "RecertificationError",
    "build_plan",
    "canonical_json",
    "estimate_historical_gpu_cost",
    "jobs_for_stage",
    "load_json",
    "load_plan",
    "require_clean_head",
    "sha256_bytes",
    "sha256_file",
    "validate_artifact",
    "validate_plan",
]
