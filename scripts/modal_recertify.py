#!/usr/bin/env python3
"""Plan and execute fail-closed Modal re-certification campaigns.

Planning and checking are local-only operations. Execution invokes the pinned
Modal CLI as a child process; this module never imports experiment launchers or
the Modal Python package.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, MutableMapping, NoReturn, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.modal_recertification import (
    MAX_ARTIFACT_BYTES,
    MODAL_CLIENT_VERSION,
    RESULTS_DIR,
    ROOT,
    RecertificationError,
    build_plan,
    canonical_json,
    estimate_historical_gpu_cost,
    jobs_for_stage,
    load_json,
    load_plan,
    require_clean_head,
    sha256_bytes,
    sha256_file,
    validate_artifact,
)

LEDGER_SCHEMA = "schemen-gate/modal-recertification-ledger-v1"
MAX_LEDGER_BYTES = 4 * 1024 * 1024
_SAFE_ENVIRONMENT = re.compile(r"[a-z0-9][a-z0-9-]{0,79}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _decimal(value: Any, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise RecertificationError(f"{field} must be a finite monetary amount")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise RecertificationError(f"{field} must be a finite monetary amount") from exc
    if not result.is_finite() or result < 0:
        raise RecertificationError(f"{field} must be a finite nonnegative amount")
    return result


def _atomic_json(path: Path, value: Any) -> None:
    """Write canonical JSON without ever following an existing destination symlink."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise RecertificationError(f"refusing to replace symlink: {path}")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise RecertificationError(f"could not write JSON atomically: {path}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sealed_ledger(ledger: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(ledger)
    base.pop("ledger_sha256", None)
    return {**base, "ledger_sha256": sha256_bytes(canonical_json(base))}


def _write_ledger(path: Path, ledger: Mapping[str, Any]) -> None:
    _atomic_json(path, _sealed_ledger(ledger))


def _validate_ledger_seal(ledger: Any) -> dict[str, Any]:
    if not isinstance(ledger, dict) or ledger.get("schema") != LEDGER_SCHEMA:
        raise RecertificationError("unexpected re-certification ledger schema")
    supplied = ledger.get("ledger_sha256")
    if not isinstance(supplied, str) or _SHA256.fullmatch(supplied) is None:
        raise RecertificationError("re-certification ledger has no SHA-256 seal")
    base = dict(ledger)
    del base["ledger_sha256"]
    if sha256_bytes(canonical_json(base)) != supplied:
        raise RecertificationError("re-certification ledger SHA-256 does not match")
    return ledger


def _release(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    release = plan.get("release")
    if not isinstance(release, dict):
        raise RecertificationError("plan release identity is invalid")
    return release


def _plan_hash(plan: Mapping[str, Any]) -> str:
    value = plan.get("plan_sha256")
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise RecertificationError("plan has no canonical SHA-256 seal")
    return value


def _source_commit(plan: Mapping[str, Any]) -> str:
    value = _release(plan).get("source_commit")
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise RecertificationError("plan source commit is invalid")
    return value


def _campaign(plan: Mapping[str, Any]) -> str:
    value = plan.get("campaign")
    if not isinstance(value, str):
        raise RecertificationError("plan campaign is invalid")
    return value


def _cost_contract(plan: Mapping[str, Any]) -> tuple[Decimal, Decimal, Decimal]:
    estimate = plan.get("estimate")
    if not isinstance(estimate, dict):
        raise RecertificationError("plan cost estimate is invalid")
    expected = estimate.get("expected_gross_usd")
    if not isinstance(expected, dict):
        raise RecertificationError("plan gross-cost range is invalid")
    low = _decimal(expected.get("low"), field="expected low cost")
    high = _decimal(expected.get("high"), field="expected high cost")
    ceiling = _decimal(
        estimate.get("campaign_approval_ceiling_usd"),
        field="campaign approval ceiling",
    )
    if low > high or high > ceiling or low <= 0:
        raise RecertificationError("plan cost range and campaign approval ceiling are inconsistent")
    return low, high, ceiling


def _validate_approval(
    plan: Mapping[str, Any],
    *,
    approved_plan_sha256: str,
    approved_max_usd: str,
) -> Decimal:
    if approved_plan_sha256 != _plan_hash(plan):
        raise RecertificationError("approval does not name this exact plan SHA-256")
    _low, high, ceiling = _cost_contract(plan)
    approved = _decimal(approved_max_usd, field="approved maximum")
    if approved < high:
        raise RecertificationError(
            f"approved maximum ${approved} is below estimated high cost ${high}"
        )
    if approved > ceiling:
        raise RecertificationError(
            f"approved maximum ${approved} exceeds the plan ceiling ${ceiling}"
        )
    return approved


def _external_root(path: Path) -> Path:
    if not path.is_absolute():
        raise RecertificationError("evidence root must be an absolute path outside the repo")
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise RecertificationError("existing evidence root must be a real directory")
    resolved = path.resolve(strict=False)
    repository = ROOT.resolve()
    if resolved == repository or repository in resolved.parents:
        raise RecertificationError("evidence root must be outside the repository")
    try:
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise RecertificationError(f"could not create evidence root: {resolved}") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise RecertificationError("evidence root did not resolve to a real directory")
    return resolved


def _modal_binary(path: Path) -> Path:
    if not path.is_absolute():
        raise RecertificationError("Modal binary path must be absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RecertificationError("Modal binary path does not exist") from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RecertificationError("Modal binary is not a regular executable file")
    return resolved


def _modal_version(modal_binary: Path) -> str:
    try:
        completed = subprocess.run(  # nosec B603
            (str(modal_binary), "--version"),
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecertificationError("could not verify the Modal client version") from exc
    output = (completed.stdout + completed.stderr).strip()
    expected = f"modal client version: {MODAL_CLIENT_VERSION}"
    if completed.returncode != 0 or output != expected:
        raise RecertificationError(f"expected {expected!r}; received {output!r}")
    return MODAL_CLIENT_VERSION


def _require_auth(modal_binary: Path) -> None:
    try:
        completed = subprocess.run(  # nosec B603
            (str(modal_binary), "token", "info"),
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RecertificationError("could not verify Modal authentication") from exc
    if completed.returncode != 0:
        raise RecertificationError("Modal authentication is not available")


def _assert_checkout(plan: Mapping[str, Any]) -> None:
    if require_clean_head() != _source_commit(plan):
        raise RecertificationError("checkout HEAD changed after the plan was sealed")


def _matching_artifacts(prefix: str) -> dict[str, Path]:
    if not RESULTS_DIR.is_dir() or RESULTS_DIR.is_symlink():
        raise RecertificationError("experiment results directory is unavailable")
    result: dict[str, Path] = {}
    for path in RESULTS_DIR.iterdir():
        if path.name.startswith(prefix) and path.suffix == ".json":
            if not path.is_file() or path.is_symlink():
                raise RecertificationError("artifact candidate is not a regular file")
            result[path.name] = path
    return result


def _copy_to_custody(source: Path, directory: Path) -> dict[str, Any]:
    if not source.is_file() or source.is_symlink():
        raise RecertificationError("refusing custody of a non-regular artifact")
    size = source.stat().st_size
    if size <= 0 or size > MAX_ARTIFACT_BYTES:
        raise RecertificationError("artifact size is outside the custody limit")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = directory / source.name
    if destination.exists() or destination.is_symlink():
        raise RecertificationError(f"artifact custody destination already exists: {destination}")
    temporary = directory / f".{source.name}.{os.getpid()}.tmp"
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        source_digest = sha256_file(source)
        if sha256_file(temporary) != source_digest:
            raise RecertificationError("artifact changed while copying to evidence custody")
        os.replace(temporary, destination)
        source.unlink()
    except OSError as exc:
        raise RecertificationError("could not move artifact into evidence custody") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return {
        "relative_path": destination.relative_to(directory.parent).as_posix(),
        "sha256": source_digest,
        "size_bytes": size,
    }


def _custody_all(paths: Sequence[Path], rejected_directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.append(_copy_to_custody(path, rejected_directory))
    return records


def _validate_canary_ledger(
    path: Path,
    *,
    plan: Mapping[str, Any],
    environment: str,
    campaign_root: Path,
) -> str:
    expected_path = campaign_root / "canary" / "ledger.json"
    if path.resolve(strict=False) != expected_path.resolve(strict=False):
        raise RecertificationError("full execution requires this evidence root's canary ledger")
    ledger = _validate_ledger_seal(load_json(path, maximum_bytes=MAX_LEDGER_BYTES))
    if (
        ledger.get("status") != "pass"
        or ledger.get("stage") != "canary"
        or ledger.get("plan_sha256") != _plan_hash(plan)
        or ledger.get("source_commit") != _source_commit(plan)
        or ledger.get("campaign") != _campaign(plan)
        or ledger.get("environment") != environment
    ):
        raise RecertificationError("canary ledger does not satisfy this full campaign")
    records = ledger.get("jobs")
    expected_jobs = jobs_for_stage(plan, "canary")
    if not isinstance(records, list) or len(records) != len(expected_jobs):
        raise RecertificationError("canary ledger job set is incomplete")
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise RecertificationError("canary ledger contains an invalid job record")
        if record["id"] in by_id or record.get("status") != "pass":
            raise RecertificationError("canary ledger contains duplicate or failed jobs")
        by_id[record["id"]] = record
    for job in expected_jobs:
        job_id = job.get("id")
        record = by_id.get(job_id) if isinstance(job_id, str) else None
        if record is None:
            raise RecertificationError("canary ledger is missing a planned job")
        artifact = record.get("artifact")
        if job.get("artifact_prefix") is None:
            if artifact is not None:
                raise RecertificationError("CPU canary ledger unexpectedly names an artifact")
            continue
        if not isinstance(artifact, dict):
            raise RecertificationError("canary ledger is missing an artifact receipt")
        relative = artifact.get("relative_path")
        digest = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise RecertificationError("canary artifact receipt is invalid")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RecertificationError("canary artifact path escapes its stage")
        artifact_path = path.parent / relative_path
        stage_root = path.parent.resolve()
        resolved_artifact = artifact_path.resolve(strict=False)
        if (
            stage_root not in resolved_artifact.parents
            or not artifact_path.is_file()
            or artifact_path.is_symlink()
            or sha256_file(artifact_path) != digest
        ):
            raise RecertificationError("canary artifact custody hash does not match")
        if artifact_path.stat().st_size != artifact.get("size_bytes"):
            raise RecertificationError("canary artifact custody size does not match")
        if validate_artifact(artifact_path, job, plan) != digest:
            raise RecertificationError("canary artifact no longer satisfies its contract")
    return sha256_file(path)


def _job_record(job: Mapping[str, Any]) -> dict[str, Any]:
    arguments = job.get("arguments")
    if not isinstance(arguments, list) or any(not isinstance(arg, str) for arg in arguments):
        raise RecertificationError("plan job arguments are invalid")
    return {
        "id": job.get("id"),
        "launcher": job.get("launcher"),
        "launcher_sha256": job.get("launcher_sha256"),
        "arguments": list(arguments),
        "gpu": job.get("gpu"),
        "remote_invocations": job.get("remote_invocations"),
        "max_wall_seconds": job.get("max_wall_seconds"),
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "exit_code": None,
        "elapsed_seconds": None,
        "remote_stop": None,
        "artifact": None,
        "rejected_artifacts": [],
        "error": None,
    }


def _stop_modal_app(modal_binary: Path, *, environment: str, app_name: str) -> dict[str, Any]:
    """Best-effort termination receipt after an ambiguous client outcome."""

    try:
        completed = subprocess.run(  # nosec B603
            (
                str(modal_binary),
                "app",
                "stop",
                "--env",
                environment,
                "--yes",
                app_name,
            ),
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"attempted": True, "confirmed": False, "exit_code": None}
    return {
        "attempted": True,
        "confirmed": completed.returncode == 0,
        "exit_code": completed.returncode,
    }


def _fail(
    *,
    ledger_path: Path,
    ledger: MutableMapping[str, Any],
    record: MutableMapping[str, Any],
    message: str,
) -> NoReturn:
    record["status"] = "fail"
    record["finished_at"] = _utc_now()
    record["error"] = message
    ledger["status"] = "fail"
    ledger["finished_at"] = record["finished_at"]
    _write_ledger(ledger_path, ledger)
    raise RecertificationError(message)


def _run_job(
    *,
    modal_binary: Path,
    environment: str,
    job: Mapping[str, Any],
    plan: Mapping[str, Any],
    artifacts_directory: Path,
    rejected_directory: Path,
    ledger_path: Path,
    ledger: MutableMapping[str, Any],
) -> None:
    _assert_checkout(plan)
    record = _job_record(job)
    records = ledger.get("jobs")
    if not isinstance(records, list):
        raise RecertificationError("internal ledger job list is invalid")
    records.append(record)
    _write_ledger(ledger_path, ledger)

    prefix = job.get("artifact_prefix")
    before = _matching_artifacts(prefix) if isinstance(prefix, str) else {}
    launcher = job.get("launcher")
    app_name = job.get("app_name")
    max_wall_seconds = job.get("max_wall_seconds")
    arguments = record["arguments"]
    if (
        not isinstance(launcher, str)
        or not isinstance(app_name, str)
        or _SAFE_ENVIRONMENT.fullmatch(app_name) is None
        or not isinstance(max_wall_seconds, int)
        or isinstance(max_wall_seconds, bool)
        or max_wall_seconds <= 0
        or not isinstance(arguments, list)
    ):
        _fail(
            ledger_path=ledger_path,
            ledger=ledger,
            record=record,
            message="planned Modal job is malformed",
        )
    command = (
        str(modal_binary),
        "run",
        f"--env={environment}",
        launcher,
        *arguments,
    )
    started = time.perf_counter()
    try:
        completed = subprocess.run(  # nosec B603
            command,
            cwd=ROOT,
            check=False,
            timeout=max_wall_seconds,
        )
        record["elapsed_seconds"] = time.perf_counter() - started
        record["exit_code"] = completed.returncode
    except subprocess.TimeoutExpired:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["remote_stop"] = _stop_modal_app(
            modal_binary,
            environment=environment,
            app_name=app_name,
        )
        after = _matching_artifacts(prefix) if isinstance(prefix, str) else {}
        new_paths = [after[name] for name in sorted(set(after) - set(before))]
        if new_paths:
            record["rejected_artifacts"] = _custody_all(new_paths, rejected_directory)
        _fail(
            ledger_path=ledger_path,
            ledger=ledger,
            record=record,
            message=(
                f"Modal job exceeded its sealed {max_wall_seconds}-second wall limit; "
                "remote completion is ambiguous and was not retried"
            ),
        )
    except KeyboardInterrupt:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["remote_stop"] = _stop_modal_app(
            modal_binary,
            environment=environment,
            app_name=app_name,
        )
        _fail(
            ledger_path=ledger_path,
            ledger=ledger,
            record=record,
            message="Modal client was interrupted; remote completion is ambiguous and was not retried",
        )
    except OSError:
        record["elapsed_seconds"] = time.perf_counter() - started
        _fail(
            ledger_path=ledger_path,
            ledger=ledger,
            record=record,
            message="Modal client could not be started",
        )

    after = _matching_artifacts(prefix) if isinstance(prefix, str) else {}
    new_paths = [after[name] for name in sorted(set(after) - set(before))]
    if completed.returncode != 0:
        record["remote_stop"] = _stop_modal_app(
            modal_binary,
            environment=environment,
            app_name=app_name,
        )
        if new_paths:
            record["rejected_artifacts"] = _custody_all(new_paths, rejected_directory)
        _fail(
            ledger_path=ledger_path,
            ledger=ledger,
            record=record,
            message=f"Modal job exited with status {completed.returncode}",
        )

    if prefix is None:
        try:
            _assert_checkout(plan)
        except RecertificationError:
            _fail(
                ledger_path=ledger_path,
                ledger=ledger,
                record=record,
                message="CPU canary changed the sealed checkout",
            )
    else:
        if len(new_paths) != 1:
            if new_paths:
                record["rejected_artifacts"] = _custody_all(new_paths, rejected_directory)
            _fail(
                ledger_path=ledger_path,
                ledger=ledger,
                record=record,
                message=f"expected exactly one new artifact, observed {len(new_paths)}",
            )
        artifact_path = new_paths[0]
        try:
            validate_artifact(artifact_path, job, plan)
        except RecertificationError as exc:
            record["rejected_artifacts"] = _custody_all([artifact_path], rejected_directory)
            _fail(
                ledger_path=ledger_path,
                ledger=ledger,
                record=record,
                message=f"artifact contract failed: {exc}",
            )
        record["artifact"] = _copy_to_custody(artifact_path, artifacts_directory)
        try:
            _assert_checkout(plan)
        except RecertificationError:
            _fail(
                ledger_path=ledger_path,
                ledger=ledger,
                record=record,
                message="checkout was not clean after artifact custody",
            )

    record["status"] = "pass"
    record["finished_at"] = _utc_now()
    _write_ledger(ledger_path, ledger)


def execute(
    *,
    plan: Mapping[str, Any],
    stage: str,
    modal_binary: Path,
    environment: str,
    evidence_root: Path,
    approved_plan_sha256: str,
    approved_max_usd: str,
    canary_ledger: Path | None,
) -> Path:
    """Execute one sealed stage sequentially, stopping on the first uncertainty."""

    if _SAFE_ENVIRONMENT.fullmatch(environment) is None:
        raise RecertificationError("Modal environment name is invalid")
    approved = _validate_approval(
        plan,
        approved_plan_sha256=approved_plan_sha256,
        approved_max_usd=approved_max_usd,
    )
    _assert_checkout(plan)
    external = _external_root(evidence_root)
    campaign_root = external / _plan_hash(plan)
    if campaign_root.exists() and (campaign_root.is_symlink() or not campaign_root.is_dir()):
        raise RecertificationError("campaign evidence path must be a real directory")
    try:
        campaign_root.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        raise RecertificationError("could not create campaign evidence directory") from exc
    if campaign_root.is_symlink() or campaign_root.resolve().parent != external:
        raise RecertificationError("campaign evidence path escapes the evidence root")
    prerequisite_digest: str | None = None
    if stage == "full":
        prerequisite = canary_ledger or campaign_root / "canary" / "ledger.json"
        prerequisite_digest = _validate_canary_ledger(
            prerequisite,
            plan=plan,
            environment=environment,
            campaign_root=campaign_root,
        )
    elif stage != "canary":
        raise RecertificationError("stage must be canary or full")

    executable = _modal_binary(modal_binary)
    modal_version = _modal_version(executable)
    _require_auth(executable)

    stage_root = campaign_root / stage
    try:
        stage_root.mkdir(mode=0o700, parents=True, exist_ok=False)
        artifacts_directory = stage_root / "artifacts"
        rejected_directory = stage_root / "rejected"
        artifacts_directory.mkdir(mode=0o700)
        rejected_directory.mkdir(mode=0o700)
    except OSError as exc:
        raise RecertificationError(
            "stage evidence directory already exists or could not be created; no retry was made"
        ) from exc
    ledger_path = stage_root / "ledger.json"
    ledger: dict[str, Any] = {
        "schema": LEDGER_SCHEMA,
        "plan_sha256": _plan_hash(plan),
        "campaign": _campaign(plan),
        "source_commit": _source_commit(plan),
        "stage": stage,
        "environment": environment,
        "approved_max_usd": str(approved),
        "campaign_approval_ceiling_usd": str(_cost_contract(plan)[2]),
        "provider_hard_cap_confirmed": False,
        "modal_client": {
            "version": modal_version,
            "executable_sha256": sha256_file(executable),
        },
        "prerequisite_canary_ledger_sha256": prerequisite_digest,
        "status": "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "jobs": [],
    }
    _write_ledger(ledger_path, ledger)

    for job in jobs_for_stage(plan, stage):
        try:
            _run_job(
                modal_binary=executable,
                environment=environment,
                job=job,
                plan=plan,
                artifacts_directory=artifacts_directory,
                rejected_directory=rejected_directory,
                ledger_path=ledger_path,
                ledger=ledger,
            )
        except RecertificationError as exc:
            if ledger.get("status") != "fail":
                records = ledger.get("jobs")
                if isinstance(records, list) and records and isinstance(records[-1], dict):
                    record = records[-1]
                    if record.get("status") == "running":
                        record["status"] = "fail"
                        record["finished_at"] = _utc_now()
                        record["error"] = str(exc)
                ledger["status"] = "fail"
                ledger["finished_at"] = _utc_now()
                _write_ledger(ledger_path, ledger)
            raise
    ledger["status"] = "pass"
    ledger["finished_at"] = _utc_now()
    _write_ledger(ledger_path, ledger)
    return ledger_path


def _write_plan(plan: Mapping[str, Any], destination: Path | None) -> None:
    if destination is None:
        sys.stdout.buffer.write(canonical_json(plan))
        return
    if not destination.is_absolute():
        raise RecertificationError("plan output path must be absolute")
    if destination.exists() or destination.is_symlink():
        raise RecertificationError("plan output path already exists")
    _atomic_json(destination, plan)
    low, high, cap = _cost_contract(plan)
    print(f"Plan: {destination}")
    print(f"Plan SHA-256: {_plan_hash(plan)}")
    print(f"Expected gross cost: ${low}-${high}; approval ceiling: ${cap}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="seal a local-only campaign plan")
    plan_parser.add_argument("--campaign", choices=("compatibility", "paper-matrix"), required=True)
    plan_parser.add_argument("--output", type=Path)

    check_parser = subparsers.add_parser("check", help="verify a plan against clean HEAD")
    check_parser.add_argument("--plan", type=Path, required=True)

    execute_parser = subparsers.add_parser("execute", help="execute one explicitly approved stage")
    execute_parser.add_argument("--plan", type=Path, required=True)
    execute_parser.add_argument("--stage", choices=("canary", "full"), required=True)
    execute_parser.add_argument("--modal-bin", type=Path, required=True)
    execute_parser.add_argument("--environment", required=True)
    execute_parser.add_argument("--evidence-root", type=Path, required=True)
    execute_parser.add_argument("--approve-plan-sha256", required=True)
    execute_parser.add_argument("--approve-max-usd", required=True)
    execute_parser.add_argument("--canary-ledger", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = _parser().parse_args(arguments)
    try:
        if parsed.command == "plan":
            _write_plan(build_plan(parsed.campaign), parsed.output)
            return 0
        if parsed.command == "check":
            plan = load_plan(parsed.plan)
            low, high, cap = _cost_contract(plan)
            canary_cost = estimate_historical_gpu_cost(plan, "canary")
            full_cost = estimate_historical_gpu_cost(plan, "full")
            print(f"PASS: plan {_plan_hash(plan)} matches clean HEAD {_source_commit(plan)}")
            print(f"Expected gross campaign cost: ${low}-${high}; approval ceiling: ${cap}")
            print(
                "Historical GPU-only point estimates: "
                f"canary ${canary_cost:.2f}; full ${full_cost:.2f}"
            )
            return 0
        if parsed.command == "execute":
            plan = load_plan(parsed.plan)
            ledger = execute(
                plan=plan,
                stage=parsed.stage,
                modal_binary=parsed.modal_bin,
                environment=parsed.environment,
                evidence_root=parsed.evidence_root,
                approved_plan_sha256=parsed.approve_plan_sha256,
                approved_max_usd=parsed.approve_max_usd,
                canary_ledger=parsed.canary_ledger,
            )
            print(f"PASS: {parsed.stage} evidence ledger: {ledger}")
            return 0
        raise RecertificationError("unknown command")
    except RecertificationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
