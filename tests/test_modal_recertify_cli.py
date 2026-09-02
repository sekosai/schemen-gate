"""Fail-closed executor tests for the Modal re-certification CLI.

Every process boundary is replaced with a local fake. These tests must never
contact Modal, execute a launcher, or depend on account credentials.
"""

from __future__ import annotations

import builtins
import json
import stat
import subprocess
from pathlib import Path
from typing import Any, Callable

import pytest

from scripts import modal_recertification as core
from scripts import modal_recertify as cli

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
PLAN_SHA256 = "c" * 64
LAUNCHER_SHA256 = "d" * 64
ENVIRONMENT = "recert-test"


def plan_with(
    *, canary_jobs: list[dict[str, Any]], full_jobs: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    return {
        "schema": "schemen-gate/modal-recertification-plan-v1",
        "campaign": "compatibility",
        "plan_sha256": PLAN_SHA256,
        "release": {
            "version": "1.0.2",
            "repository": "https://github.com/sekosai/schemen-gate",
            "source_commit": COMMIT,
        },
        "estimate": {
            "expected_gross_usd": {"low": 2.5, "high": 4.0},
            "campaign_approval_ceiling_usd": 8.0,
            "credits_included": False,
        },
        "stages": {
            "canary": canary_jobs,
            "full": full_jobs if full_jobs is not None else canary_jobs,
        },
    }


def job(*, artifact_prefix: str | None = "executor_test_") -> dict[str, Any]:
    return {
        "id": "executor-test",
        "app_name": "executor-test-app",
        "launcher": "research/cdp/experiments/modal_executor_test.py",
        "launcher_sha256": LAUNCHER_SHA256,
        "arguments": ["--smoke", "--seed", "42"],
        "artifact_prefix": artifact_prefix,
        "expected_experiment": "executor_test" if artifact_prefix is not None else None,
        "expected_artifact_records": 1 if artifact_prefix is not None else None,
        "gpu": "A100" if artifact_prefix is not None else None,
        "historical_seconds": 10,
        "max_wall_seconds": 90,
        "remote_invocations": 1,
        "variant": "canary",
    }


def executable(tmp_path: Path) -> Path:
    path = tmp_path / "modal"
    path.write_text("#!/bin/sh\nexit 97\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def read_ledger(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def assert_ledger_seal(ledger: dict[str, Any]) -> None:
    supplied = ledger["ledger_sha256"]
    base = dict(ledger)
    del base["ledger_sha256"]
    assert supplied == core.sha256_bytes(core.canonical_json(base))
    assert cli._validate_ledger_seal(ledger) == ledger


def configure_local_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Path:
    results = tmp_path / "results"
    results.mkdir()
    monkeypatch.setattr(cli, "RESULTS_DIR", results)
    monkeypatch.setattr(cli, "require_clean_head", lambda: COMMIT)
    monkeypatch.setattr(cli, "_modal_version", lambda _path: "1.5.4")
    monkeypatch.setattr(cli, "_require_auth", lambda _path: None)
    monkeypatch.setattr(
        cli,
        "validate_artifact",
        lambda path, _job, _plan: core.sha256_file(path),
    )
    return results


def test_executor_uses_exact_argv_without_a_shell_and_seals_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    results = configure_local_execution(monkeypatch, tmp_path)
    selected_job = job()
    selected_plan = plan_with(canary_jobs=[selected_job])
    modal_binary = executable(tmp_path)
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        (results / "executor_test_one.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    evidence = tmp_path / "evidence"
    ledger_path = cli.execute(
        plan=selected_plan,
        stage="canary",
        modal_binary=modal_binary,
        environment=ENVIRONMENT,
        evidence_root=evidence,
        approved_plan_sha256=PLAN_SHA256,
        approved_max_usd="8",
        canary_ledger=None,
    )

    assert calls == [
        (
            (
                str(modal_binary.resolve()),
                "run",
                f"--env={ENVIRONMENT}",
                selected_job["launcher"],
                "--smoke",
                "--seed",
                "42",
            ),
            {"cwd": core.ROOT, "check": False, "timeout": 90},
        )
    ]
    assert "shell" not in calls[0][1]
    assert not (results / "executor_test_one.json").exists()
    artifact = ledger_path.parent / "artifacts" / "executor_test_one.json"
    assert artifact.read_bytes() == b"{}\n"

    ledger = read_ledger(ledger_path)
    assert ledger["status"] == "pass"
    assert ledger["provider_hard_cap_confirmed"] is False
    assert ledger["modal_client"] == {
        "version": "1.5.4",
        "executable_sha256": core.sha256_file(modal_binary),
    }
    assert ledger["jobs"][0]["artifact"]["sha256"] == core.sha256_file(artifact)
    assert ledger["jobs"][0]["max_wall_seconds"] == 90
    assert ledger["jobs"][0]["remote_stop"] is None
    assert_ledger_seal(ledger)

    campaign = evidence / PLAN_SHA256
    for directory in (
        evidence,
        campaign,
        ledger_path.parent,
        ledger_path.parent / "artifacts",
        ledger_path.parent / "rejected",
    ):
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(ledger_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o600


def test_modal_client_version_must_be_exactly_1_5_4(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    modal_binary = executable(tmp_path)
    observed: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def exact(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="modal client version: 1.5.4\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", exact)
    assert cli._modal_version(modal_binary) == "1.5.4"
    assert observed == [
        (
            (str(modal_binary), "--version"),
            {
                "cwd": core.ROOT,
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": 10,
            },
        )
    ]
    assert "shell" not in observed[0][1]

    def wrong(command: tuple[str, ...], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="modal client version: 1.5.5\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", wrong)
    with pytest.raises(core.RecertificationError, match=r"expected .*1\.5\.4"):
        cli._modal_version(modal_binary)


@pytest.mark.parametrize("dirty", [False, True])
def test_executor_rejects_dirty_or_different_head(
    monkeypatch: pytest.MonkeyPatch,
    dirty: bool,
) -> None:
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    if dirty:

        def reject_dirty() -> str:
            raise core.RecertificationError("re-certification requires a completely clean checkout")

        monkeypatch.setattr(cli, "require_clean_head", reject_dirty)
        expected = "completely clean"
    else:
        monkeypatch.setattr(cli, "require_clean_head", lambda: OTHER_COMMIT)
        expected = "HEAD changed"

    with pytest.raises(core.RecertificationError, match=expected):
        cli._assert_checkout(selected_plan)


def test_approval_is_bound_to_exact_plan_hash_and_ceiling() -> None:
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    with pytest.raises(core.RecertificationError, match="exact plan SHA-256"):
        cli._validate_approval(
            selected_plan,
            approved_plan_sha256="f" * 64,
            approved_max_usd="8",
        )
    with pytest.raises(core.RecertificationError, match="below estimated high"):
        cli._validate_approval(
            selected_plan,
            approved_plan_sha256=PLAN_SHA256,
            approved_max_usd="3.99",
        )
    with pytest.raises(core.RecertificationError, match="exceeds the plan ceiling"):
        cli._validate_approval(
            selected_plan,
            approved_plan_sha256=PLAN_SHA256,
            approved_max_usd="8.01",
        )


def test_existing_stage_is_never_overwritten(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_local_execution(monkeypatch, tmp_path)
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    evidence = tmp_path / "evidence"
    existing = evidence / PLAN_SHA256 / "canary"
    existing.mkdir(parents=True)
    sentinel = existing / "keep.txt"
    sentinel.write_text("do not replace\n", encoding="utf-8")

    with pytest.raises(core.RecertificationError, match="already exists"):
        cli.execute(
            plan=selected_plan,
            stage="canary",
            modal_binary=executable(tmp_path),
            environment=ENVIRONMENT,
            evidence_root=evidence,
            approved_plan_sha256=PLAN_SHA256,
            approved_max_usd="8",
            canary_ledger=None,
        )
    assert sentinel.read_text(encoding="utf-8") == "do not replace\n"


def test_campaign_symlink_cannot_escape_the_evidence_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_local_execution(monkeypatch, tmp_path)
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (evidence / PLAN_SHA256).symlink_to(outside, target_is_directory=True)

    with pytest.raises(core.RecertificationError, match="campaign evidence path"):
        cli.execute(
            plan=selected_plan,
            stage="canary",
            modal_binary=executable(tmp_path),
            environment=ENVIRONMENT,
            evidence_root=evidence,
            approved_plan_sha256=PLAN_SHA256,
            approved_max_usd="8",
            canary_ledger=None,
        )
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("artifact_count", [0, 1, 2])
def test_job_requires_exactly_one_new_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    artifact_count: int,
) -> None:
    results = configure_local_execution(monkeypatch, tmp_path)
    selected_job = job()
    selected_plan = plan_with(canary_jobs=[selected_job])
    stage = tmp_path / "stage"
    artifacts = stage / "artifacts"
    rejected = stage / "rejected"
    artifacts.mkdir(parents=True)
    rejected.mkdir()
    ledger_path = stage / "ledger.json"
    ledger: dict[str, Any] = {"schema": cli.LEDGER_SCHEMA, "status": "running", "jobs": []}

    def fake_run(command: tuple[str, ...], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        for index in range(artifact_count):
            (results / f"executor_test_{index}.json").write_text("{}\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    if artifact_count == 1:
        cli._run_job(
            modal_binary=tmp_path / "modal",
            environment=ENVIRONMENT,
            job=selected_job,
            plan=selected_plan,
            artifacts_directory=artifacts,
            rejected_directory=rejected,
            ledger_path=ledger_path,
            ledger=ledger,
        )
        assert [path.name for path in artifacts.iterdir()] == ["executor_test_0.json"]
        assert ledger["jobs"][0]["status"] == "pass"
    else:
        with pytest.raises(
            core.RecertificationError,
            match=f"expected exactly one new artifact, observed {artifact_count}",
        ):
            cli._run_job(
                modal_binary=tmp_path / "modal",
                environment=ENVIRONMENT,
                job=selected_job,
                plan=selected_plan,
                artifacts_directory=artifacts,
                rejected_directory=rejected,
                ledger_path=ledger_path,
                ledger=ledger,
            )
        assert ledger["jobs"][0]["status"] == "fail"
        assert sorted(path.name for path in rejected.iterdir()) == [
            f"executor_test_{index}.json" for index in range(artifact_count)
        ]
    assert_ledger_seal(read_ledger(ledger_path))


@pytest.mark.parametrize("outcome", ["nonzero", "interrupt"])
def test_nonzero_and_interrupted_jobs_are_failed_without_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    outcome: str,
) -> None:
    configure_local_execution(monkeypatch, tmp_path)
    selected_job = job()
    selected_plan = plan_with(canary_jobs=[selected_job])
    stage = tmp_path / "stage"
    artifacts = stage / "artifacts"
    rejected = stage / "rejected"
    artifacts.mkdir(parents=True)
    rejected.mkdir()
    ledger_path = stage / "ledger.json"
    ledger: dict[str, Any] = {"schema": cli.LEDGER_SCHEMA, "status": "running", "jobs": []}
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[1:3] == ("app", "stop"):
            return subprocess.CompletedProcess(command, 0)
        if outcome == "interrupt":
            raise KeyboardInterrupt
        return subprocess.CompletedProcess(command, 19)

    monkeypatch.setattr(subprocess, "run", fake_run)
    expected = "ambiguous and was not retried" if outcome == "interrupt" else "status 19"
    with pytest.raises(core.RecertificationError, match=expected):
        cli._run_job(
            modal_binary=tmp_path / "modal",
            environment=ENVIRONMENT,
            job=selected_job,
            plan=selected_plan,
            artifacts_directory=artifacts,
            rejected_directory=rejected,
            ledger_path=ledger_path,
            ledger=ledger,
        )
    modal_path = str(tmp_path / "modal")
    assert calls == [
        (
            (
                modal_path,
                "run",
                f"--env={ENVIRONMENT}",
                selected_job["launcher"],
                "--smoke",
                "--seed",
                "42",
            ),
            {"cwd": core.ROOT, "check": False, "timeout": 90},
        ),
        (
            (
                modal_path,
                "app",
                "stop",
                "--env",
                ENVIRONMENT,
                "--yes",
                "executor-test-app",
            ),
            {
                "cwd": core.ROOT,
                "check": False,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "timeout": 60,
            },
        ),
    ]
    assert all("shell" not in kwargs for _command, kwargs in calls)
    persisted = read_ledger(ledger_path)
    assert persisted["status"] == "fail"
    assert persisted["jobs"][0]["status"] == "fail"
    assert persisted["jobs"][0]["remote_stop"] == {
        "attempted": True,
        "confirmed": True,
        "exit_code": 0,
    }
    assert_ledger_seal(persisted)


def test_timeout_attempts_exact_remote_stop_and_seals_ambiguity_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configure_local_execution(monkeypatch, tmp_path)
    selected_job = job()
    selected_plan = plan_with(canary_jobs=[selected_job])
    stage = tmp_path / "stage"
    artifacts = stage / "artifacts"
    rejected = stage / "rejected"
    artifacts.mkdir(parents=True)
    rejected.mkdir()
    ledger_path = stage / "ledger.json"
    ledger: dict[str, Any] = {"schema": cli.LEDGER_SCHEMA, "status": "running", "jobs": []}
    calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def fake_run(command: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        if command[1:3] == ("app", "stop"):
            return subprocess.CompletedProcess(command, 0)
        raise subprocess.TimeoutExpired(command, timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(
        core.RecertificationError,
        match=r"sealed 90-second wall limit.*ambiguous and was not retried",
    ):
        cli._run_job(
            modal_binary=tmp_path / "modal",
            environment=ENVIRONMENT,
            job=selected_job,
            plan=selected_plan,
            artifacts_directory=artifacts,
            rejected_directory=rejected,
            ledger_path=ledger_path,
            ledger=ledger,
        )

    assert calls[0][0][1] == "run"
    assert calls[0][1]["timeout"] == 90
    assert calls[1][0] == (
        str(tmp_path / "modal"),
        "app",
        "stop",
        "--env",
        ENVIRONMENT,
        "--yes",
        "executor-test-app",
    )
    assert calls[1][1]["timeout"] == 60
    persisted = read_ledger(ledger_path)
    record = persisted["jobs"][0]
    assert record["status"] == "fail"
    assert record["exit_code"] is None
    assert record["max_wall_seconds"] == 90
    assert record["remote_stop"] == {
        "attempted": True,
        "confirmed": True,
        "exit_code": 0,
    }
    assert "ambiguous and was not retried" in record["error"]
    assert_ledger_seal(persisted)


def test_artifact_custody_hashes_then_removes_source(tmp_path: Path) -> None:
    source = tmp_path / "source" / "artifact.json"
    source.parent.mkdir()
    source.write_bytes(b'{"evidence":true}\n')
    expected = core.sha256_file(source)
    custody = tmp_path / "stage" / "artifacts"

    receipt = cli._copy_to_custody(source, custody)

    destination = custody / "artifact.json"
    assert not source.exists()
    assert destination.read_bytes() == b'{"evidence":true}\n'
    assert receipt == {
        "relative_path": "artifacts/artifact.json",
        "sha256": expected,
        "size_bytes": len(b'{"evidence":true}\n'),
    }
    assert stat.S_IMODE(custody.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_plan_output_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    destination = tmp_path / "plan.json"
    destination.write_text("operator-owned\n", encoding="utf-8")

    with pytest.raises(core.RecertificationError, match="already exists"):
        cli._write_plan(selected_plan, destination)

    assert destination.read_text(encoding="utf-8") == "operator-owned\n"


def canary_ledger(
    tmp_path: Path,
    *,
    selected_plan: dict[str, Any],
    environment: str = ENVIRONMENT,
) -> tuple[Path, Path]:
    campaign_root = tmp_path / "evidence" / PLAN_SHA256
    ledger_path = campaign_root / "canary" / "ledger.json"
    ledger_path.parent.mkdir(parents=True)
    value = {
        "schema": cli.LEDGER_SCHEMA,
        "plan_sha256": PLAN_SHA256,
        "campaign": "compatibility",
        "source_commit": COMMIT,
        "stage": "canary",
        "environment": environment,
        "status": "pass",
        "jobs": [
            {
                "id": selected_plan["stages"]["canary"][0]["id"],
                "status": "pass",
                "artifact": None,
            }
        ],
    }
    ledger_path.write_bytes(core.canonical_json(cli._sealed_ledger(value)))
    return ledger_path, campaign_root


def test_canary_ledger_rejects_tampering_and_wrong_environment(tmp_path: Path) -> None:
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    ledger_path, campaign_root = canary_ledger(tmp_path, selected_plan=selected_plan)
    assert cli._validate_canary_ledger(
        ledger_path,
        plan=selected_plan,
        environment=ENVIRONMENT,
        campaign_root=campaign_root,
    ) == core.sha256_file(ledger_path)

    tampered = read_ledger(ledger_path)
    tampered["status"] = "fail"
    ledger_path.write_bytes(core.canonical_json(tampered))
    with pytest.raises(core.RecertificationError, match="SHA-256 does not match"):
        cli._validate_canary_ledger(
            ledger_path,
            plan=selected_plan,
            environment=ENVIRONMENT,
            campaign_root=campaign_root,
        )

    wrong_path = tmp_path / "wrong" / PLAN_SHA256 / "canary" / "ledger.json"
    wrong_path.parent.mkdir(parents=True)
    wrong_value = dict(tampered)
    wrong_value["status"] = "pass"
    wrong_value["environment"] = "other-environment"
    wrong_path.write_bytes(core.canonical_json(cli._sealed_ledger(wrong_value)))
    with pytest.raises(core.RecertificationError, match="does not satisfy"):
        cli._validate_canary_ledger(
            wrong_path,
            plan=selected_plan,
            environment=ENVIRONMENT,
            campaign_root=wrong_path.parents[1],
        )


def test_plan_and_check_paths_never_import_or_call_modal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    selected_plan = plan_with(canary_jobs=[job(artifact_prefix=None)])
    real_import: Callable[..., Any] = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "modal" or name.startswith("modal."):
            raise AssertionError("planning/checking attempted to import Modal")
        return real_import(name, *args, **kwargs)

    def no_process(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("planning/checking attempted to start a process")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(cli, "build_plan", lambda _campaign: selected_plan)
    monkeypatch.setattr(cli, "load_plan", lambda _path: selected_plan)
    monkeypatch.setattr(cli, "estimate_historical_gpu_cost", lambda _plan, _stage: 0.0)

    destination = tmp_path / "plan.json"
    assert cli.main(["plan", "--campaign", "compatibility", "--output", str(destination)]) == 0
    assert destination.is_file()
    assert cli.main(["check", "--plan", str(destination)]) == 0
