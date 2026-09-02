"""Permanent regressions for the release boundary."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

import schemen_gate
from schemen_gate._rag import CachePolicy, GatedRAGAdapter

ROOT = Path(__file__).resolve().parents[1]


def _git(repository: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )


def test_lossless_folding_is_not_exposed_as_a_gate() -> None:
    """Lossless row encoding must never carry a gate/confinement contract."""

    assert "regime0_mask" not in inspect.signature(schemen_gate.fold_vector).parameters
    assert not hasattr(schemen_gate, "fold_and_gate")

    module_doc = schemen_gate.fold_vector.__module__
    assert module_doc == "schemen_gate._regime0_fold"
    assert "not a security boundary" in (schemen_gate.fold_vector.__doc__ or "").lower()


def test_security_evidence_is_executable_and_history_check_is_shipped() -> None:
    matrix = (ROOT / "docs" / "CLAIM_TEST_MATRIX.md").read_text(encoding="utf-8")
    assert "test_release_boundary_security.py" in matrix
    checker = (ROOT / "scripts" / "release_check.py").read_text(encoding="utf-8")
    assert "--require-history-free" in checker


def test_rag_adapter_has_no_unsafe_training_surface() -> None:
    """An arbitrary model/optimizer pair cannot be made support-safe by this adapter."""

    assert "absorb" not in GatedRAGAdapter.__dict__


@pytest.mark.parametrize("policy", [CachePolicy.WRITE_THROUGH, CachePolicy.LAZY])
def test_rag_adapter_rejects_reserved_training_cache_policies(policy: CachePolicy) -> None:
    with pytest.raises(ValueError, match=r"CachePolicy\.NONE"):
        GatedRAGAdapter(object(), object(), cache_policy=policy)


def _new_repository(path: Path) -> None:
    path.mkdir()
    _git(path, "init", "-q", "-b", "main")
    _git(path, "config", "user.name", "Release Test")
    _git(path, "config", "user.email", "release-test@example.invalid")
    _git(path, "config", "commit.gpgsign", "false")
    (path / "payload.txt").write_text("first\n", encoding="utf-8")
    _git(path, "add", "payload.txt")
    _git(path, "commit", "-q", "-m", "root")


def _verify_history(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, ROOT / "scripts" / "verify_history.py", path],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_history_verifier_accepts_a_clean_single_root(tmp_path: Path) -> None:
    repository = tmp_path / "clean"
    _new_repository(repository)

    result = _verify_history(repository)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "history-free repository verified" in result.stdout


def test_history_verifier_rejects_recoverable_pre_amend_commit(tmp_path: Path) -> None:
    repository = tmp_path / "amended"
    _new_repository(repository)
    (repository / "payload.txt").write_text("second\n", encoding="utf-8")
    _git(repository, "add", "payload.txt")
    _git(repository, "commit", "-q", "--amend", "-m", "root")

    result = _verify_history(repository)

    assert result.returncode == 1
    assert "recoverable prior Git objects found" in result.stdout


@pytest.mark.parametrize("namespace", ["heads", "tags"])
def test_history_verifier_rejects_prior_commit_retained_by_ref(
    tmp_path: Path, namespace: str
) -> None:
    repository = tmp_path / f"retained-by-{namespace}"
    _new_repository(repository)
    retained = _git(repository, "commit-tree", "HEAD^{tree}", "-m", "retained root").stdout.strip()
    _git(repository, "update-ref", f"refs/{namespace}/retained", retained)

    result = _verify_history(repository)

    assert result.returncode == 1
    assert "expected only the current branch reference" in result.stdout
