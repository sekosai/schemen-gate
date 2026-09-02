#!/usr/bin/env python3
"""Verify that a release repository contains no recoverable prior history."""

from __future__ import annotations

import argparse

# History verification executes only fixed Git argument vectors without a shell.
import subprocess  # nosec B404
from pathlib import Path


class HistoryVerificationError(RuntimeError):
    """Raised when a repository violates the history-free release contract."""


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=repository,
        check=check,
        capture_output=True,
        text=True,
    )


def verify_history_free(repository: Path) -> str:
    """Return HEAD when ``repository`` has one root and no recoverable prior objects."""

    repository = repository.resolve()
    if not (repository / ".git").is_dir():
        raise HistoryVerificationError(f"not a Git repository: {repository}")

    try:
        head = _git(repository, "rev-parse", "HEAD").stdout.strip()
        head_ref = _git(repository, "symbolic-ref", "-q", "HEAD").stdout.strip()
        refs = _git(
            repository,
            "for-each-ref",
            "--format=%(refname) %(objectname) %(objecttype)",
            "refs",
        ).stdout.splitlines()
        commit_count = _git(repository, "rev-list", "--count", "--all").stdout.strip()
        root_count = _git(
            repository,
            "rev-list",
            "--max-parents=0",
            "--count",
            "--all",
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HistoryVerificationError("unable to resolve the release commit") from exc

    expected_ref = f"{head_ref} {head} commit"
    if not head_ref.startswith("refs/heads/") or refs != [expected_ref]:
        raise HistoryVerificationError(
            "expected only the current branch reference; found: "
            + (", ".join(refs) if refs else "none")
        )
    if commit_count != "1" or root_count != "1":
        raise HistoryVerificationError(
            f"expected exactly one all-ref reachable root commit, found commits={commit_count}, "
            f"roots={root_count}"
        )
    if _git(repository, "rev-parse", "HEAD^", check=False).returncode == 0:
        raise HistoryVerificationError("the release root commit has a parent")

    fsck = _git(
        repository,
        "fsck",
        "--full",
        "--strict",
        "--no-reflogs",
        "--unreachable",
        check=False,
    )
    if fsck.returncode != 0:
        raise HistoryVerificationError(
            "Git object verification failed: " + (fsck.stdout + fsck.stderr).strip()
        )
    recoverable = [
        line
        for line in (fsck.stdout + fsck.stderr).splitlines()
        if line.startswith(("unreachable ", "dangling "))
    ]
    if recoverable:
        raise HistoryVerificationError(
            "recoverable prior Git objects found: " + "; ".join(recoverable)
        )

    reflog_hashes = {
        line
        for line in _git(repository, "reflog", "show", "--all", "--format=%H").stdout.splitlines()
        if line
    }
    if reflog_hashes != {head}:
        raise HistoryVerificationError(
            "reflogs retain commits other than the release root: "
            + ", ".join(sorted(reflog_hashes - {head}))
        )
    return head


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        head = verify_history_free(args.repository)
    except HistoryVerificationError as exc:
        print(f"history-free verification failed: {exc}")
        return 1
    print(f"history-free repository verified at {head}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
