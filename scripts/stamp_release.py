#!/usr/bin/env python3
"""Stamp an already-existing Git commit into release distributions.

The generated module is ignored by Git because a commit cannot contain its own
final object ID.  CI derives the value from ``github.sha`` after checkout,
builds the distributions, and attests their digests separately.
"""

from __future__ import annotations

import argparse
import os
import re

# Release stamping executes only fixed Git argument vectors without a shell.
import subprocess  # nosec B404
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src" / "schemen_gate" / "_build_identity.py"
EXPECTED_VERSION = "1.0.2"
_COMMIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_GITHUB_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


def validate_repository(value: str) -> str:
    parsed = urlsplit(value)
    path_parts = tuple(part for part in parsed.path.split("/") if part)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.query
        or parsed.fragment
        or len(path_parts) != 2
        or any(
            part in {".", ".."} or _GITHUB_COMPONENT.fullmatch(part) is None for part in path_parts
        )
        or path_parts[1].endswith(".git")
        or value != f"https://github.com/{path_parts[0]}/{path_parts[1]}"
    ):
        raise ValueError("repository must be canonical https://github.com/OWNER/REPO")
    return value


def render(*, version: str, repository: str, commit: str) -> str:
    if version != EXPECTED_VERSION:
        raise ValueError(f"version must be exactly {EXPECTED_VERSION}")
    validate_repository(repository)
    if _COMMIT_SHA.fullmatch(commit) is None:
        raise ValueError("commit must be a lowercase 40- or 64-character Git SHA")
    return (
        '"""Generated release identity; do not commit this file."""\n\n'
        f"SOURCE_VERSION = {version!r}\n"
        f"SOURCE_REPOSITORY = {repository!r}\n"
        f"SOURCE_COMMIT = {commit!r}\n"
    )


def write_atomically(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        raise ValueError("refusing to replace a symlinked build-identity module")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_source_commit(commit: str) -> None:
    """Require the stamp to name this exact clean checkout."""
    try:
        head = subprocess.run(  # nosec B603
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(  # nosec B603
            ("git", "diff", "--quiet"), cwd=ROOT, check=True
        )
        subprocess.run(  # nosec B603
            ("git", "diff", "--cached", "--quiet"), cwd=ROOT, check=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("release stamping requires a clean Git checkout") from exc
    if commit != head:
        raise ValueError(f"requested commit {commit!r} does not match checkout HEAD {head!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify an existing stamp instead of writing it.",
    )
    args = parser.parse_args()
    try:
        expected = render(
            version=args.version,
            repository=args.repository,
            commit=args.commit,
        )
        verify_source_commit(args.commit)
        if args.check:
            if not DESTINATION.is_file() or DESTINATION.read_text(encoding="utf-8") != expected:
                raise ValueError("build-identity stamp is absent or does not match")
        else:
            write_atomically(DESTINATION, expected)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"release stamp failed: {exc}") from exc
    action = "verified" if args.check else "wrote"
    print(f"{action} Gate {args.version} source commit {args.commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
