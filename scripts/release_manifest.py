#!/usr/bin/env python3
"""Write or verify the SHA-256 manifest for the tracked release tree."""

from __future__ import annotations

import argparse
import hashlib

# Manifest generation executes one fixed Git argument vector without a shell.
import subprocess  # nosec B404
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "RELEASE_MANIFEST.sha256"


def tracked_paths() -> list[Path]:
    try:
        output = subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(ROOT), "ls-files", "-z"],
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "release-manifest verification requires a Git checkout; "
            "package installation and examples do not"
        ) from exc
    paths = []
    for raw_path in output.decode("utf-8").split("\0"):
        if not raw_path or raw_path == MANIFEST.name:
            continue
        if "\n" in raw_path or "\r" in raw_path:
            raise ValueError(f"release paths may not contain newlines: {raw_path!r}")
        paths.append(Path(raw_path))
    return sorted(paths, key=lambda path: path.as_posix())


def render_manifest() -> str:
    lines = []
    for relative in tracked_paths():
        digest = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    try:
        expected = render_manifest()
    except RuntimeError as exc:
        print(str(exc))
        return 2
    if args.write:
        MANIFEST.write_text(expected, encoding="utf-8")
        print(f"wrote {MANIFEST.name} for {len(tracked_paths())} tracked files")
        return 0

    if not MANIFEST.is_file():
        print(f"missing {MANIFEST.name}")
        return 1
    if MANIFEST.read_text(encoding="utf-8") != expected:
        print(f"{MANIFEST.name} does not match the tracked release tree")
        return 1
    print(f"verified {MANIFEST.name} for {len(tracked_paths())} tracked files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
