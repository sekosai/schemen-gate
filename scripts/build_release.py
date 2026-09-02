#!/usr/bin/env python3
"""Build Gate 1.0.2 offline from an exact tracked-source export."""

from __future__ import annotations

import argparse
import os
import shutil

# Release tooling executes only fixed program-controlled argument vectors and
# never invokes a shell.
import subprocess  # nosec B404
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAMP = ROOT / "src" / "schemen_gate" / "_build_identity.py"
EXPECTED_VERSION = "1.0.2"
EXPECTED_REPOSITORY = "https://github.com/sekosai/schemen-gate"
EXPECTED_TOOLS = {
    "build": "1.6.0",
    "colorama": "0.4.6",
    "packaging": "26.3",
    "pip": "26.2.1",
    "pyproject-hooks": "1.2.0",
    "setuptools": "84.0.0",
    "wheel": "0.47.0",
}


def fail(message: str) -> None:
    raise SystemExit(f"release build failed: {message}")


def run(*command: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)  # nosec B603


def git_output(*arguments: str) -> str:
    return subprocess.run(  # nosec B603
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_release_tree() -> str:
    run("git", "diff", "--exit-code")
    run("git", "diff", "--cached", "--exit-code")
    run("git", "diff", "--check")
    run("git", "diff", "--cached", "--check")
    untracked = git_output("ls-files", "--others", "--exclude-standard", "-z")
    if untracked:
        paths = sorted(path for path in untracked.split("\0") if path)
        fail(f"untracked, non-ignored paths are forbidden: {paths!r}")
    staged = subprocess.run(  # nosec B603
        ("git", "ls-files", "--stage", "-z"),
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.decode("utf-8")
    symlinks = []
    for entry in staged.split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        if metadata.split(" ", 1)[0] == "120000":
            symlinks.append(path)
    if symlinks:
        fail(f"tracked symlinks are forbidden in the release tree: {symlinks!r}")
    return git_output("rev-parse", "HEAD")


def verify_build_python(python: Path) -> None:
    if not python.is_file():
        fail(f"locked build Python does not exist: {python}")
    expected = repr(EXPECTED_TOOLS)
    probe = (
        "import importlib.metadata as m, sys; "
        f"expected={expected}; "
        "actual={name:m.version(name) for name in expected}; "
        "sys.exit(None if actual == expected else "
        "f'locked build environment mismatch: expected {expected!r}, found {actual!r}')"
    )
    run(str(python), "-I", "-c", probe)


def export_tracked_tree(destination: Path) -> None:
    prefix = f"{destination}{os.sep}"
    run("git", "checkout-index", "--all", f"--prefix={prefix}")
    exported_stamp = destination / STAMP.relative_to(ROOT)
    exported_stamp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(STAMP, exported_stamp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-python",
        type=Path,
        default=ROOT
        / ".release-build-env"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        help="Python from scripts/bootstrap_build_env.py.",
    )
    args = parser.parse_args()

    commit = verify_release_tree()
    run(sys.executable, "scripts/release_manifest.py", "--verify")
    run(
        sys.executable,
        "scripts/stamp_release.py",
        "--version",
        EXPECTED_VERSION,
        "--repository",
        EXPECTED_REPOSITORY,
        "--commit",
        commit,
        "--check",
    )
    # Preserve the venv executable path. Resolving a POSIX venv symlink selects
    # the base interpreter and silently drops the hash-locked site-packages.
    build_python = args.build_python.absolute()
    verify_build_python(build_python)

    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    for artifact in dist.glob("schemen_gate-*"):
        if artifact.is_file():
            artifact.unlink()

    with tempfile.TemporaryDirectory(prefix="schemen-gate-export-") as raw_tmp:
        export = Path(raw_tmp) / "source"
        export.mkdir()
        export_tracked_tree(export)
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env.update(
            {
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INDEX": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        run(
            str(build_python),
            "-I",
            "-m",
            "build",
            "--no-isolation",
            "--outdir",
            str(dist),
            str(export),
            env=env,
        )

    artifacts = sorted(path.name for path in dist.glob(f"schemen_gate-{EXPECTED_VERSION}*"))
    if artifacts != [
        f"schemen_gate-{EXPECTED_VERSION}-py3-none-any.whl",
        f"schemen_gate-{EXPECTED_VERSION}.tar.gz",
    ]:
        fail(f"expected one canonical wheel and sdist, found: {artifacts!r}")
    print(f"offline tracked-export build passed for {commit}: {artifacts!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
