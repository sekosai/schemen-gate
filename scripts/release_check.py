#!/usr/bin/env python3
"""Run the complete local Schemen Gate 1.0.2 release-candidate verification."""

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
EXPECTED_VERSION = "1.0.2"
SOURCE_REPOSITORY = "https://github.com/sekosai/schemen-gate"


def run(*command: str, cwd: Path = ROOT, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)  # nosec B603


def source_test_environment() -> dict[str, str]:
    """Return an environment that imports only this checkout's source tree."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env.pop("PYTHONHOME", None)
    return env


def clean_install_environment() -> dict[str, str]:
    """Remove source-tree overrides before validating an installed wheel."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def git_output(*arguments: str) -> str:
    return subprocess.run(  # nosec B603
        ("git", *arguments),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_clean_install(expected_commit: str) -> None:
    wheel = ROOT / "dist" / f"schemen_gate-{EXPECTED_VERSION}-py3-none-any.whl"
    if not wheel.is_file():
        raise SystemExit(f"missing expected wheel: {wheel}")

    with tempfile.TemporaryDirectory(prefix="schemen-gate-release-") as raw_tmp:
        tmp = Path(raw_tmp)
        venv = tmp / "venv"
        env = clean_install_environment()
        run(sys.executable, "-m", "venv", str(venv), env=env)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run(
            str(python),
            "-m",
            "pip",
            "install",
            f"{wheel}[lockbox]",
            env=env,
        )
        run(
            str(python),
            "-c",
            (
                "import schemen_gate, sys; "
                f"expected={EXPECTED_VERSION!r}; actual=schemen_gate.__version__; "
                f"expected_commit={expected_commit!r}; "
                "release=schemen_gate.current_release_identity(); "
                "ok=(actual == expected and release.version == expected "
                "and release.source_commit == expected_commit); "
                "sys.exit(None if ok else "
                "f'expected Gate {expected} from {expected_commit}, "
                "found {actual} from {release.source_commit}')"
            ),
            env=env,
        )
        run(
            str(python),
            str(ROOT / "examples" / "quickstart.py"),
            cwd=tmp,
            env=env,
        )
        run(
            str(python),
            str(ROOT / "examples" / "pkcs12_identity.py"),
            cwd=tmp,
            env=env,
        )
        run(
            str(python),
            str(ROOT / "examples" / "ai_pki_quickstart.py"),
            cwd=tmp,
            env=env,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-lean",
        action="store_true",
        help="Skip the Lean build for a partial local check; the result is not release evidence.",
    )
    parser.add_argument(
        "--require-history-free",
        action="store_true",
        help="Reject reachable or recoverable Git history beyond the release root.",
    )
    parser.add_argument(
        "--build-python",
        type=Path,
        default=ROOT
        / ".release-build-env"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python"),
        help="Python from scripts/bootstrap_build_env.py.",
    )
    args = parser.parse_args()

    run("git", "diff", "--exit-code")
    run("git", "diff", "--cached", "--exit-code")
    run("git", "diff", "--check")
    run("git", "diff", "--cached", "--check")
    if args.require_history_free:
        run(sys.executable, "scripts/verify_history.py", str(ROOT))
    run(sys.executable, "scripts/public_hygiene.py")
    run(sys.executable, "scripts/check_docs.py")
    run(sys.executable, "scripts/release_manifest.py", "--verify")
    source_commit = git_output("rev-parse", "HEAD")
    run(
        sys.executable,
        "scripts/stamp_release.py",
        "--version",
        EXPECTED_VERSION,
        "--repository",
        SOURCE_REPOSITORY,
        "--commit",
        source_commit,
    )
    run(
        sys.executable,
        "-m",
        "ruff",
        "check",
        "src",
        "tests",
        "examples",
        "scripts",
        "research/cdp/experiments",
        "research/cdp/scripts",
        "research/cdp/examples",
    )
    run(sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts", "examples")
    run(
        sys.executable,
        "-m",
        "ruff",
        "check",
        "--select",
        "C901",
        "--config",
        "lint.mccabe.max-complexity=20",
        "src/schemen_gate",
    )
    run(sys.executable, "-m", "mypy", "src/schemen_gate")
    run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        env=source_test_environment(),
    )
    run(
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "research/cdp/experiments/tests",
        env=source_test_environment(),
    )
    run(sys.executable, "research/cdp/scripts/release_check.py")

    lake = None if args.skip_lean else shutil.which("lake")
    if lake:
        run(lake, "build", cwd=ROOT / "research" / "cdp")
        run("bash", "scripts/check_lean_claims.sh", cwd=ROOT / "research" / "cdp")
    elif args.skip_lean:
        print("Lean build skipped by --skip-lean: partial check, not release evidence.", flush=True)
    else:
        raise SystemExit(
            "Lake is required; install elan/Lake or pass --skip-lean for a partial check"
        )

    run(
        sys.executable,
        "scripts/build_release.py",
        "--build-python",
        str(args.build_python),
    )
    dist = ROOT / "dist"
    artifacts = sorted(str(path.relative_to(ROOT)) for path in dist.glob("schemen_gate-1.0.2*"))
    if len(artifacts) != 2:
        raise SystemExit(f"expected wheel and sdist, found: {artifacts}")
    run(sys.executable, "scripts/verify_dist.py")
    run(sys.executable, "-m", "twine", "check", *artifacts)
    verify_clean_install(source_commit)
    print("Schemen Gate 1.0.2 release candidate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
