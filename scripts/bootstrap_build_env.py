#!/usr/bin/env python3
"""Bootstrap the hash-locked release builder, then verify it offline."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os

# Release tooling executes fixed argument vectors and never invokes a shell.
import subprocess  # nosec B404
import sys
import urllib.request
import venv
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "requirements" / "build.lock"
PYPI_INDEX = "https://pypi.org/simple"
BOOTSTRAP_PIP_FILENAME = "pip-26.2.1-py3-none-any.whl"
BOOTSTRAP_PIP_URL = (
    "https://files.pythonhosted.org/packages/f3/6e/"
    "1736e5b4ae2b778ef2f81c47d797de9f891d4d8acb047a24ca37a60294dd/"
    f"{BOOTSTRAP_PIP_FILENAME}"
)
BOOTSTRAP_PIP_SHA256 = "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"
BOOTSTRAP_PIP_SIZE = 1_816_632
EXPECTED = {
    "build": "1.6.0",
    "colorama": "0.4.6",
    "packaging": "26.3",
    "pip": "26.2.1",
    "pyproject-hooks": "1.2.0",
    "setuptools": "84.0.0",
    "wheel": "0.47.0",
}
EXPECTED_WHEELS = {
    "build-1.6.0-py3-none-any.whl",
    "colorama-0.4.6-py2.py3-none-any.whl",
    "packaging-26.3-py3-none-any.whl",
    BOOTSTRAP_PIP_FILENAME,
    "pyproject_hooks-1.2.0-py3-none-any.whl",
    "setuptools-84.0.0-py3-none-any.whl",
    "wheel-0.47.0-py3-none-any.whl",
}


def fail(message: str) -> None:
    raise SystemExit(f"build bootstrap failed: {message}")


def python_in(venv_path: Path) -> Path:
    return venv_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_new_directory(path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        fail(f"{label} already exists; remove the disposable path explicitly: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def run_exact(*command: str, env: dict[str, str] | None = None) -> None:
    """Run an argument vector without shell parsing."""
    subprocess.run(  # nosec B603
        command,
        cwd=ROOT,
        env=env,
        check=True,
        shell=False,
    )


def download_exact_wheel(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    expected_size: int,
) -> None:
    """Fetch inert bytes to a fixed path and reject any byte-level drift."""
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "files.pythonhosted.org"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        fail("bootstrap wheel URL must be credential-free HTTPS on files.pythonhosted.org")
    digest = hashlib.sha256()
    size = 0
    try:
        # The value has just been constrained to one credential-free HTTPS
        # origin and redirects are rejected before any returned bytes are used.
        with urllib.request.urlopen(  # nosec B310  # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
            url, timeout=60
        ) as response:
            if response.geturl() != url:
                fail(f"bootstrap wheel redirected to an unexpected URL: {response.geturl()}")
            if response.status != 200:
                fail(f"bootstrap wheel returned HTTP status {response.status}")
            declared_size = response.headers.get("Content-Length")
            if declared_size is not None and declared_size != str(expected_size):
                fail(
                    "bootstrap wheel Content-Length mismatch: "
                    f"expected {expected_size}, found {declared_size}"
                )
            with destination.open("xb") as handle:
                while chunk := response.read(64 * 1024):
                    size += len(chunk)
                    if size > expected_size:
                        fail("bootstrap wheel exceeded its exact expected size")
                    digest.update(chunk)
                    handle.write(chunk)
    except (Exception, SystemExit):
        destination.unlink(missing_ok=True)
        raise
    if size != expected_size:
        destination.unlink(missing_ok=True)
        fail(f"bootstrap wheel size mismatch: expected {expected_size}, found {size}")
    actual_sha256 = digest.hexdigest()
    if not hmac.compare_digest(actual_sha256, expected_sha256):
        destination.unlink(missing_ok=True)
        fail(f"bootstrap wheel SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}")


def verify_wheelhouse(wheelhouse: Path) -> None:
    members = {path.name for path in wheelhouse.iterdir() if path.is_file()}
    if members != EXPECTED_WHEELS:
        fail(
            "wheelhouse member allowlist mismatch: "
            f"expected {sorted(EXPECTED_WHEELS)!r}, found {sorted(members)!r}"
        )
    unsafe = [path.name for path in wheelhouse.iterdir() if path.is_symlink() or not path.is_file()]
    if unsafe:
        fail(f"wheelhouse contains non-regular members: {sorted(unsafe)!r}")


def verify_python(python: Path) -> None:
    expected = repr(EXPECTED)
    probe = (
        "import importlib.metadata as m, json, sys; "
        f"expected={expected}; "
        "actual={name:m.version(name) for name in expected}; "
        "print(json.dumps(actual, sort_keys=True)); "
        "sys.exit(None if actual == expected else "
        "f'locked build environment mismatch: expected {expected!r}, found {actual!r}')"
    )
    run_exact(
        str(python),
        "-I",
        "-c",
        probe,
        env={
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )


def verify_bootstrap_pip(python: Path) -> None:
    probe = (
        "import importlib.metadata as m, sys; "
        f"expected={EXPECTED['pip']!r}; "
        "actual=m.version('pip'); "
        "sys.exit(None if actual == expected else "
        "f'bootstrap pip mismatch: expected {expected!r}, found {actual!r}')"
    )
    run_exact(
        str(python),
        "-I",
        "-c",
        probe,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        type=Path,
        default=ROOT / ".release-build-env",
        help="New disposable virtual environment (must not already exist).",
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        default=ROOT / ".release-wheelhouse",
        help="New disposable wheel download directory (must not already exist).",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 11):
        fail("the release builder requires Python 3.11 or newer")
    venv_path = args.venv.resolve()
    wheelhouse = args.wheelhouse.resolve()
    ensure_new_directory(venv_path, label="virtual environment")
    ensure_new_directory(wheelhouse, label="wheelhouse")
    wheelhouse.mkdir()

    # Establish the exact pip bytes before any package installer contacts an
    # index. The host's bundled pip sees only this already verified local file.
    bootstrap_pip = wheelhouse / BOOTSTRAP_PIP_FILENAME
    download_exact_wheel(
        url=BOOTSTRAP_PIP_URL,
        destination=bootstrap_pip,
        expected_sha256=BOOTSTRAP_PIP_SHA256,
        expected_size=BOOTSTRAP_PIP_SIZE,
    )
    # POSIX standalone Python distributions (including uv-managed CPython) may
    # locate libpython relative to the original executable. Copying that
    # executable into the venv breaks its loader path; the standard POSIX
    # symlink preserves it. Windows uses its native copied-launcher behavior.
    venv.EnvBuilder(
        with_pip=True,
        clear=False,
        symlinks=os.name != "nt",
    ).create(venv_path)
    python = python_in(venv_path)
    run_exact(
        str(python),
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-deps",
        "--force-reinstall",
        str(bootstrap_pip),
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    verify_bootstrap_pip(python)

    # This is the only networked package-index phase. It uses the exact pip
    # above and may download only the explicitly enumerated, hash-locked wheels.
    run_exact(
        str(python),
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "download",
        "--require-hashes",
        "--only-binary=:all:",
        "--no-deps",
        "--index-url",
        PYPI_INDEX,
        "--dest",
        str(wheelhouse),
        "--requirement",
        str(LOCK),
    )
    verify_wheelhouse(wheelhouse)

    run_exact(
        str(python),
        "-m",
        "pip",
        "--isolated",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-deps",
        "--only-binary=:all:",
        "--require-hashes",
        "--find-links",
        str(wheelhouse),
        "--requirement",
        str(LOCK),
        env={
            **os.environ,
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INDEX": "1",
            "PYTHONNOUSERSITE": "1",
        },
    )
    verify_python(python)
    print(f"hash-locked build environment ready: {python}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
