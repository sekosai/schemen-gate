"""Modal CPU canary for the certificate-to-Gate AI-PKI path."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from scripts.modal_source import ModalSourceExport

REPOSITORY_ROOT = next(
    (
        candidate
        for candidate in Path(__file__).resolve().parents
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "scripts" / "modal_source.py").is_file()
        and (candidate / "src" / "schemen_gate" / "__init__.py").is_file()
    ),
    Path(__file__).resolve().parent,
)
EXAMPLES = Path(__file__).resolve().parent
EXPECTED_PASSES = (
    "PASS: certificate -> signed grant -> resolved Regime -> Gate",
    "PASS: wrong trust root denied before Gate",
    "PASS: wrong recipient certificate denied before Gate",
)

app = modal.App("schemen-gate-quickstart")
_EXPECTED_VERSION_ENV = "SCHEMEN_GATE_EXPECTED_VERSION"
_EXPECTED_REPOSITORY_ENV = "SCHEMEN_GATE_EXPECTED_REPOSITORY"
_EXPECTED_COMMIT_ENV = "SCHEMEN_GATE_EXPECTED_COMMIT"


def _required_remote_identity(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"remote Gate identity environment is missing {name}")
    return value


SOURCE_EXPORT: ModalSourceExport | None
if modal.is_local():
    # The exact tracked-source export is a client-side construction. Modal
    # imports this module again in the completed image, where neither the Git
    # checkout nor the repository-only ``scripts`` helper is intentionally
    # present.
    from scripts.modal_source import prepare_modal_source_export

    SOURCE_EXPORT = prepare_modal_source_export(REPOSITORY_ROOT)
    EXPECTED_GATE_VERSION = SOURCE_EXPORT.version
    EXPECTED_SOURCE_REPOSITORY = SOURCE_EXPORT.repository
    EXPECTED_SOURCE_COMMIT = SOURCE_EXPORT.commit
else:
    SOURCE_EXPORT = None
    EXPECTED_GATE_VERSION = _required_remote_identity(_EXPECTED_VERSION_ENV)
    EXPECTED_SOURCE_REPOSITORY = _required_remote_identity(_EXPECTED_REPOSITORY_ENV)
    EXPECTED_SOURCE_COMMIT = _required_remote_identity(_EXPECTED_COMMIT_ENV)

image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "numpy==2.4.6",
    "cryptography==50.0.0",
    "pyyaml==6.0.3",
    "fastapi[standard]==0.116.1",
)
if SOURCE_EXPORT is not None:
    image = image.env(
        {
            _EXPECTED_VERSION_ENV: EXPECTED_GATE_VERSION,
            _EXPECTED_REPOSITORY_ENV: EXPECTED_SOURCE_REPOSITORY,
            _EXPECTED_COMMIT_ENV: EXPECTED_SOURCE_COMMIT,
        }
    ).add_local_file(
        EXAMPLES / "ai_pki_quickstart.py",
        "/root/examples/ai_pki_quickstart.py",
        copy=True,
    )
    for package_file in SOURCE_EXPORT.package_files:
        image = image.add_local_file(
            SOURCE_EXPORT.root / "src" / "schemen_gate" / package_file,
            f"/root/schemen_gate/{package_file}",
            copy=True,
        )


def _run_ai_pki_canary() -> dict[str, object]:
    from schemen_gate import current_release_identity

    release = current_release_identity()
    source_commit = release.require_source_commit()
    if (
        release.version != EXPECTED_GATE_VERSION
        or release.source_repository != EXPECTED_SOURCE_REPOSITORY
        or source_commit != EXPECTED_SOURCE_COMMIT
    ):
        raise RuntimeError("remote Gate release identity does not match the exact local Git HEAD")
    # Both the interpreter and image-baked example path are fixed; request data
    # cannot influence process selection or arguments.
    result = subprocess.run(  # nosec B603
        [sys.executable, "-m", "examples.ai_pki_quickstart"],
        cwd="/root",
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    lines = tuple(line for line in result.stdout.splitlines() if line)
    if result.returncode != 0 or lines[: len(EXPECTED_PASSES)] != EXPECTED_PASSES:
        raise RuntimeError(
            "AI-PKI canary failed: "
            f"returncode={result.returncode}, stdout={lines!r}, stderr={result.stderr!r}"
        )
    return {
        "status": "ok",
        "surface": "schemen-gate-ai-pki-modal-cpu-canary",
        "gate_release": release.to_dict(),
        "checks": list(lines[: len(EXPECTED_PASSES)]),
        "evidence": list(lines[len(EXPECTED_PASSES) :]),
    }


@app.function(image=image, timeout=60, max_containers=1)
def remote_gate() -> dict[str, object]:
    return _run_ai_pki_canary()


@app.function(image=image, timeout=60, max_containers=1)
@modal.fastapi_endpoint(requires_proxy_auth=True)
def health() -> dict[str, object]:
    return _run_ai_pki_canary()


@app.local_entrypoint()
def main() -> None:
    result = remote_gate.remote()
    release = result.get("gate_release")
    if (
        result.get("status") != "ok"
        or not isinstance(release, dict)
        or release.get("version") != EXPECTED_GATE_VERSION
        or release.get("source_repository") != EXPECTED_SOURCE_REPOSITORY
        or release.get("source_commit") != EXPECTED_SOURCE_COMMIT
    ):
        raise RuntimeError(f"unexpected Modal canary result: {result!r}")
    print(json.dumps(result, indent=2, sort_keys=True))
