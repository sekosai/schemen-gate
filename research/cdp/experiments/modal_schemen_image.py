"""Build Modal images from the current Schemen Gate repository.

Gate source is copied from the current clean ``schemen-gate`` checkout. This
keeps experiments on the code under review and avoids bundling unrelated
executable server artifacts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from library_provenance import collect_experiment_provenance, gate_source_digest

_BUILD_PROVENANCE_ENV = "SCHEMEN_GATE_BUILD_PROVENANCE"
_REMOTE_GATE_ROOT = Path("/opt/schemen-gate")
_REMOTE_LAUNCHER_ROOT = Path("/opt/schemen-research")


def _is_repository_root(candidate: Path) -> bool:
    """Return whether ``candidate`` is an accessible Gate repository root."""

    try:
        return (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "scripts" / "modal_source.py").is_file()
            and (candidate / "src" / "schemen_gate" / "__init__.py").is_file()
        )
    except OSError:
        return False


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"missing remote provenance input: {path.name}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_gate_mask_algorithm_identity(source: dict[str, Any]) -> str:
    """Bind the recorded mask algorithm to the verified Gate source version."""

    try:
        source_version = source["dependency_bundle"]["gate_source"]["version"]
        verified_version = source["remote_verification"]["gate_version"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("verified Gate version provenance is incomplete") from exc
    if (
        not isinstance(source_version, str)
        or not source_version
        or not isinstance(verified_version, str)
        or not verified_version
        or source_version != verified_version
    ):
        raise RuntimeError("verified Gate version differs from source provenance")
    return f"schemen_gate.GateMask.derive@{verified_version}"


def assert_remote_schemen_versions(
    source: dict[str, Any],
    *,
    launcher_name: str,
    gate_root: Path = _REMOTE_GATE_ROOT,
    launcher_root: Path = _REMOTE_LAUNCHER_ROOT,
    package_dir: Path | None = None,
) -> dict[str, Any]:
    """Return provenance measured against bytes baked into the remote image."""

    encoded_build = os.environ.get(_BUILD_PROVENANCE_ENV)
    if not encoded_build:
        raise RuntimeError("remote image is missing its build provenance")
    try:
        build_source = json.loads(encoded_build)
    except json.JSONDecodeError as exc:
        raise RuntimeError("remote image build provenance is invalid") from exc
    if not isinstance(source, dict) or source != build_source:
        raise RuntimeError("client source provenance differs from image build provenance")
    if not isinstance(build_source, dict):
        raise RuntimeError("remote image build provenance must be a JSON object")
    try:
        dependency_bundle = build_source["dependency_bundle"]
        gate_source = dependency_bundle["gate_source"]
        mismatches = dependency_bundle["mismatches"]
        expected_version = gate_source["version"]
        expected_repository = gate_source["repository"]
        expected_commit = gate_source["commit"]
        expected_tree = gate_source["tree_sha256"]
        package_files = gate_source["package_files"]
        expected_script = build_source["script_sha256"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("remote image build provenance is incomplete") from exc
    if mismatches != []:
        raise RuntimeError("remote image was built from a mismatched dependency bundle")
    if any(
        not isinstance(value, str) or not value
        for value in (
            expected_version,
            expected_repository,
            expected_commit,
            expected_tree,
            expected_script,
        )
    ):
        raise RuntimeError("remote image build provenance has invalid fields")
    if build_source.get("commit") != expected_commit:
        raise RuntimeError("experiment commit differs from its Gate source commit")
    if (
        not isinstance(package_files, list)
        or not package_files
        or any(not isinstance(name, str) for name in package_files)
    ):
        raise RuntimeError("remote image build provenance has an invalid package manifest")
    if not launcher_name or Path(launcher_name).name != launcher_name:
        raise RuntimeError("launcher_name must be one local filename")

    installed_version = metadata.version("schemen-gate")
    if installed_version != expected_version:
        raise RuntimeError(
            f"unexpected schemen-gate version in remote image: {installed_version}"
        )
    if package_dir is None:
        module = import_module("schemen_gate")
        package_dir = Path(module.__file__).resolve().parent
    else:
        module = import_module("schemen_gate")
    release = module.current_release_identity()
    actual_commit = release.require_source_commit()
    if (
        release.version != expected_version
        or release.source_repository != expected_repository
        or actual_commit != expected_commit
    ):
        raise RuntimeError("remote installed Gate release identity differs from image provenance")
    actual_tree = gate_source_digest(
        gate_root / "pyproject.toml",
        package_dir,
        package_files=package_files,
    )
    actual_script = _sha256(launcher_root / launcher_name)
    if not hmac.compare_digest(actual_tree, expected_tree):
        raise RuntimeError("remote installed Gate source differs from image provenance")
    if not hmac.compare_digest(actual_script, expected_script):
        raise RuntimeError("remote launcher bytes differ from image provenance")

    verified = json.loads(json.dumps(build_source, sort_keys=True))
    verified["remote_verification"] = {
        "gate_version": installed_version,
        "gate_repository": release.source_repository,
        "gate_source_commit": actual_commit,
        "gate_tree_sha256": actual_tree,
        "script_sha256": actual_script,
    }
    return verified


def install_current_schemen(image: Any, *, launcher: Path | None = None) -> Any:
    """Install current Gate source plus the small research preflight module."""

    repository_root = next(
        (
            candidate
            for candidate in Path(__file__).resolve().parents
            if _is_repository_root(candidate)
        ),
        None,
    )
    # Modal imports this module again inside the completed image. Construction
    # is client-side; repository source and its local-only helper are
    # intentionally absent remotely.
    if repository_root is None:
        return image
    root_text = str(repository_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.modal_source import prepare_modal_source_export

    if launcher is None:
        raise ValueError("launcher is required to bind remote experiment provenance")
    launcher = launcher.resolve()
    if not launcher.is_file() or launcher.parent != Path(__file__).resolve().parent:
        raise ValueError("launcher must be a tracked experiment source file")

    source_export = prepare_modal_source_export(repository_root)
    provenance = collect_experiment_provenance(launcher)
    gate_source = provenance["dependency_bundle"]["gate_source"]
    if (
        source_export.commit != provenance["commit"]
        or source_export.commit != gate_source["commit"]
        or list(source_export.package_files) != gate_source["package_files"]
    ):
        raise RuntimeError("Modal Gate export differs from experiment provenance")
    image = image.env(
        {
            _BUILD_PROVENANCE_ENV: json.dumps(
                provenance,
                sort_keys=True,
                separators=(",", ":"),
            )
        }
    )
    for package_file in source_export.package_files:
        image = image.add_local_file(
            source_export.root / "src" / "schemen_gate" / package_file,
            f"/opt/schemen-gate/src/schemen_gate/{package_file}",
            copy=True,
        )
    for filename in ("pyproject.toml", "PYPI.md", "LICENSE", "NOTICE"):
        image = image.add_local_file(
            source_export.root / filename,
            f"/opt/schemen-gate/{filename}",
            copy=True,
        )
    image = image.run_commands(
        "python -m pip install --no-deps /opt/schemen-gate"
    )
    image = image.add_local_file(
        launcher,
        f"{_REMOTE_LAUNCHER_ROOT}/{launcher.name}",
        copy=True,
    )
    for support_file in (
        Path(__file__),
        Path(__file__).with_name("library_provenance.py"),
        Path(__file__).with_name("execution_preflight.py"),
    ):
        image = image.add_local_file(
            support_file,
            f"/root/{support_file.name}",
            copy=True,
        )
    return image


# Compatibility name for archived command lines and external notebooks.
install_locked_schemen_wheels = install_current_schemen
