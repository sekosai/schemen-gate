"""Fail-closed source and dependency provenance for paper experiments."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse


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


LOCK_PATH = Path(__file__).with_name("schemen-library-lock.json")
REPOSITORY_ROOT = next(
    (
        candidate
        for candidate in LOCK_PATH.parents
        if _is_repository_root(candidate)
    ),
    LOCK_PATH.parent,
)


def _git_value(path: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), *arguments],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def gate_source_digest(
    pyproject: Path,
    package_dir: Path,
    *,
    package_files: Sequence[str] | None = None,
) -> str:
    """Hash one exact Gate package manifest using stable relative names."""

    digest = hashlib.sha256()
    paths = [("pyproject.toml", pyproject)]
    if package_files is None:
        package_files = tuple(
            path.name
            for path in sorted(package_dir.iterdir())
            if path.is_file() and (path.suffix == ".py" or path.name == "py.typed")
        )
    names = tuple(package_files)
    if (
        not names
        or len(names) != len(set(names))
        or any(
            Path(name).name != name
            or (not name.endswith(".py") and name != "py.typed")
            for name in names
        )
    ):
        raise RuntimeError("Gate source digest requires a canonical package manifest")
    actual_names = {path.name for path in package_dir.iterdir() if path.is_file()}
    if actual_names != set(names):
        raise RuntimeError(
            "Gate package files differ from the source-custody manifest: "
            f"missing={sorted(set(names) - actual_names)!r}, "
            f"unexpected={sorted(actual_names - set(names))!r}"
        )
    paths.extend(
        (f"src/schemen_gate/{name}", package_dir / name)
        for name in sorted(names)
    )
    if not pyproject.is_file() or len(paths) == 1:
        raise RuntimeError("Gate source digest requires pyproject.toml and package sources")
    for relative_name, path in paths:
        relative = relative_name.encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _prepare_source_export(*, require_clean: bool):
    root_text = str(REPOSITORY_ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from scripts.modal_source import prepare_modal_source_export

    return prepare_modal_source_export(
        REPOSITORY_ROOT,
        require_clean=require_clean,
    )


def collect_remote_dependency_provenance(*, enforce: bool = True) -> dict[str, Any]:
    """Verify the current Gate source used by local and remote runners."""

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    gate_expected = lock["libraries"]["schemen-gate"]
    commit = _git_value(REPOSITORY_ROOT, "rev-parse", "HEAD")
    status = _git_value(REPOSITORY_ROOT, "status", "--porcelain")
    mismatches: list[str] = []
    if commit is None or status is None:
        mismatches.append("schemen-gate source is not in a Git checkout")
    if status:
        mismatches.append("schemen-gate source checkout is dirty")
    source_export = _prepare_source_export(require_clean=False)
    try:
        if source_export.commit != commit:
            mismatches.append("Modal source export does not match Gate HEAD")
        if source_export.version != gate_expected["version"]:
            mismatches.append("Modal source export version differs from the research lock")
        if source_export.repository != gate_expected["repository"]:
            mismatches.append("Modal source export repository differs from the research lock")
        tree_sha256 = gate_source_digest(
            source_export.root / "pyproject.toml",
            source_export.root / "src" / "schemen_gate",
            package_files=source_export.package_files,
        )
        package_files = list(source_export.package_files)
    finally:
        source_export.cleanup()
    result = {
        "lock_schema_version": lock["schema_version"],
        "lock_sha256": hashlib.sha256(LOCK_PATH.read_bytes()).hexdigest(),
        "gate_source": {
            "repository": gate_expected["repository"],
            "version": gate_expected["version"],
            "commit": commit,
            "dirty": bool(status),
            "tree_sha256": tree_sha256,
            "package_files": package_files,
        },
        "wheels": {},
        "research_execution_preflight": lock["research_execution_preflight"],
        "mismatches": mismatches,
    }
    if enforce and mismatches:
        raise RuntimeError("Schemen dependency custody mismatch: " + "; ".join(mismatches))
    return result


def collect_wheel_provenance(*, enforce: bool = True) -> dict[str, Any]:
    """Compatibility view proving that no executable wheel is vendored."""

    dependency = collect_remote_dependency_provenance(enforce=enforce)
    return {"wheels": dependency["wheels"], "mismatches": dependency["mismatches"]}


def _direct_url(distribution: metadata.Distribution) -> dict[str, Any]:
    raw = distribution.read_text("direct_url.json")
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _package_record(distribution_name: str, module_name: str) -> dict[str, Any]:
    distribution = metadata.distribution(distribution_name)
    module = importlib.import_module(module_name)
    module_path = Path(module.__file__).resolve()
    direct_url = _direct_url(distribution)
    install_url = direct_url.get("url")
    artifact_sha256 = None
    if isinstance(install_url, str) and install_url.startswith("file:"):
        artifact_path = Path(unquote(urlparse(install_url).path))
        artifact_sha256 = _sha256(artifact_path)
    vcs_info = direct_url.get("vcs_info", {})
    commit = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    source_root = _git_value(module_path.parent, "rev-parse", "--show-toplevel")
    dirty = None
    if source_root:
        commit = _git_value(Path(source_root), "rev-parse", "HEAD")
        dirty = bool(_git_value(Path(source_root), "status", "--porcelain"))
    return {
        "version": distribution.version,
        "commit": commit,
        "source_dirty": dirty,
        "artifact_sha256": artifact_sha256,
    }


def collect_library_provenance(*, enforce: bool = True) -> dict[str, Any]:
    """Return the installed Gate version and reject source-custody drift."""

    dependency = collect_remote_dependency_provenance(enforce=False)
    records = {
        "schemen-gate": _package_record("schemen-gate", "schemen_gate"),
    }
    mismatches = list(dependency["mismatches"])
    expected = {
        "schemen-gate": dependency["gate_source"],
    }
    for name, expected_record in expected.items():
        actual = records[name]
        if actual["version"] != expected_record["version"]:
            mismatches.append(
                f"{name} version: expected {expected_record['version']!r}, "
                f"got {actual['version']!r}"
            )
        if actual["commit"] != expected_record["commit"]:
            mismatches.append(
                f"{name} commit: expected {expected_record['commit']!r}, "
                f"got {actual['commit']!r}"
            )
        if actual["source_dirty"]:
            mismatches.append(f"{name} source checkout is dirty")
    result = {
        "lock_schema_version": dependency["lock_schema_version"],
        "lock_sha256": dependency["lock_sha256"],
        "packages": records,
        "dependency_bundle": dependency,
        "mismatches": mismatches,
    }
    if enforce and mismatches:
        raise RuntimeError("Schemen library lock mismatch: " + "; ".join(mismatches))
    return result


def collect_source_provenance(repo_root: Path | None = None) -> dict[str, Any]:
    """Record the exact repository revision and whether any path is modified."""

    root = repo_root or REPOSITORY_ROOT
    commit = _git_value(root, "rev-parse", "HEAD")
    status = _git_value(root, "status", "--porcelain")
    if commit is None or status is None:
        raise RuntimeError(f"cannot resolve source provenance for {root}")
    return {
        "commit": commit,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status.encode()).hexdigest(),
    }


def collect_experiment_provenance(script: Path, *, enforce_clean: bool = True) -> dict[str, Any]:
    """Bind a remote run to clean Gate source, script bytes, and dependencies."""

    source = collect_source_provenance(REPOSITORY_ROOT)
    dependency = collect_remote_dependency_provenance(enforce=enforce_clean)
    if enforce_clean and source["dirty"]:
        raise RuntimeError("canonical remote experiment requires a clean source tree")
    return {
        **source,
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "dependency_bundle": dependency,
    }


if __name__ == "__main__":
    print(json.dumps(collect_library_provenance(), indent=2, sort_keys=True))
