"""Prepare an exact, tracked-only Gate source export for Modal images."""

from __future__ import annotations

import atexit
import json
import re
import shutil
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_COMMIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SEMVER = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)")
_ROOT_FILES = (
    "pyproject.toml",
    "PYPI.md",
    "LICENSE",
    "NOTICE",
    "release-contract.json",
)


@dataclass(frozen=True)
class ModalSourceExport:
    """Exact source bytes admitted to a Modal image."""

    root: Path
    commit: str
    version: str
    repository: str
    package_files: tuple[str, ...]

    def cleanup(self) -> None:
        """Remove only this helper's dedicated temporary export."""

        _cleanup_export(self.root)


def _cleanup_export(path: Path) -> None:
    if path.parent == Path(tempfile.gettempdir()) and path.name.startswith(
        "schemen-gate-modal-source-"
    ):
        shutil.rmtree(path, ignore_errors=True)


def _git(
    repository_root: Path,
    *arguments: str,
    text: bool = False,
) -> bytes | str:
    try:
        output = subprocess.run(  # nosec B603
            ("git", *arguments),
            cwd=repository_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Modal source export requires a readable Git checkout") from exc
    return output.decode("utf-8") if text else output


def _release_contract(repository_root: Path, commit: str) -> dict[str, str]:
    try:
        source = bytes(_git(repository_root, "show", f"{commit}:release-contract.json")).decode(
            "utf-8"
        )
        raw: Any = json.loads(source)
    except (RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Modal source export requires a valid release contract in the selected commit"
        ) from exc
    expected_fields = {"schema", "package", "version", "tag", "repository"}
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise RuntimeError("release contract fields do not match the v1 schema")
    if (
        raw.get("schema") != "schemen/gate-release-contract-v1"
        or raw.get("package") != "schemen-gate"
        or not isinstance(raw.get("version"), str)
        or _SEMVER.fullmatch(raw["version"]) is None
        or raw.get("tag") != f"v{raw['version']}"
        or raw.get("repository") != "https://github.com/sekosai/schemen-gate"
    ):
        raise RuntimeError("release contract values are invalid")
    return {key: str(value) for key, value in raw.items()}


def _tracked_package_paths(repository_root: Path, commit: str) -> tuple[str, ...]:
    output = bytes(
        _git(
            repository_root,
            "ls-tree",
            "-r",
            "-z",
            commit,
            "--",
            "src/schemen_gate",
        )
    )
    paths: list[str] = []
    for raw_entry in output.split(b"\0"):
        if not raw_entry:
            continue
        try:
            header, raw_path = raw_entry.split(b"\t", 1)
            mode, object_type, _object_id = header.decode("ascii").split(" ", 2)
            path = raw_path.decode("utf-8")
        except (UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError("Git returned an invalid source-tree entry") from exc
        pure = PurePosixPath(path)
        if (
            mode != "100644"
            or object_type != "blob"
            or pure.parts[:2] != ("src", "schemen_gate")
            or len(pure.parts) != 3
            or (pure.suffix != ".py" and pure.name != "py.typed")
            or pure.name == "_build_identity.py"
        ):
            raise RuntimeError(f"unexpected tracked Gate package entry: {path!r}")
        paths.append(path)
    if not paths or "src/schemen_gate/__init__.py" not in paths:
        raise RuntimeError("tracked Gate package is incomplete")
    if "src/schemen_gate/py.typed" not in paths:
        raise RuntimeError("tracked Gate package is missing its typing marker")
    if len(paths) != len(set(paths)):
        raise RuntimeError("tracked Gate package contains duplicate paths")
    return tuple(sorted(paths))


def render_build_identity(*, version: str, repository: str, commit: str) -> bytes:
    """Render the canonical generated release-identity module."""

    if _SEMVER.fullmatch(version) is None:
        raise RuntimeError("Gate version is not canonical SemVer")
    if repository != "https://github.com/sekosai/schemen-gate":
        raise RuntimeError("Gate repository is not the canonical release repository")
    if _COMMIT_SHA.fullmatch(commit) is None:
        raise RuntimeError("Gate source commit is not a canonical Git SHA")
    return (
        '"""Generated release identity; do not commit this file."""\n\n'
        f"SOURCE_VERSION = {version!r}\n"
        f"SOURCE_REPOSITORY = {repository!r}\n"
        f"SOURCE_COMMIT = {commit!r}\n"
    ).encode("utf-8")


def prepare_modal_source_export(
    repository_root: Path,
    *,
    require_clean: bool = True,
) -> ModalSourceExport:
    """Export exact ``HEAD`` bytes and one deliberate generated commit stamp.

    Ignored files, bytecode caches, build state, and arbitrary untracked files
    are never considered as Modal inputs. A canonical run additionally requires
    the working tree and index to be clean before the export is constructed.
    """

    repository_root = repository_root.resolve()
    commit = str(_git(repository_root, "rev-parse", "HEAD", text=True)).strip()
    if _COMMIT_SHA.fullmatch(commit) is None:
        raise RuntimeError("Git HEAD is not a canonical commit SHA")
    contract = _release_contract(repository_root, commit)
    status = str(
        _git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            text=True,
        )
    )
    if require_clean and status:
        raise RuntimeError("Modal source export requires a clean Git checkout")

    package_paths = _tracked_package_paths(repository_root, commit)
    for relative in _ROOT_FILES:
        mode = str(_git(repository_root, "ls-tree", commit, "--", relative, text=True)).split(
            maxsplit=1
        )
        if not mode or mode[0] != "100644":
            raise RuntimeError(f"required Modal source file is not a tracked blob: {relative}")

    export_root = Path(tempfile.mkdtemp(prefix="schemen-gate-modal-source-"))
    atexit.register(_cleanup_export, export_root)
    try:
        for relative in (*_ROOT_FILES, *package_paths):
            destination = export_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(bytes(_git(repository_root, "show", f"{commit}:{relative}")))
        identity_name = "_build_identity.py"
        (export_root / "src" / "schemen_gate" / identity_name).write_bytes(
            render_build_identity(
                version=contract["version"],
                repository=contract["repository"],
                commit=commit,
            )
        )
    except Exception:
        _cleanup_export(export_root)
        raise

    package_files = tuple(PurePosixPath(path).name for path in package_paths) + (identity_name,)
    return ModalSourceExport(
        root=export_root,
        commit=commit,
        version=contract["version"],
        repository=contract["repository"],
        package_files=tuple(sorted(package_files)),
    )


__all__ = [
    "ModalSourceExport",
    "prepare_modal_source_export",
    "render_build_identity",
]
