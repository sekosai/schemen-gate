"""Authenticated Schemen Gate software-release identity.

Wire-format schemas and software releases are separate version axes.  A schema
states how authenticated bytes are interpreted; :class:`GateReleaseIdentity`
states which Gate source release produced or verified those bytes.

The source commit is deliberately generated at build time.  A Git commit
cannot contain its own final object ID: changing a tracked file to add that ID
would produce a different commit.  Release builds therefore stamp the already
existing GitHub commit into the wheel and source distribution, then GitHub
attests the resulting artifact digests.
"""

from __future__ import annotations

import importlib
import re

# Source-stamp verification executes only fixed Git argument vectors without a
# shell and never includes contract or caller-controlled values.
import subprocess  # nosec B404
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

GATE_PACKAGE = "schemen-gate"
GATE_VERSION = "1.0.2"
GATE_SOURCE_REPOSITORY = "https://github.com/sekosai/schemen-gate"
GATE_RELEASE_IDENTITY_SCHEMA = "schemen/gate-release-identity-v1"

_SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
    r"(?:\.(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*))*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
_COMMIT_SHA = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_GITHUB_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+")


class ReleaseIdentityError(ValueError):
    """A release identity is malformed, unpinned, or unexpected."""


def _canonical_github_repository(value: object) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise ReleaseIdentityError("source_repository must be an exact GitHub URL")
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
        raise ReleaseIdentityError(
            "source_repository must be canonical https://github.com/OWNER/REPO"
        )
    return value


@dataclass(frozen=True)
class GateReleaseIdentity:
    """Exact software identity included in authenticated Gate contracts.

    ``source_commit=None`` is an explicit statement that an archive was not
    stamped and had no verifiable Git checkout. Clean source checkouts bind to
    their exact HEAD; production verifiers should call
    :meth:`require_source_commit` or compare against an expected identity.
    """

    package: str
    version: str
    source_repository: str
    source_commit: str | None

    def __post_init__(self) -> None:
        if self.package != GATE_PACKAGE:
            raise ReleaseIdentityError(f"package must be {GATE_PACKAGE!r}")
        if type(self.version) is not str or _SEMVER.fullmatch(self.version) is None:
            raise ReleaseIdentityError("version must be canonical SemVer")
        _canonical_github_repository(self.source_repository)
        if self.source_commit is not None and (
            type(self.source_commit) is not str or _COMMIT_SHA.fullmatch(self.source_commit) is None
        ):
            raise ReleaseIdentityError(
                "source_commit must be a lowercase 40- or 64-character Git SHA"
            )

    @property
    def is_source_pinned(self) -> bool:
        """Whether this identity names an exact source commit."""
        return self.source_commit is not None

    def require_source_commit(self) -> str:
        """Return the commit or fail closed for an unstamped build."""
        if self.source_commit is None:
            raise ReleaseIdentityError(
                "Gate release identity is unstamped; an exact source commit is required"
            )
        return self.source_commit

    def to_dict(self) -> dict[str, str | None]:
        return {
            "schema": GATE_RELEASE_IDENTITY_SCHEMA,
            "package": self.package,
            "version": self.version,
            "source_repository": self.source_repository,
            "source_commit": self.source_commit,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> GateReleaseIdentity:
        expected_fields = {
            "schema",
            "package",
            "version",
            "source_repository",
            "source_commit",
        }
        if type(value) is not dict or set(value) != expected_fields:
            raise ReleaseIdentityError("Gate release identity fields do not match the schema")
        if value.get("schema") != GATE_RELEASE_IDENTITY_SCHEMA:
            raise ReleaseIdentityError("unsupported Gate release identity schema")
        return cls(
            package=value["package"],
            version=value["version"],
            source_repository=value["source_repository"],
            source_commit=value["source_commit"],
        )


def _gate_source_checkout(module_file: str) -> Path | None:
    """Return the Git checkout that owns ``module_file`` as Gate source, if any.

    Only the canonical source-tree layout ``<checkout>/src/schemen_gate/`` binds
    a release identity to a Git checkout. A wheel installed into a virtual
    environment that merely lives inside some other repository is not a Gate
    source checkout and must neither inherit that repository's commit nor fail
    because that repository is dirty.
    """
    module_path = Path(module_file).resolve()
    package_dir = module_path.parent
    if package_dir.name != "schemen_gate" or package_dir.parent.name != "src":
        return None
    source_root = package_dir.parent.parent
    for candidate in (source_root, *source_root.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _clean_checkout_head(repository: Path, failure: str) -> str:
    """Return ``HEAD`` of a clean checkout or fail closed with ``failure``."""
    try:
        head = subprocess.run(  # nosec B603
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(("git", "diff", "--quiet"), cwd=repository, check=True)  # nosec B603
        subprocess.run(  # nosec B603
            ("git", "diff", "--cached", "--quiet"), cwd=repository, check=True
        )
        status = subprocess.run(  # nosec B603
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if status:
            raise subprocess.CalledProcessError(1, "git status --porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseIdentityError(failure) from exc
    return head


def _verify_source_checkout_stamp(module_file: str, source_commit: str) -> None:
    """Reject a stale build stamp when importing directly from a checkout."""
    repository = _gate_source_checkout(module_file)
    if repository is None:
        return
    head = _clean_checkout_head(
        repository, "stamped Gate source checkout is dirty or cannot be verified"
    )
    if source_commit != head:
        raise ReleaseIdentityError("stamped Gate source commit does not match checkout HEAD")


def _clean_source_checkout_commit(module_file: str) -> str | None:
    """Return the exact HEAD for a clean source checkout, otherwise fail closed."""
    repository = _gate_source_checkout(module_file)
    if repository is None:
        return None
    head = _clean_checkout_head(
        repository, "Gate source checkout is dirty or cannot be bound to an exact commit"
    )
    if _COMMIT_SHA.fullmatch(head) is None:
        raise ReleaseIdentityError("Gate source checkout HEAD is not a canonical commit")
    return head


@lru_cache(maxsize=1)
def current_release_identity() -> GateReleaseIdentity:
    """Return this installed build's signed-contract identity.

    A clean source checkout is bound to its exact Git HEAD. Outside a checkout,
    the generated module supplies the release stamp; an unstamped archive stays
    explicit as ``source_commit=None`` rather than trusting an environment
    variable.
    """
    source_repository = GATE_SOURCE_REPOSITORY
    source_commit = _clean_source_checkout_commit(__file__)
    if source_commit is not None:
        return GateReleaseIdentity(
            package=GATE_PACKAGE,
            version=GATE_VERSION,
            source_repository=source_repository,
            source_commit=source_commit,
        )
    try:
        _build_identity = importlib.import_module("schemen_gate._build_identity")
    except ModuleNotFoundError as exc:
        if exc.name != "schemen_gate._build_identity":
            raise
    else:
        if _build_identity.SOURCE_VERSION != GATE_VERSION:
            raise ReleaseIdentityError(
                "build stamp version does not match the installed Gate version"
            )
        source_repository = _build_identity.SOURCE_REPOSITORY
        source_commit = _build_identity.SOURCE_COMMIT
        module_path = _build_identity.__file__
        if module_path is None:
            raise ReleaseIdentityError("build identity module has no source path")
        _verify_source_checkout_stamp(module_path, source_commit)
    return GateReleaseIdentity(
        package=GATE_PACKAGE,
        version=GATE_VERSION,
        source_repository=source_repository,
        source_commit=source_commit,
    )


def release_identity_matches(
    actual: GateReleaseIdentity,
    expected: GateReleaseIdentity,
    *,
    require_source_commit: bool = True,
) -> bool:
    """Compare an authenticated identity to a verifier-owned expectation."""
    if not isinstance(actual, GateReleaseIdentity) or not isinstance(expected, GateReleaseIdentity):
        return False
    if require_source_commit and (actual.source_commit is None or expected.source_commit is None):
        return False
    return actual == expected


__all__ = [
    "GATE_PACKAGE",
    "GATE_RELEASE_IDENTITY_SCHEMA",
    "GATE_SOURCE_REPOSITORY",
    "GATE_VERSION",
    "GateReleaseIdentity",
    "ReleaseIdentityError",
    "current_release_identity",
    "release_identity_matches",
]
