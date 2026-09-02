"""Software release identity contract and build-stamp tests."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from schemen_gate import (
    GATE_PACKAGE,
    GATE_SOURCE_REPOSITORY,
    GATE_VERSION,
    GateReleaseIdentity,
    ReleaseIdentityError,
    current_release_identity,
    release_identity_matches,
)
from schemen_gate._release import (
    _clean_source_checkout_commit,
    _verify_source_checkout_stamp,
)

PINNED = GateReleaseIdentity(
    package=GATE_PACKAGE,
    version=GATE_VERSION,
    source_repository=GATE_SOURCE_REPOSITORY,
    source_commit="1" * 40,
)


def test_release_identity_round_trip_is_exact() -> None:
    assert GateReleaseIdentity.from_dict(PINNED.to_dict()) == PINNED
    assert PINNED.require_source_commit() == "1" * 40
    assert PINNED.is_source_pinned


def test_current_source_checkout_is_bound_to_an_exact_commit() -> None:
    current = current_release_identity()
    assert current.package == GATE_PACKAGE
    assert current.version == GATE_VERSION
    assert current.source_repository == GATE_SOURCE_REPOSITORY
    assert current.source_commit is not None
    assert len(current.source_commit) in {40, 64}


def test_source_checkout_rejects_untracked_nonignored_files(tmp_path: Path) -> None:
    repository = tmp_path / "checkout"
    module = repository / "src" / "schemen_gate" / "module.py"
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    subprocess.run(("git", "init", "-q", repository), check=True)
    subprocess.run(("git", "-C", repository, "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate-test@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )

    assert _clean_source_checkout_commit(str(module)) is not None
    (repository / "untracked.py").write_text("raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(ReleaseIdentityError, match="dirty"):
        _clean_source_checkout_commit(str(module))


def test_unstamped_identity_fails_when_a_source_commit_is_required() -> None:
    unstamped = replace(PINNED, source_commit=None)
    assert not unstamped.is_source_pinned
    with pytest.raises(ReleaseIdentityError, match="exact source commit"):
        unstamped.require_source_commit()
    assert not release_identity_matches(
        unstamped,
        unstamped,
        require_source_commit=True,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"package": "other"},
        {"version": "01.0.0"},
        {"source_repository": "http://github.com/sekosai/schemen-gate"},
        {"source_repository": "https://github.com/sekosai/schemen-gate.git"},
        {"source_commit": "ABC"},
    ],
)
def test_malformed_release_identity_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ReleaseIdentityError):
        GateReleaseIdentity(
            **{
                **PINNED.__dict__,
                **change,
            }
        )


def test_release_identity_parser_rejects_missing_or_unknown_fields() -> None:
    missing = PINNED.to_dict()
    missing.pop("source_commit")
    with pytest.raises(ReleaseIdentityError, match="fields"):
        GateReleaseIdentity.from_dict(missing)

    unknown = {**PINNED.to_dict(), "other": "value"}
    with pytest.raises(ReleaseIdentityError, match="fields"):
        GateReleaseIdentity.from_dict(unknown)


def test_release_comparison_is_exact_and_can_require_a_pin() -> None:
    assert release_identity_matches(PINNED, PINNED)
    assert not release_identity_matches(
        PINNED,
        replace(PINNED, source_commit="2" * 40),
    )


def _commit_fixture_repository(repository: Path, module: Path) -> None:
    module.parent.mkdir(parents=True)
    module.write_text("", encoding="utf-8")
    subprocess.run(("git", "init", "-q", repository), check=True)
    subprocess.run(("git", "-C", repository, "add", "."), check=True)
    subprocess.run(
        (
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Gate Test",
            "-c",
            "user.email=gate-test@example.invalid",
            "-c",
            "commit.gpgSign=false",
            "commit",
            "-qm",
            "fixture",
        ),
        check=True,
    )


def test_installed_copy_inside_a_foreign_repository_is_not_bound_to_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    module = project / ".venv" / "lib" / "python3" / "site-packages" / "schemen_gate" / "module.py"
    _commit_fixture_repository(project, module)
    (project / "notes.txt").write_text("uncommitted project work\n", encoding="utf-8")

    assert _clean_source_checkout_commit(str(module)) is None
    _verify_source_checkout_stamp(str(module), "1" * 40)


def test_nested_source_layout_binds_to_the_enclosing_checkout(tmp_path: Path) -> None:
    monorepo = tmp_path / "monorepo"
    module = monorepo / "vendor" / "gate" / "src" / "schemen_gate" / "module.py"
    _commit_fixture_repository(monorepo, module)

    head = _clean_source_checkout_commit(str(module))
    assert head is not None
    assert len(head) in {40, 64}
    _verify_source_checkout_stamp(str(module), head)
    with pytest.raises(ReleaseIdentityError, match="does not match"):
        _verify_source_checkout_stamp(str(module), "1" * 40)
