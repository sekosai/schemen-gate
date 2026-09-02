"""Offline release usability, packaging, and source-custody contracts."""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(*command: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _modal_fixture_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "modal-source"
    package = repository / "src/schemen_gate"
    package.mkdir(parents=True)
    files = {
        "pyproject.toml": "[build-system]\nrequires = []\nbuild-backend = 'unused'\n",
        "PYPI.md": "# Schemen Gate\n",
        "LICENSE": "fixture\n",
        "NOTICE": "fixture\n",
        "release-contract.json": json.dumps(
            {
                "schema": "schemen/gate-release-contract-v1",
                "package": "schemen-gate",
                "version": "1.0.2",
                "tag": "v1.0.2",
                "repository": "https://github.com/sekosai/schemen-gate",
            },
            indent=2,
        )
        + "\n",
        "src/schemen_gate/__init__.py": "__version__ = '1.0.2'\n",
        "src/schemen_gate/py.typed": "",
    }
    for relative, contents in files.items():
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    commands = (
        ("git", "init", "--quiet"),
        ("git", "config", "user.name", "Release test"),
        ("git", "config", "user.email", "release-test@example.invalid"),
        ("git", "config", "commit.gpgsign", "false"),
        ("git", "add", "--all"),
        ("git", "commit", "--quiet", "-m", "fixture"),
    )
    for command in commands:
        result = _run(*command, cwd=repository)
        assert result.returncode == 0, result.stdout + result.stderr
    return repository


def test_machine_readable_release_contract_validates() -> None:
    result = _run(sys.executable, "scripts/validate_release_contract.py")
    assert result.returncode == 0, result.stdout + result.stderr


def test_pypi_long_description_has_only_release_bound_links() -> None:
    result = _run(sys.executable, "scripts/check_pypi_readme.py")
    assert result.returncode == 0, result.stdout + result.stderr
    pypi = (ROOT / "PYPI.md").read_text(encoding="utf-8")
    assert "/blob/main/" not in pypi
    assert "/tree/main/" not in pypi
    assert "/blob/v1.0.2/" in pypi
    assert "/tree/v1.0.2/" in pypi


def test_modal_source_export_matches_exact_tracked_checkout() -> None:
    from scripts.modal_source import prepare_modal_source_export

    export = prepare_modal_source_export(ROOT, require_clean=False)
    try:
        tracked = _run(
            "git",
            "ls-tree",
            "-r",
            "--name-only",
            export.commit,
            "--",
            "src/schemen_gate",
        )
        assert tracked.returncode == 0, tracked.stderr
        tracked_files = tuple(
            sorted(Path(line).name for line in tracked.stdout.splitlines() if line)
        )
        assert export.package_files == tuple(sorted((*tracked_files, "_build_identity.py")))
        assert "py.typed" in export.package_files
        assert not any(name.endswith((".pyc", ".pyo")) for name in export.package_files)
        assert not any("__pycache__" in name for name in export.package_files)
        for path in tracked.stdout.splitlines():
            expected = subprocess.run(
                ("git", "show", f"{export.commit}:{path}"),
                cwd=ROOT,
                check=False,
                capture_output=True,
            )
            assert expected.returncode == 0, expected.stderr
            actual = export.root / path
            assert actual.read_bytes() == expected.stdout
        identity = ast.parse(
            (export.root / "src/schemen_gate/_build_identity.py").read_text(encoding="utf-8")
        )
        values = {
            statement.targets[0].id: statement.value.value
            for statement in identity.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Constant)
        }
        assert values == {
            "SOURCE_VERSION": export.version,
            "SOURCE_REPOSITORY": export.repository,
            "SOURCE_COMMIT": export.commit,
        }
    finally:
        export.cleanup()


def test_modal_source_export_uses_only_committed_contract_and_source(
    tmp_path: Path,
) -> None:
    from scripts.modal_source import prepare_modal_source_export

    repository = _modal_fixture_repository(tmp_path)
    committed_contract = (repository / "release-contract.json").read_bytes()
    (repository / "release-contract.json").write_text("{}\n", encoding="utf-8")
    stale_identity = repository / "src/schemen_gate/_build_identity.py"
    stale_identity.write_text("raise RuntimeError('stale working-tree input')\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="clean Git checkout"):
        prepare_modal_source_export(repository)

    export = prepare_modal_source_export(repository, require_clean=False)
    try:
        assert export.version == "1.0.2"
        assert (export.root / "release-contract.json").read_bytes() == committed_contract
        generated = export.root / "src/schemen_gate/_build_identity.py"
        assert b"stale working-tree input" not in generated.read_bytes()
        assert f"SOURCE_COMMIT = {export.commit!r}" in generated.read_text(encoding="utf-8")
    finally:
        export.cleanup()


def test_modal_stamp_bytes_match_release_builder() -> None:
    from scripts.modal_source import render_build_identity
    from scripts.stamp_release import render

    commit = "a" * 40
    repository = "https://github.com/sekosai/schemen-gate"
    expected = render(version="1.0.2", repository=repository, commit=commit)
    assert render_build_identity(
        version="1.0.2",
        repository=repository,
        commit=commit,
    ) == expected.encode("utf-8")


def test_research_setup_paths_are_caller_directory_independent(tmp_path: Path) -> None:
    result = _run(
        str(ROOT / "research/cdp/scripts/setup.sh"),
        "check-paths",
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{ROOT / 'research/cdp'} -> {ROOT}" in result.stdout


def test_active_research_and_modal_dependencies_use_current_exact_pins() -> None:
    experiments = ROOT / "research/cdp/experiments"
    requirements = (experiments / "requirements.txt").read_text(encoding="utf-8")
    test_requirements = (experiments / "requirements-test.txt").read_text(encoding="utf-8")
    assert not any(
        line.strip().startswith(("-e ", "--editable ")) for line in requirements.splitlines()
    )
    setup = (ROOT / "research/cdp/scripts/setup.sh").read_text(encoding="utf-8")
    assert '"${GATE_REPO_ROOT}[crypto,lockbox]"' in setup
    assert "torch==2.13.0" in requirements
    assert "datasets==5.0.1" in requirements
    assert "cryptography==50.0.0" in requirements
    assert "modal==1.5.4" in requirements
    assert "torch==2.13.0" in test_requirements
    assert "cryptography==50.0.0" in test_requirements

    launchers = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(experiments.glob("modal_*.py"))
    )
    assert set(re.findall(r'"torch==(\d+\.\d+\.\d+)"', launchers)) == {"2.13.0"}
    assert set(re.findall(r'"datasets==(\d+\.\d+\.\d+)"', launchers)) == {"5.0.1"}
    assert set(re.findall(r'"cryptography==(\d+\.\d+\.\d+)"', launchers)) == {"50.0.0"}


def test_nested_research_commands_and_contacts_resolve() -> None:
    makefile = (ROOT / "research/cdp/Makefile").read_text(encoding="utf-8")
    examples = (ROOT / "research/cdp/examples/README.md").read_text(encoding="utf-8")
    security = (ROOT / "research/cdp/SECURITY.md").read_text(encoding="utf-8")
    assert "../../scripts/modal.sh canary" in makefile
    assert "./research/cdp/scripts/setup.sh examples" in examples
    assert "./scripts/modal.sh canary" in examples
    assert "../../SECURITY.md" in security
    assert "ryan@sekos.ai" in security
