"""Release-boundary checks for the bundled CDP research snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CDP = ROOT / "research" / "cdp"

if not CDP.is_dir():
    pytest.skip("research bundle checks are unavailable in the sdist", allow_module_level=True)


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatchcase(path, pattern) for pattern in patterns)


def test_cdp_snapshot_has_exact_source_custody_and_separate_license() -> None:
    source = json.loads((CDP / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["source_repository"] == "https://github.com/sekosai/cdp-paper"
    assert source["source_commit"] == "d47b7fe880fa762b99882a6b2c3a169feb1d2958"
    assert source["history_imported"] is False
    custody_note = source["note"]
    research_readme = (CDP / "README.md").read_text(encoding="utf-8")
    assert "Absolute workstation paths in historical receipts were normalized" in custody_note
    assert "Numerical results, dependency versions, source commits" in custody_note
    assert "Historical receipts remain byte-identical" not in custody_note
    assert "absolute workstation paths in historical receipts were replaced" in research_readme
    assert "Numerical results, dependency versions, source" in research_readme


def test_gated_transformer_snapshot_has_exact_source_custody() -> None:
    module = CDP / "gated-transformer-regime-lanes"
    source = json.loads((module / "SOURCE.json").read_text(encoding="utf-8"))
    assert source["schema"] == "schemen-gate/gated-transformer-regime-lanes-import-v1"
    assert source["source_repository"] == "https://github.com/sekosai/cdp-paper"
    assert source["source_commit"] == "963e8d19cde31a441d7c92cb04593caad0df0ca0"
    assert source["source_tree"] == "3bf5a44abd760fee5e010a1417baf015cdcee585"
    assert source["source_file_count"] == 17
    assert source["source_size_bytes"] == 314799
    assert source["history_imported"] is False
    assert source["imported_into"] == (
        "https://github.com/sekosai/schemen-gate/tree/main/"
        "research/cdp/gated-transformer-regime-lanes"
    )
    assert "numerical results were not changed" in source["note"]
    assert all(item["reason"] for item in source["content_transformations"])

    tracked = subprocess.run(
        ["git", "ls-files", "research/cdp/gated-transformer-regime-lanes"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    imported = [ROOT / path for path in tracked if not path.endswith("/SOURCE.json")]
    assert len(imported) == source["source_file_count"]
    assert source["source_size_bytes"] == 314799
    assert {item["path"] for item in source["content_transformations"]} == {
        "EVIDENCE_ARCHIVE.md",
        "REPRODUCIBILITY.md",
        "SCHEMEN_GATED_TRANSFORMER_REGIME_LANES_PAPER.md",
        "toy_gated_delta_head.py",
        "toy_multi_regime_transformer.py",
        "test_toy_gated_delta_head.py",
        "test_toy_multi_regime_transformer.py",
        "results/toy_gated_delta_head_result.json",
    }
    assert "Creative Commons Attribution 4.0" in (CDP / "LICENSE").read_text()
    license_map = (CDP / "LICENSES.md").read_text(encoding="utf-8")
    assert "Apache-2.0" in license_map
    assert "CC-BY-4.0" in license_map
    assert "Third-party exceptions" in license_map
    cc_text = (CDP / "LICENSES" / "CC-BY-4.0.txt").read_text(encoding="utf-8")
    assert "Creative Commons Attribution 4.0 International Public License" in cc_text


def test_research_bundle_contains_papers_proofs_receipts_and_all_launchers() -> None:
    assert (CDP / "paper" / "cdp.tex").is_file()
    assert (CDP / "paper" / "cdp.pdf").is_file()
    assert len(list((CDP / "proofs").rglob("*.lean"))) == 21
    assert (CDP / "experiments" / "results" / "README.md").is_file()
    launchers = list((CDP / "experiments").glob("modal_*.py"))
    assert len(launchers) == 16  # fifteen launchers plus the image helper
    assert (CDP / "lakefile.lean").is_file()
    assert (CDP / "lean-toolchain").is_file()
    assert (CDP / "Makefile").is_file()
    assert (CDP / "scripts" / "release_check.py").is_file()


def test_result_manifest_binds_canonical_artifact_digests_and_hardening() -> None:
    results = CDP / "experiments" / "results"
    manifest = (results / "README.md").read_text(encoding="utf-8")
    normalized_manifest = " ".join(manifest.split())
    canonical = (
        "authorized_learned_moe_20260831T182431_055309Z.json",
        "capability_prefix_token_moe_20260831T182453_013039Z.json",
    )

    assert "regenerated after public release-boundary hardening" in normalized_manifest
    assert "These are local CPU experiments, not Modal runs." in normalized_manifest
    for filename in canonical:
        digest = hashlib.sha256((results / filename).read_bytes()).hexdigest()
        section = manifest.split(f"`{filename}`", 1)[1].split("\n## ", 1)[0]
        assert f"`{digest}`" in section


def test_public_evidence_exports_declare_custody_and_are_indexed() -> None:
    import re

    results = CDP / "experiments" / "results"
    manifest = (results / "README.md").read_text(encoding="utf-8")
    exports: list[Path] = []
    for path in sorted(results.rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        block = record.get("public_evidence_export") if isinstance(record, dict) else None
        if block is None:
            continue
        exports.append(path)
        assert block["schema"] == "schemen-gate/public-evidence-export-v1"
        assert re.fullmatch(r"[0-9a-f]{64}", block["original_sha256"])
        assert isinstance(block["transformations"], list)
        assert {entry["rule"] for entry in block["transformations"]} == set(block["rules"])
        assert re.search(rf"`(?:[A-Za-z0-9_./-]+/)?{re.escape(path.name)}`", manifest)
        assert f"`{block['original_sha256']}`" in manifest
    assert len(exports) == 14


def test_research_bundle_has_no_vendored_executable_wheels() -> None:
    experiments = CDP / "experiments"
    lock = json.loads((experiments / "schemen-library-lock.json").read_text())
    assert not list((experiments / "vendor").glob("*.whl"))
    assert set(lock["libraries"]) == {"schemen-gate"}
    assert lock["research_execution_preflight"]["production_authority_claim"] is False
    assert lock["libraries"]["schemen-gate"]["installation"] == ("current-repository-source")
    assert lock["libraries"]["schemen-gate"]["version"] == "1.0.2"
    setup = (CDP / "scripts" / "setup.sh").read_text(encoding="utf-8")
    assert '"${GATE_REPO_ROOT}[crypto,lockbox]"' in setup
    assert "experiments/vendor/schemen_gate-" not in setup
    assert not list((experiments / "cdp_research_runtime").glob("*.py"))


def test_research_python_has_no_companion_server_import() -> None:
    import ast

    findings: list[str] = []
    for path in sorted((CDP / "experiments").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            else:
                continue
            findings.extend(
                f"{path.name}:{node.lineno}:{module}"
                for module in modules
                if module.split(".", 1)[0].startswith("schemen")
                and module.split(".", 1)[0] != "schemen_gate"
            )
    assert findings == []


def test_research_modal_images_do_not_install_unused_server_stack() -> None:
    launchers = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((CDP / "experiments").glob("modal_*.py"))
    )
    assert "fastapi==" not in launchers
    assert "msgpack==" not in launchers
    assert "pydantic==" not in launchers


def test_papers_point_to_the_canonical_gate_repository() -> None:
    main_paper = (CDP / "paper" / "cdp.tex").read_text(encoding="utf-8")
    assert "github.com/sekosai/schemen-gate/tree/main/research/cdp" in main_paper
    assert "github.com/sekosai/cdp-paper" not in main_paper


def test_every_tracked_research_path_has_a_license_classification() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "research/cdp"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    relative = [path.removeprefix("research/cdp/") for path in tracked]

    third_party: tuple[str, ...] = ()
    apache = (
        "proofs/**/*.lean",
        "experiments/*.py",
        "experiments/**/*.py",
        "examples/*.py",
        "examples/**/*.py",
        "scripts/*.py",
        "scripts/*.sh",
        "scripts/*.lean",
        "scripts/**/*.py",
        "scripts/**/*.sh",
        "scripts/**/*.lean",
        "Makefile",
        "pyproject.toml",
        "lakefile.lean",
        "lake-manifest.json",
        "lean-toolchain",
        "experiments/requirements*.txt",
        "experiments/modal-recertification.json",
        "experiments/schemen-library-lock.json",
        "gated-transformer-regime-lanes/*.py",
        "gated-transformer-regime-lanes/.gitignore",
        ".gitignore",
    )
    creative_commons = (
        "paper/**",
        "output/pdf/**",
        "experiments/results/**",
        "docs/**",
        "README.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "CODE_OF_CONDUCT.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "SOURCE.json",
        "THIRD_PARTY_NOTICES.md",
        "experiments/PLAN.md",
        "experiments/README.md",
        "experiments/orthogonal-superposition-experiment.md",
        "examples/README.md",
        "gated-transformer-regime-lanes/*.md",
        "gated-transformer-regime-lanes/*.json",
        "gated-transformer-regime-lanes/*.png",
        "gated-transformer-regime-lanes/figures/**",
        "gated-transformer-regime-lanes/results/**",
    )
    packaging = ("LICENSE", "LICENSES.md", "LICENSES/**")

    unclassified: list[str] = []
    conflicting: list[str] = []
    for path in relative:
        classes = [
            name
            for name, patterns in (
                ("third-party", third_party),
                ("Apache-2.0", apache),
                ("CC-BY-4.0", creative_commons),
                ("packaging", packaging),
            )
            if _matches(path, patterns)
        ]
        if not classes:
            unclassified.append(path)
        elif len(classes) > 1:
            conflicting.append(f"{path}: {', '.join(classes)}")

    assert unclassified == []
    assert conflicting == []
