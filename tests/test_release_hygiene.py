"""Release-boundary tests for the isolated repository."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_IMPORT_ROOTS = {"inference", "poc", "sekos", "schemen"}

if not (ROOT / ".github" / "workflows" / "ci.yml").is_file():
    pytest.skip(
        "repository-only release checks are unavailable in the sdist", allow_module_level=True
    )


def test_release_tree_excludes_generated_package_state() -> None:
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "build/" in ignored
    assert "dist/" in ignored
    assert "*.egg-info/" in ignored
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "CODE_OF_CONDUCT.md").is_file()
    assert (ROOT / "SECURITY.md").is_file()
    assert (ROOT / "RELEASE_MANIFEST.sha256").is_file()
    assert (ROOT / "docs" / "PRODUCTION_DEPLOYMENT.md").is_file()
    assert (ROOT / "scripts" / "release_check.py").is_file()
    assert (ROOT / "scripts" / "bootstrap_build_env.py").is_file()
    assert (ROOT / "scripts" / "build_release.py").is_file()
    assert (ROOT / "scripts" / "public_hygiene.py").is_file()
    assert (ROOT / "scripts" / "check_docs.py").is_file()
    assert (ROOT / "scripts" / "verify_history.py").is_file()
    assert (ROOT / "requirements" / "build.lock").is_file()
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include CITATION.cff" in manifest
    assert "include PYPI.md" in manifest
    assert "include release-contract.json" in manifest
    assert "include ROADMAP.md" in manifest
    assert "prune research" in manifest
    assert (ROOT / "scripts" / "verify_dist.py").is_file()


EXPECTED_TOP_LEVEL = {
    ".gitattributes",
    ".github",
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "NOTICE",
    "PYPI.md",
    "README.md",
    "RELEASE_MANIFEST.sha256",
    "ROADMAP.md",
    "SECURITY.md",
    "docs",
    "examples",
    "pyproject.toml",
    "release-contract.json",
    "requirements",
    "research",
    "scripts",
    "src",
    "tests",
}


def test_release_tree_top_level_is_exactly_the_documented_set() -> None:
    """A stray scratch file must fail the release, not ride along into the root."""

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.split("\n")
    top_level = {path.split("/", 1)[0] for path in tracked if path}
    assert top_level == EXPECTED_TOP_LEVEL, sorted(top_level ^ EXPECTED_TOP_LEVEL)


def test_source_has_no_private_workspace_imports() -> None:
    findings: list[str] = []
    for path in sorted((ROOT / "src" / "schemen_gate").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module] if node.module else []
            else:
                continue
            for module in modules:
                root = module.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS:
                    findings.append(f"{path.name}:{node.lineno}:{module}")
    assert findings == []


def test_project_urls_point_to_sekosai_repository() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/sekosai/schemen-gate" in project
    assert "github.com/ryanr" not in project
    assert "github.com/ryanr" not in readme


def test_public_hygiene_checker_passes_the_tracked_tree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/public_hygiene.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_licensing_and_citation_are_finalized() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    policy = (ROOT / "docs" / "LICENSING_DECISION.md").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "proposed adoption-first policy" not in readme
    assert "publication still requires" not in readme
    assert "Status: **adopted" in policy
    assert "Decision: **adoption-first open-source distribution**" in policy
    assert "title: Schemen Gate" in citation
    assert "version: 1.0.2" in citation
    research_citation = (ROOT / "research" / "cdp" / "CITATION.cff").read_text(encoding="utf-8")
    assert "type: software" in research_citation
    assert "version: 1.0.2" in research_citation
    assert "license: CC-BY-4.0" not in research_citation
    assert "under the path map in LICENSES.md" in research_citation


def test_declared_cryptography_floor_matches_the_x509_api_contract() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    dependencies = (ROOT / "docs" / "DEPENDENCIES.md").read_text(encoding="utf-8")

    assert project.count("cryptography>=50.0") == 2
    assert "`cryptography>=50.0`" in dependencies
    assert "cryptography>=42.0" not in project
    assert '"pypdf>=6.0"' in project
    assert '"twine==7.0.0"' in project
    assert "`pypdf>=6.0`" in dependencies
    assert "`twine==7.0.0`" in dependencies


def test_pytest_imports_the_current_checkout_source_tree() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    contributing = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    conftest = (ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")

    assert 'pythonpath = ["src"]' in project
    assert "python -m pytest -q" in contributing
    assert 'os.environ["PYTHONPATH"]' in conftest


def test_release_docs_track_current_version_and_canonical_research() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    claims = (ROOT / "docs" / "SECURITY_CLAIMS.md").read_text(encoding="utf-8")
    assert "## Install version 1.0.2" in readme
    assert "schemen_gate-1.0.2-py3-none-any.whl" in readme
    assert "github.com/sekosai/cdp-paper/blob/main" not in readme
    assert "## 1.0.2 - Production-ready release candidate" in changelog
    assert "## 1.0.1 - Historical retained release" in changelog
    assert "## 1.0.0 - Initial release line" in changelog
    assert changelog.count("## ") == 3
    assert "research/cdp/proofs/" in claims


def test_readme_quickstart_is_executable_and_public_first() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    quickstart = ROOT / "examples" / "ai_pki_quickstart.py"
    assert quickstart.is_file()
    assert "python examples/ai_pki_quickstart.py" in readme
    assert readme.index("## Two minutes to working AI PKI") < readme.index(
        "## Install version 1.0.2"
    )
    assert "Install the private" not in readme


def test_readme_states_the_complete_identity_and_custody_boundary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "external AuthN" in readme
    assert "boundary AuthZ" in readme
    assert "downstream Regime AuthN" in readme
    assert "Pkcs12KeyProvider" in readme
    assert "loads the signing key into process memory" in readme
    assert "bundle cannot nominate its own root" in readme


def test_open_source_release_contract_exists() -> None:
    contract = (ROOT / "docs" / "OPEN_SOURCE_RELEASE.md").read_text(encoding="utf-8")
    assert "complete open-source product" in contract
    assert "audited release lineage" in contract
    assert "Exclude archival refs" in contract
    assert "do not publish, tag, or" in contract
    assert "only public endpoints" in contract


def test_public_patent_notice_is_adoption_first_and_private_safe() -> None:
    notice_text = "A U.S. provisional patent application was filed before release"
    paths = (
        ROOT / "README.md",
        ROOT / "NOTICE",
        ROOT / "docs" / "LICENSING_DECISION.md",
        ROOT / "docs" / "OPEN_SOURCE_RELEASE.md",
        ROOT / "research" / "cdp" / "README.md",
    )
    documents = {path: path.read_text(encoding="utf-8") for path in paths}

    for path, document in documents.items():
        assert notice_text in " ".join(document.split()), path

    combined = "\n".join(documents.values())
    assert "adds no separate restriction" in combined
    assert "Section 3" in combined
    assert "CC BY 4.0 does not license patent rights" in combined

    result = subprocess.run(
        [
            "git",
            "grep",
            "-n",
            "-I",
            "-E",
            r"[0-9]{2}/[0-9]{3},[0-9]{3}|[A-Z]{4}\.[0-9]{3}PR",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr


def test_public_tree_uses_executable_security_evidence_not_audit_diaries() -> None:
    assert list((ROOT / "docs").glob("SECURITY_AUDIT*.md")) == []
    assert list((ROOT / "docs").glob("PUBLIC_RELEASE_SWEEP*.md")) == []
    matrix = (ROOT / "docs" / "CLAIM_TEST_MATRIX.md").read_text(encoding="utf-8")
    assert "Claim" in matrix
    assert "Executable evidence" in matrix
    assert "tests/test_ai_pki_quickstart.py" in matrix


def test_production_deployment_contract_is_complete() -> None:
    deployment = (ROOT / "docs" / "PRODUCTION_DEPLOYMENT.md").read_text(encoding="utf-8")
    for required in (
        "Required trust path",
        "Key custody",
        "Authority contract",
        "Placement and bypass closure",
        "Minimum production tests",
        "Observability and evidence",
        "Rollout sequence",
        "Production claim",
    ):
        assert required in deployment
    assert (ROOT / "scripts" / "release_check.py").is_file()


def test_release_manifest_matches_tracked_tree() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/release_manifest.py", "--verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_release_manifest_explains_when_git_metadata_is_absent(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy2(ROOT / "scripts" / "release_manifest.py", scripts)

    result = subprocess.run(
        [sys.executable, scripts / "release_manifest.py", "--verify"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "requires a Git checkout" in result.stdout
    assert "Traceback" not in result.stderr


def test_ci_builds_the_pinned_lean_research_bundle() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "leanprover/lean-action@50fcf42d2e460296f1a34b402e990d1b24f8b596" in workflow
    assert "lake-package-directory: research/cdp" in workflow
    assert "research/cdp/scripts/release_check.py" in workflow
    assert "./scripts/check_lean_claims.sh" in workflow


def test_real_model_fixture_is_bound_to_an_immutable_revision() -> None:
    fixture = (ROOT / "tests" / "test_real_model_folding.py").read_text(encoding="utf-8")
    assert 'MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"' in fixture
    assert fixture.count("revision=MODEL_REVISION") == 2


def test_complete_local_release_check_runs_headline_axiom_audit_after_lake_build() -> None:
    release_check = (ROOT / "scripts" / "release_check.py").read_text(encoding="utf-8")
    lake_build = 'run(lake, "build", cwd=ROOT / "research" / "cdp")'
    axiom_audit = 'run("bash", "scripts/check_lean_claims.sh", cwd=ROOT / "research" / "cdp")'

    assert lake_build in release_check
    assert axiom_audit in release_check
    assert release_check.index(lake_build) < release_check.index(axiom_audit)
    assert 'run(sys.executable, "-m", "twine", "check", *artifacts)' in release_check


def test_ci_attestation_is_locked_to_the_release_contract_version() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    attestation = (ROOT / ".github" / "workflows" / "release-attestation.yml").read_text(
        encoding="utf-8"
    )
    assert 'tags: ["v1.0.2"]' in workflow
    assert "github.event.workflow_run.head_branch == 'v1.0.2'" in attestation
    assert 'tags: ["v*"]' not in workflow


def test_ci_stamps_canonical_public_repository_and_runs_ai_pki_example() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "SCHEMEN_GATE_SOURCE_REPOSITORY: https://github.com/sekosai/schemen-gate" in workflow
    assert "https://github.com/${GITHUB_REPOSITORY}" not in workflow
    assert "python examples/ai_pki_quickstart.py" in workflow


def test_modal_onboarding_is_credential_safe_and_cpu_only() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    helper = (ROOT / "scripts" / "modal.sh").read_text(encoding="utf-8")
    app = (ROOT / "examples" / "modal_quickstart.py").read_text(encoding="utf-8")
    assert "./scripts/modal.sh setup" in readme
    assert "./scripts/modal.sh canary" in readme
    assert 'GATE_MODAL_VERSION="1.5.4"' in helper
    assert "pip install --upgrade pip" not in helper
    assert "token info" in helper
    assert "token set" not in helper
    assert "--token-secret" not in helper
    assert "MODAL_TOKEN_SECRET" not in helper
    assert "ai_pki_quickstart.py" in app
    assert "wrong trust root denied before Gate" in app
    assert "wrong recipient certificate denied before Gate" in app
    assert ".add_local_file(" in app
    assert ".add_local_dir(EXAMPLES" not in app
    assert ".add_local_dir(" not in app
    assert "release.require_source_commit()" in app
    assert "gpu=" not in app
    assert "requires_proxy_auth=True" in app
