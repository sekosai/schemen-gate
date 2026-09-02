"""Fast, deterministic checks for a public CDP paper release candidate."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_FILES = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    "docs/OPEN_SOURCE_PLAN.md",
    "paper/cdp.tex",
    "paper/cdp.pdf",
    "experiments/schemen-library-lock.json",
    "experiments/results/README.md",
    "lakefile.lean",
    "lake-manifest.json",
    "lean-toolchain",
)

CANONICAL_DIGESTS = {
    "experiments/results/authorized_learned_moe_20260831T182431_055309Z.json": (
        "8988c2bb2e9dc8ddda08da6b4f75cf043bd05c2d8c234e702211e48fbe4f1c03"
    ),
    "experiments/results/capability_prefix_token_moe_20260831T182453_013039Z.json": (
        "37e0323a0d4657cf4f06739974476e0aa7ce94daefda494b73d43fc3d986cdc1"
    ),
}

EXPECTED_AXIOMS = {
    "Recovers",
    "prf_brute_force_optimal",
    "IsDistributionMatched",
    "camouflage_indistinguishable",
    "gradient_probing_hard",
}

PINNED_HUGGING_FACE_INPUTS = {
    "orthogonal_superposition_sweep.py": {
        "MODEL_REVISION": "12040accade4e8a0f71eabdb258fecc2e7e948be",
        "DATASET_REVISION": "eb185aade064a813bc0b7f42de02595523103ca4",
    },
    "true_multiplexing_test.py": {
        "MODEL_REVISION": "12040accade4e8a0f71eabdb258fecc2e7e948be",
        "DATASET_REVISION": "eb185aade064a813bc0b7f42de02595523103ca4",
    },
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_lean_comments(text: str) -> str:
    """Remove nested Lean comments while preserving line boundaries."""

    output: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        pair = text[index : index + 2]
        if pair == "/-":
            depth += 1
            output.extend("  ")
            index += 2
        elif pair == "-/" and depth:
            depth -= 1
            output.extend("  ")
            index += 2
        elif depth:
            output.append("\n" if text[index] == "\n" else " ")
            index += 1
        else:
            output.append(text[index])
            index += 1
    if depth:
        fail("unterminated Lean block comment")
    return "".join(output)


def check_required_files() -> None:
    missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
    if missing:
        fail(f"missing release files: {missing}")


def check_json_and_digests() -> None:
    json_paths = sorted((ROOT / "experiments/results").rglob("*.json"))
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    if not json_paths:
        fail("no result artifacts found")

    result_manifest = (ROOT / "experiments/results/README.md").read_text(
        encoding="utf-8"
    )
    for relative, expected in CANONICAL_DIGESTS.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            fail(f"canonical digest mismatch for {relative}: {actual}")
        filename = Path(relative).name
        marker = f"`{filename}`"
        if marker not in result_manifest:
            fail(f"result manifest does not name canonical artifact {filename}")
        section = result_manifest.split(marker, 1)[1].split("\n## ", 1)[0]
        if f"`{expected}`" not in section:
            fail(f"result manifest digest mismatch for {filename}")

    lock = json.loads(
        (ROOT / "experiments/schemen-library-lock.json").read_text(
            encoding="utf-8"
        )
    )
    for package, record in lock["libraries"].items():
        if record.get("installation") == "current-repository-source":
            gate_root = ROOT.parents[1]
            gate_init = gate_root / "src" / "schemen_gate" / "__init__.py"
            gate_release = gate_root / "src" / "schemen_gate" / "_release.py"
            gate_project = gate_root / "pyproject.toml"
            if (
                package != "schemen-gate"
                or not gate_init.is_file()
                or not gate_release.is_file()
                or not gate_project.is_file()
            ):
                fail(f"invalid current-repository-source binding for {package}")
            version_marker = f'GATE_VERSION = "{record["version"]}"'
            if version_marker not in gate_release.read_text(encoding="utf-8"):
                fail(f"current Gate source version mismatch for {record['version']}")
            continue
        wheel = ROOT / "experiments" / record["wheel"]
        actual = sha256(wheel)
        if actual != record["wheel_sha256"]:
            fail(f"locked wheel digest mismatch for {package}: {actual}")

    print(f"JSON artifacts: {len(json_paths)} valid")


RESULT_REFERENCE = re.compile(
    r"[A-Za-z0-9_\-]+_\d{8}(?:T\d{6}_\d{6}Z|_\d{6})(?:_[A-Za-z0-9_\-]+)?\.(?:json|txt)"
)
EVIDENCE_EXPORT_SCHEMA = "schemen-gate/public-evidence-export-v1"


def tracked_research_paths() -> list[str]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode()
    return [name for name in output.split("\0") if name]


def check_concrete_artifact_references() -> None:
    """Every concrete result filename named in Markdown or TeX must be tracked."""

    tracked = tracked_research_paths()
    basenames = {Path(name).name for name in tracked}
    unresolved: list[str] = []
    references = 0
    for name in tracked:
        if not name.endswith((".md", ".tex")):
            continue
        text = (ROOT / name).read_text(encoding="utf-8")
        for match in RESULT_REFERENCE.finditer(text):
            references += 1
            if match.group(0) not in basenames:
                line = text[: match.start()].count("\n") + 1
                unresolved.append(f"{name}:{line}: {match.group(0)}")
    if unresolved:
        fail("concrete result references without a tracked artifact:\n" + "\n".join(unresolved))
    print(f"Artifact references: {references} concrete result filenames resolve")


def check_public_evidence_exports() -> None:
    """Exports of privately retained run records must declare custody and be indexed."""

    manifest = (ROOT / "experiments/results/README.md").read_text(encoding="utf-8")
    exports = 0
    for path in sorted((ROOT / "experiments/results").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        block = record.get("public_evidence_export") if isinstance(record, dict) else None
        if block is None:
            continue
        exports += 1
        if block.get("schema") != EVIDENCE_EXPORT_SCHEMA:
            fail(f"{path.name}: unexpected evidence-export schema")
        digest = block.get("original_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            fail(f"{path.name}: evidence export lacks the original record digest")
        transformations = block.get("transformations")
        rules = block.get("rules")
        if not isinstance(transformations, list) or not isinstance(rules, dict):
            fail(f"{path.name}: evidence export does not record its transformations")
        if {entry.get("rule") for entry in transformations} != set(rules):
            fail(f"{path.name}: evidence export rules do not match its transformations")
        indexed = re.search(rf"`(?:[A-Za-z0-9_./-]+/)?{re.escape(path.name)}`", manifest)
        if indexed is None or f"`{digest}`" not in manifest:
            fail(f"{path.name}: evidence export is not indexed with its original digest")
    print(f"Evidence exports: {exports} documented public-safe exports indexed")


def check_python_sources() -> None:
    paths = sorted((ROOT / "experiments").rglob("*.py"))
    paths += sorted((ROOT / "examples").rglob("*.py"))
    paths += sorted((ROOT / "scripts").rglob("*.py"))
    forbidden_imports: list[str] = []
    for path in paths:
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
                if root.startswith("schemen") and root != "schemen_gate":
                    forbidden_imports.append(
                        f"{path.relative_to(ROOT)}:{node.lineno}:{module}"
                    )
    if forbidden_imports:
        fail("companion server imports found:\n" + "\n".join(forbidden_imports))
    if list((ROOT / "experiments" / "cdp_research_runtime").glob("*.py")):
        fail("obsolete duplicate research implementation is present")
    print(f"Python sources: {len(paths)} parsed")


def check_research_input_pins() -> None:
    experiments = ROOT / "experiments"
    for filename, expected in PINNED_HUGGING_FACE_INPUTS.items():
        tree = ast.parse(
            (experiments / filename).read_text(encoding="utf-8"),
            filename=filename,
        )
        constants = {
            node.targets[0].id: node.value.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        }
        for name, revision in expected.items():
            if constants.get(name) != revision:
                fail(f"{filename} does not pin {name} to {revision}")

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            called = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if isinstance(node.func, ast.Name):
                called = node.func.id
            if called not in {"from_pretrained", "load_dataset"}:
                continue
            revision = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "revision"),
                None,
            )
            if not isinstance(revision, ast.Name) or revision.id not in expected:
                fail(f"{filename} has an unpinned {called} call")
    print("Research inputs: immutable Hugging Face revisions enforced")


def check_lean_inventory() -> None:
    paths = sorted((ROOT / "proofs").rglob("*.lean"))
    if len(paths) != 21:
        fail(f"expected 21 Lean proof modules, found {len(paths)}")
    stripped = "\n".join(
        strip_lean_comments(path.read_text(encoding="utf-8")) for path in paths
    )
    theorem_count = len(re.findall(r"^\s*(?:theorem|lemma)\s+", stripped, re.M))
    if theorem_count != 312:
        fail(f"expected 312 theorem/lemma declarations, found {theorem_count}")
    if re.search(r"\bsorry\b|\badmit\b", stripped):
        fail("active Lean source contains sorry or admit")
    axiom_names = set(
        re.findall(r"^\s*axiom\s+([A-Za-z0-9_']+)", stripped, re.M)
    )
    if axiom_names != EXPECTED_AXIOMS:
        fail(f"unexpected custom axiom inventory: {sorted(axiom_names)}")
    if re.search(r"^\s*module\s*$", stripped, re.M):
        fail("standalone module command reintroduced into mixed proof graph")
    print(
        "Lean inventory: 21 modules, 312 theorem/lemma declarations, "
        "5 declared custom axioms, no sorry/admit"
    )


def check_markdown_links() -> None:
    broken: list[str] = []
    pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    for path in sorted(ROOT.rglob("*.md")):
        if any(part in {".git", ".lake", ".venv", ".venv-modal"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            target = match.group(1).split("#", 1)[0]
            if (
                not target
                or "://" in target
                or target.startswith("mailto:")
            ):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                line = text[: match.start()].count("\n") + 1
                broken.append(f"{path.relative_to(ROOT)}:{line}: {target}")
    if broken:
        fail("broken local Markdown links:\n" + "\n".join(broken))
    print("Markdown links: local targets exist")


def check_repository_hygiene() -> None:
    requirements = (ROOT / "experiments/requirements.txt").read_text(
        encoding="utf-8"
    )
    if "git+https://github.com/sekosai/" in requirements:
        fail("public requirements still depend on private GitHub source")
    if list((ROOT / "experiments" / "vendor").glob("*.whl")):
        fail("vendored executable wheels are not allowed")

    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode().split("\0")
    oversized = [
        name
        for name in tracked
        if name and (ROOT / name).is_file() and (ROOT / name).stat().st_size > 5_000_000
    ]
    if oversized:
        fail(f"tracked files exceed 5 MB: {oversized}")

    sensitive_patterns = (
        re.compile(rb"gh[opurs]_[A-Za-z0-9]{30,}"),
        re.compile(rb"AKIA[0-9A-Z]{16}"),
        re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(rb"(?:ak|as)-[A-Za-z0-9_-]{24,}"),
        re.compile(rb"/Users/[A-Za-z0-9._-]+/"),
        re.compile(rb"/home/[A-Za-z0-9._-]+/"),
        re.compile(rb"file:///(?:Users|home)/"),
        re.compile(rb"\." + rb"codex-worktrees/"),
    )
    hits: list[str] = []
    for name in tracked:
        path = ROOT / name
        if not name or not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in sensitive_patterns):
            hits.append(name)
    if hits:
        fail(f"possible credential material in tracked files: {hits}")
    print(f"Repository hygiene: {len([name for name in tracked if name])} tracked files checked")


def main() -> int:
    checks = (
        check_required_files,
        check_json_and_digests,
        check_concrete_artifact_references,
        check_public_evidence_exports,
        check_python_sources,
        check_research_input_pins,
        check_lean_inventory,
        check_markdown_links,
        check_repository_hygiene,
    )
    try:
        for check in checks:
            check()
    except Exception as exc:
        print(f"RELEASE CHECK FAILED: {exc}", file=sys.stderr)
        return 1
    print("Release candidate checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
