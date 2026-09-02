"""Adversarial tests for the release artifact supply-chain boundary."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path
from types import ModuleType
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]

if not (ROOT / ".github" / "workflows" / "ci.yml").is_file():
    pytest.skip(
        "repository-only release checks are unavailable in the sdist",
        allow_module_level=True,
    )


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gate_test_{name}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_toolchain_is_exactly_versioned_and_hashed() -> None:
    lock = (ROOT / "requirements" / "build.lock").read_text(encoding="utf-8")
    expected = {
        "build==1.6.0",
        "colorama==0.4.6",
        "packaging==26.3",
        "pip==26.2.1",
        "pyproject-hooks==1.2.0",
        "setuptools==84.0.0",
        "wheel==0.47.0",
    }
    pinned = {line.split(" ", 1)[0] for line in lock.splitlines() if "==" in line}
    assert pinned == expected
    assert lock.count("--hash=sha256:") == len(expected)
    for line in lock.splitlines():
        if "--hash=sha256:" in line:
            assert len(line.rsplit(":", 1)[1]) == 64


def test_build_backend_requirements_match_the_lock() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires = ["setuptools==84.0.0", "wheel==0.47.0"]' in project
    assert "setuptools>=" not in project
    bootstrap = load_script("bootstrap_build_env")
    builder = load_script("build_release")
    assert bootstrap.EXPECTED == builder.EXPECTED_TOOLS


def test_bootstrap_pip_is_byte_pinned_before_index_access() -> None:
    bootstrap = load_script("bootstrap_build_env")
    lock = (ROOT / "requirements" / "build.lock").read_text(encoding="utf-8")
    assert bootstrap.BOOTSTRAP_PIP_URL.startswith("https://files.pythonhosted.org/packages/")
    assert bootstrap.BOOTSTRAP_PIP_URL.endswith(bootstrap.BOOTSTRAP_PIP_FILENAME)
    assert bootstrap.BOOTSTRAP_PIP_SIZE == 1_816_632
    assert f"--hash=sha256:{bootstrap.BOOTSTRAP_PIP_SHA256}" in lock
    assert bootstrap.BOOTSTRAP_PIP_FILENAME in bootstrap.EXPECTED_WHEELS


def test_build_bootstrap_preserves_posix_python_loader_path() -> None:
    source = (ROOT / "scripts" / "bootstrap_build_env.py").read_text(encoding="utf-8")
    assert 'symlinks=os.name != "nt"' in source


def test_release_builder_preserves_the_virtualenv_executable_path() -> None:
    source = (ROOT / "scripts" / "build_release.py").read_text(encoding="utf-8")
    assert "build_python = args.build_python.absolute()" in source
    assert "build_python = args.build_python.resolve()" not in source


def test_exact_bootstrap_download_rejects_corrupt_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bootstrap = load_script("bootstrap_build_env")
    payload = b"reviewed wheel bytes"

    class FakeResponse:
        status = 200
        headers: ClassVar[dict[str, str]] = {"Content-Length": str(len(payload))}

        def __init__(self) -> None:
            self._remaining = payload

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def geturl(self) -> str:
            return "https://files.pythonhosted.org/fixed.whl"

        def read(self, _size: int) -> bytes:
            chunk, self._remaining = self._remaining, b""
            return chunk

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda *_a, **_k: FakeResponse())
    destination = tmp_path / "fixed.whl"
    with pytest.raises(SystemExit, match="SHA-256 mismatch"):
        bootstrap.download_exact_wheel(
            url="https://files.pythonhosted.org/fixed.whl",
            destination=destination,
            expected_sha256="0" * 64,
            expected_size=len(payload),
        )
    assert not destination.exists()


def test_ci_attests_only_the_locked_offline_tracked_export_build() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    build_job = workflow.split("  build:\n", 1)[1].split("\n  research:", 1)[0]
    assert "python scripts/bootstrap_build_env.py" in build_job
    assert "python scripts/build_release.py" in build_job
    assert "python scripts/verify_dist.py" in build_job
    assert "python -m build" not in build_job
    assert "pip install --upgrade" not in build_job
    assert "id-token: write" not in build_job
    assert "attestations: write" not in build_job
    assert "actions/attest@" not in build_job
    assert "name: schemen-gate-dist-${{ github.sha }}" in build_job
    assert "id-token: write" not in workflow
    assert "attestations: write" not in workflow


def test_ci_attestation_authority_is_tag_only_and_release_gated() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release-attestation.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_run:" in workflow
    assert "workflows: [CI]" in workflow
    assert "types: [completed]" in workflow
    attest_job = workflow.split("\n  attest:\n", 1)[1]
    assert "github.event.workflow_run.conclusion == 'success'" in attest_job
    assert "github.event.workflow_run.event == 'push'" in attest_job
    assert "github.event.workflow_run.head_branch == 'v1.0.2'" in attest_job
    assert (
        "github.event.workflow_run.head_repository.id == github.event.repository.id" in attest_job
    )
    assert "github.event.workflow_run.head_repository.full_name == github.repository" in attest_job
    assert "github.event.workflow_run.repository.full_name == github.repository" in attest_job
    assert "github.event.workflow_run.repository.id == github.event.repository.id" in attest_job
    assert "github.event.workflow_run.path == '.github/workflows/ci.yml'" in attest_job
    assert "github.sha == github.event.workflow_run.head_sha" in attest_job
    assert "github.event.repository.visibility == 'public'" in attest_job
    assert "environment: public-release" in attest_job
    assert "actions: read" in attest_job
    assert "id-token: write" in attest_job
    assert "attestations: write" in attest_job
    assert "artifact-metadata: write" in attest_job
    assert "actions/checkout@" not in attest_job
    assert "actions/setup-python@" not in attest_job
    assert "run:" not in attest_job
    assert "name: schemen-gate-dist-${{ github.event.workflow_run.head_sha }}" in attest_job
    assert "digest-mismatch: error" in attest_job
    assert "github-token: ${{ github.token }}" in attest_job
    assert "repository: ${{ github.repository }}" in attest_job
    assert "run-id: ${{ github.event.workflow_run.id }}" in attest_job
    assert "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c" in attest_job
    assert attest_job.index("actions/download-artifact@") < attest_job.index("actions/attest@")
    assert attest_job.count("actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d") == 2
    assert "dist/schemen_gate-1.0.2-py3-none-any.whl" in attest_job
    assert "dist/schemen_gate-1.0.2.tar.gz" in attest_job
    assert "subject-path: dist/*" not in attest_job
    assert "subject-version:" not in attest_job
    assert (
        "predicate-type: https://github.com/sekosai/schemen-gate/attestations/release/v1"
        in attest_job
    )
    assert '"schema": "schemen/gate-release-attestation-v1"' in attest_job
    assert '"version": "1.0.2"' in attest_job
    assert (
        '"source_repository_id": '
        '"${{ github.event.workflow_run.head_repository.id }}"' in attest_job
    )
    assert '"source_ref": "refs/tags/v1.0.2"' in attest_job
    assert '"source_commit": "${{ github.event.workflow_run.head_sha }}"' in attest_job
    assert '"ci_workflow": ".github/workflows/ci.yml"' in attest_job
    assert '"ci_workflow_id": "${{ github.event.workflow_run.workflow_id }}"' in attest_job


def test_release_builder_rejects_an_untracked_setup_py() -> None:
    builder = load_script("build_release")
    injection = ROOT / "setup.py"
    assert not injection.exists()
    injection.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
    try:
        with pytest.raises(SystemExit, match="untracked, non-ignored paths are forbidden"):
            builder.verify_release_tree()
    finally:
        injection.unlink()


@pytest.mark.parametrize("path", ["../escape.py", "/absolute.py", "a\\b.py", "a/../b.py"])
def test_distribution_verifier_rejects_unsafe_archive_paths(path: str) -> None:
    verifier = load_script("verify_dist")
    with pytest.raises(SystemExit, match="distribution verification failed"):
        verifier.validate_archive_path(path, label="hostile path")


def test_wheel_record_verifier_rejects_unlisted_or_tampered_bytes() -> None:
    verifier = load_script("verify_dist")
    record_name = f"{verifier.DIST_INFO}/RECORD"
    payload = b"reviewed source\n"
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["schemen_gate/module.py", f"sha256={digest.decode()}", len(payload)])
    writer.writerow([record_name, "", ""])
    files = {
        "schemen_gate/module.py": payload,
        record_name: stream.getvalue().encode("utf-8"),
    }
    facts = {name: verifier.member_fact(value) for name, value in files.items()}
    verifier.verify_wheel_record(files[record_name], facts)
    files["schemen_gate/module.py"] = b"tampered\n"
    facts["schemen_gate/module.py"] = verifier.member_fact(files["schemen_gate/module.py"])
    with pytest.raises(SystemExit, match="digest or size mismatch"):
        verifier.verify_wheel_record(files[record_name], facts)


def test_distribution_verifier_rejects_zip_bombs_before_payload_read(
    tmp_path: Path,
) -> None:
    verifier = load_script("verify_dist")
    bomb = tmp_path / "bomb.whl"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("member.bin", b"\0" * (1024 * 1024))
    with zipfile.ZipFile(bomb) as archive:
        info = archive.getinfo("member.bin")
        assert info.file_size == 1024 * 1024
        assert info.compress_size < verifier.COMPRESSION_RATIO_FLOOR
        with pytest.raises(SystemExit, match="compression-ratio release bound"):
            verifier.validate_compressed_member(
                declared_size=info.file_size,
                compressed_size=info.compress_size,
                label="hostile member",
            )


def test_distribution_verifier_bounds_stream_reads() -> None:
    verifier = load_script("verify_dist")
    stream = io.BytesIO(b"x" * 1024)
    with pytest.raises(SystemExit, match="expanded beyond its declared size"):
        verifier.read_bounded(stream, declared_size=16, label="hostile member")
    assert stream.tell() == 17


@pytest.mark.parametrize("label", ["wheel", "sdist"])
def test_distribution_verifier_rejects_extra_archive_members(label: str) -> None:
    verifier = load_script("verify_dist")
    with pytest.raises(SystemExit, match=f"{label} member allowlist mismatch"):
        verifier.require_exact_members(
            actual={"reviewed.py", "unexpected.py"},
            expected={"reviewed.py"},
            label=label,
        )


def test_sdist_manifest_includes_the_reproducible_build_lock() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "recursive-include requirements *.lock" in manifest
