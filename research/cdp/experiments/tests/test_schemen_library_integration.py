"""Contract tests for the exact Schemen libraries used in experiment reruns."""

from __future__ import annotations

import builtins
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any

import library_provenance
import modal_schemen_image
import pytest
from execution_preflight import GateExecutionPreflight
from library_provenance import (
    collect_library_provenance,
    collect_remote_dependency_provenance,
    collect_wheel_provenance,
    gate_source_digest,
)
from local_transformer_cotenancy_suite import (
    Config,
    cargo_authorization_test,
    runtime_inference_authorization_test,
)
from modal_schemen_image import (
    assert_remote_schemen_versions,
    verified_gate_mask_algorithm_identity,
)

import schemen_gate
from schemen_gate import GateMask, GateReleaseIdentity
from scripts.modal_source import prepare_modal_source_export


def test_library_lock_matches_loaded_sources() -> None:
    provenance = collect_library_provenance()
    assert provenance["mismatches"] == []
    assert set(provenance["packages"]) == {"schemen-gate"}
    assert provenance["dependency_bundle"]["wheels"] == {}
    package = provenance["packages"]["schemen-gate"]
    assert "source_root" not in package
    assert "direct_url" not in package


def test_remote_dependency_bundle_matches_lock() -> None:
    dependency = collect_remote_dependency_provenance()
    assert dependency["mismatches"] == []
    assert dependency["gate_source"]["dirty"] is False
    assert dependency["gate_source"]["commit"]
    assert dependency["gate_source"]["tree_sha256"]

    provenance = collect_wheel_provenance()
    assert provenance["mismatches"] == []
    assert provenance["wheels"] == {}


def test_modal_image_remote_guard_needs_no_repository_helper(monkeypatch) -> None:
    monkeypatch.setattr(
        modal_schemen_image,
        "__file__",
        "/root/modal_schemen_image.py",
    )
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "scripts" or name.startswith("scripts."):
            raise ModuleNotFoundError("repository helper is absent remotely")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    image = object()
    assert modal_schemen_image.install_current_schemen(image) is image


def test_repository_root_probes_reject_inaccessible_parents(monkeypatch) -> None:
    def denied(_path: Path) -> bool:
        raise PermissionError("inaccessible parent")

    monkeypatch.setattr(Path, "is_file", denied)
    inaccessible = Path("/inaccessible")

    assert modal_schemen_image._is_repository_root(inaccessible) is False
    assert library_provenance._is_repository_root(inaccessible) is False


def test_library_provenance_imports_from_the_shallow_remote_path() -> None:
    source_path = Path(__file__).parents[1] / "library_provenance.py"
    namespace = {
        "__file__": "/root/library_provenance.py",
        "__name__": "library_provenance_remote_test",
    }

    exec(compile(source_path.read_bytes(), namespace["__file__"], "exec"), namespace)  # nosec B102

    assert namespace["LOCK_PATH"] == Path("/root/schemen-library-lock.json")
    assert namespace["REPOSITORY_ROOT"] == Path("/root")


def test_partition_algorithm_identity_uses_verified_gate_source_version() -> None:
    source = {
        "dependency_bundle": {"gate_source": {"version": "1.0.2"}},
        "remote_verification": {"gate_version": "1.0.2"},
    }
    assert (
        verified_gate_mask_algorithm_identity(source)
        == "schemen_gate.GateMask.derive@1.0.2"
    )


@pytest.mark.parametrize(
    "source",
    [
        {},
        {
            "dependency_bundle": {"gate_source": {"version": "1.0.2"}},
            "remote_verification": {"gate_version": "1.0.1"},
        },
    ],
)
def test_partition_algorithm_identity_rejects_unverified_version(
    source: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError, match="Gate version"):
        verified_gate_mask_algorithm_identity(source)


def test_remote_provenance_rejects_caller_spoof_and_measures_image_bytes(
    monkeypatch,
) -> None:
    repository_root = Path(__file__).parents[4]
    experiments = Path(__file__).parents[1]
    launcher = experiments / "modal_cargo_transformer_authorization.py"
    dependency = collect_remote_dependency_provenance(enforce=False)
    dependency["mismatches"] = []
    dependency["gate_source"]["dirty"] = False
    source = {
        "commit": dependency["gate_source"]["commit"],
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "script_sha256": hashlib.sha256(launcher.read_bytes()).hexdigest(),
        "dependency_bundle": dependency,
    }
    source_export = prepare_modal_source_export(
        repository_root,
        require_clean=False,
    )
    monkeypatch.setattr(
        schemen_gate,
        "current_release_identity",
        lambda: GateReleaseIdentity(
            package="schemen-gate",
            version=source_export.version,
            source_repository=source_export.repository,
            source_commit=source_export.commit,
        ),
    )
    try:
        assert source_export.commit == source["commit"]
        assert source["dependency_bundle"]["gate_source"]["tree_sha256"] == (
            gate_source_digest(
                source_export.root / "pyproject.toml",
                source_export.root / "src" / "schemen_gate",
                package_files=source_export.package_files,
            )
        )
        monkeypatch.setenv(
            "SCHEMEN_GATE_BUILD_PROVENANCE",
            json.dumps(source, sort_keys=True, separators=(",", ":")),
        )

        verified = assert_remote_schemen_versions(
            source,
            launcher_name=launcher.name,
            gate_root=source_export.root,
            launcher_root=experiments,
            package_dir=source_export.root / "src" / "schemen_gate",
        )
        assert verified["remote_verification"]["gate_source_commit"] == source_export.commit
        assert verified["remote_verification"]["gate_tree_sha256"] == (
            source["dependency_bundle"]["gate_source"]["tree_sha256"]
        )
        assert verified["remote_verification"]["script_sha256"] == source["script_sha256"]

        monkeypatch.setattr(
            schemen_gate,
            "current_release_identity",
            lambda: GateReleaseIdentity(
                package="schemen-gate",
                version=source_export.version,
                source_repository=source_export.repository,
                source_commit="0" * 40,
            ),
        )
        with pytest.raises(RuntimeError, match="release identity differs"):
            assert_remote_schemen_versions(
                source,
                launcher_name=launcher.name,
                gate_root=source_export.root,
                launcher_root=experiments,
                package_dir=source_export.root / "src" / "schemen_gate",
            )

        spoofed = json.loads(json.dumps(source))
        spoofed["commit"] = "attacker-commit"
        spoofed["script_sha256"] = "0" * 64
        spoofed["dependency_bundle"]["gate_source"]["tree_sha256"] = "1" * 64
        with pytest.raises(RuntimeError, match="differs from image build provenance"):
            assert_remote_schemen_versions(
                spoofed,
                launcher_name=launcher.name,
                gate_root=source_export.root,
                launcher_root=experiments,
                package_dir=source_export.root / "src" / "schemen_gate",
            )
    finally:
        source_export.cleanup()


def test_gate_v1_partition_is_complete_and_disjoint() -> None:
    key = hashlib.sha256(b"paper-rerun-gate-key-material").digest()
    masks = [
        GateMask.derive(key, regime_id, n_dims=64, n_regimes=4).to_numpy() for regime_id in range(4)
    ]
    assert all(int(mask.sum()) == 16 for mask in masks)
    assert sum(masks).min() == 1
    assert sum(masks).max() == 1


def test_cargo_scope_controls_fail_closed() -> None:
    result = cargo_authorization_test(Config())
    assert result["separation_pass"] is True
    assert result["wrong_key_model_calls"] == 0


def test_runtime_stream_controls_fail_before_backbone_callback() -> None:
    result = runtime_inference_authorization_test()
    assert result["separation_pass"] is True
    assert result["unauthorized_model_calls"] == 0


def test_execution_preflight_surface_authorizes_before_model_callback() -> None:
    surface = GateExecutionPreflight(
        model_id="test-model",
        dimensions=8,
        authorized_regime_ids=[0, 1],
    )
    callback_calls = 0

    def callback() -> str:
        nonlocal callback_calls
        callback_calls += 1
        return "evaluated"

    assert surface.invoke(1, callback) == "evaluated"
    assert callback_calls == 1
    assert surface.evidence()["authorized_model_calls"] == 1
    assert len(surface.evidence()["authorization_receipts"]) == 1


def test_execution_preflight_surface_rejects_before_model_callback() -> None:
    surface = GateExecutionPreflight(
        model_id="test-model",
        dimensions=8,
        authorized_regime_ids=[0, 1],
    )
    result = surface.rejection_probe()
    assert result["all_rejected"] is True
    assert result["unauthorized_model_calls"] == 0
    assert surface.model_calls == 0


def test_execution_preflight_evidence_is_atomic_and_detached() -> None:
    surface = GateExecutionPreflight(
        model_id="test-model",
        dimensions=8,
        authorized_regime_ids=[0, 1],
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        grants = list(pool.map(lambda _: surface.authorize(0), range(32)))

    assert len({grant.frame_id for grant in grants}) == 32
    evidence = surface.evidence()
    evidence["authorization_receipts"][0]["model_id"] = "mutated"
    assert surface.evidence()["authorization_receipts"][0]["model_id"] == "test-model"


def test_generative_runner_uses_locked_gate_and_runtime_surfaces() -> None:
    source = (Path(__file__).parents[1] / "modal_generative_intermediate.py").read_text()
    assert "install_current_schemen(" in source
    assert "GateMask.derive(" in source
    assert "GateExecutionPreflight(" in source
    assert "source = assert_remote_schemen_versions(" in source
    assert "revision=MODEL_REVISION" in source
    assert "revision=DATASET_REVISION" in source
    assert "import os" in source
    assert 'job.get("status") == "pass"' in source
    assert 'job.get("status") != "pass"' in source


def test_generative_gate_is_immediately_before_down_projection() -> None:
    source = (Path(__file__).parents[1] / "modal_generative_intermediate.py").read_text()
    assert "layer.mlp.down_proj.register_forward_pre_hook(" in source
    assert '"post-SwiGLU expanded activation, pre-down-projection"' in source
