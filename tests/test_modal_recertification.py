from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.modal_recertification import RecertificationError, validate_artifact

COMMIT = "a" * 40
LAUNCHER_SHA256 = "b" * 64
TREE_SHA256 = "c" * 64
LOCK_SHA256 = "d" * 64
FINGERPRINT = "e" * 64
REPOSITORY = "https://github.com/sekosai/schemen-gate"
AG_ASSETS = {
    "model": "distilbert-base-uncased",
    "model_revision": "12040accade4e8a0f71eabdb258fecc2e7e948be",
    "dataset": "fancyzhx/ag_news",
    "dataset_revision": "eb185aade064a813bc0b7f42de02595523103ca4",
}


def verified_source() -> dict[str, Any]:
    return {
        "commit": COMMIT,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "script_sha256": LAUNCHER_SHA256,
        "dependency_bundle": {
            "lock_schema_version": 1,
            "lock_sha256": LOCK_SHA256,
            "gate_source": {
                "repository": REPOSITORY,
                "version": "1.0.2",
                "commit": COMMIT,
                "dirty": False,
                "tree_sha256": TREE_SHA256,
                "package_files": ["__init__.py", "mask.py", "py.typed"],
            },
            "wheels": {},
            "mismatches": [],
        },
        "remote_verification": {
            "gate_version": "1.0.2",
            "gate_repository": REPOSITORY,
            "gate_source_commit": COMMIT,
            "gate_tree_sha256": TREE_SHA256,
            "script_sha256": LAUNCHER_SHA256,
        },
    }


def local_source() -> dict[str, Any]:
    source = verified_source()
    del source["remote_verification"]
    return source


def runtime() -> dict[str, Any]:
    return {
        "rejection_probe": {
            "all_rejected": True,
            "unauthorized_model_calls": 0,
        }
    }


def plan() -> dict[str, Any]:
    return {
        "release": {
            "version": "1.0.2",
            "repository": REPOSITORY,
            "source_commit": COMMIT,
        }
    }


def job(
    job_id: str,
    experiment: str,
    arguments: list[str],
    *,
    variant: str = "canary",
    records: int = 1,
    invocations: int = 1,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "expected_experiment": experiment,
        "expected_artifact_records": records,
        "remote_invocations": invocations,
        "launcher_sha256": LAUNCHER_SHA256,
        "arguments": arguments,
        "variant": variant,
    }


def dense_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    config = {
        "R": 2,
        "seed": 42,
        "train_examples": 256,
        "test_examples": 256,
        "max_length": 64,
        "batch_size": 32,
        "epochs": 1,
        "ffn_learning_rate": 2e-5,
        "head_learning_rate": 1e-3,
        "smoke": True,
    }
    result = {
        "schema_version": 1,
        "experiment": "dense_ffn_cotenancy",
        "status": "pass",
        "config": config,
        "source": verified_source(),
        "assets": AG_ASSETS,
        "gate_placement": "post-activation-pre-down-projection",
        "partition_algorithm": "schemen_gate.GateMask.derive@1.0.2",
        "maximum_frozen_shared_parameter_delta": 0.0,
        "maximum_off_partition_parameter_delta": 0.0,
        "maximum_off_partition_optimizer_moment": 0.0,
        "maximum_inactive_classifier_delta": 0.0,
        "runtime": runtime(),
    }
    artifact = {
        "schema_version": 1,
        "experiment": "dense_ffn_cotenancy_combined",
        "results": [result],
    }
    return artifact, job("dense-ffn-cotenancy", "dense_ffn_cotenancy_combined", ["--smoke"])


def private_config(design: str, *, smoke: bool) -> dict[str, Any]:
    return {
        "design": design,
        "seed": 42,
        "regimes": 4,
        "train_examples": 256 if smoke else 8000,
        "test_examples": 256 if smoke else 2000,
        "max_length": 64 if smoke else 128,
        "batch_size": 32,
        "epochs": 1 if smoke else 2,
        "learning_rate": 5e-4 if design == "adapter" else 1e-4,
        "smoke": smoke,
    }


def private_result(design: str = "adapter", *, smoke: bool = True) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "private_transformer_lanes",
        "status": "pass",
        "config": private_config(design, smoke=smoke),
        "source": verified_source(),
        "assets": AG_ASSETS,
        "gate_placement": "gate-authorized-hard-route-to-complete-private-lane",
        "maximum_shared_backbone_delta": 0.0,
        "maximum_inactive_lane_delta": 0.0,
        "runtime": runtime(),
    }


def private_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = {
        "schema_version": 1,
        "experiment": "private_transformer_lanes_combined",
        "source": local_source(),
        "results": [private_result()],
    }
    return artifact, job(
        "private-transformer-lanes", "private_transformer_lanes_combined", ["--smoke"]
    )


def factorial_config(*, reduced: bool) -> dict[str, Any]:
    return {
        "seed": 42,
        "R": 8,
        "public_examples": 256 if reduced else 8000,
        "tenant_examples": 256 if reduced else 8000,
        "test_examples": 256 if reduced else 2000,
        "max_length": 64 if reduced else 128,
        "batch_size": 32,
        "public_epochs": 1 if reduced else 2,
        "adaptation_epochs": 1,
        "tenant_epochs": 1 if reduced else 2,
        "public_encoder_lr": 2e-5,
        "public_head_lr": 1e-3,
        "adaptation_encoder_lr": 2e-5,
        "adaptation_head_lr": 5e-4,
        "tenant_ffn_lr": 2e-5,
        "tenant_head_lr": 1e-3,
        "temperature": 2.0,
        "hard_loss_weight": 0.5,
        "smoke": reduced,
    }


def factorial_result(*, reduced: bool) -> dict[str, Any]:
    zeroes = {
        "maximum_frozen_shared_parameter_delta": 0.0,
        "maximum_off_partition_parameter_delta": 0.0,
        "maximum_off_partition_optimizer_moment": 0.0,
        "maximum_inactive_classifier_delta": 0.0,
    }
    names = (
        "no_extra_public_adaptation",
        "ungated_hard_label",
        "all_mask_hard_label",
        "all_mask_distillation",
    )
    return {
        "schema_version": 2,
        "experiment": "public_gate_adaptation_factorial",
        "status": "pass",
        "config": factorial_config(reduced=reduced),
        "source": verified_source(),
        "assets": AG_ASSETS,
        "gate_placement": "post-activation-pre-down-projection",
        "partition_algorithm": "schemen_gate.GateMask.derive@1.0.2",
        "public_and_tenant_splits_disjoint": True,
        "equal_compute_adapted_conditions": {
            "same_public_examples": True,
            "same_batch_order": True,
            "same_dropout_seed_schedule": True,
            "same_optimizer_steps": True,
            "same_teacher_forward_passes": True,
            "same_forward_backward_passes": True,
            "ungated_repetitions_per_batch": 8,
        },
        "conditions": {
            name: {"condition": name, "separation_pass": True, **zeroes} for name in names
        },
        "runtime": runtime(),
    }


def factorial_fixture(*, full: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = {
        "schema_version": 2,
        "experiment": "public_gate_adaptation_factorial_combined",
        "source": local_source(),
        "preflight_pilot": factorial_result(reduced=True) if full else None,
        "summary": {"seeds": [42], "all_separation_checks_pass": True},
        "results": [factorial_result(reduced=not full)],
    }
    return artifact, job(
        "public-gate-adaptation-factorial",
        "public_gate_adaptation_factorial_combined",
        ["--seeds", "42"] if full else ["--smoke"],
        variant="compatibility" if full else "canary",
        invocations=2 if full else 1,
    )


def cargo_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    owners = [
        {
            "tenant_id": tenant,
            "sub_id": sub_id,
            "retrieved_partition": tenant,
            "correct": True,
        }
        for tenant, sub_id in (
            ("customer-alpha", "claims-team"),
            ("customer-beta", "renewals-team"),
        )
        for _ in range(2)
    ]
    attacks = (
        "random",
        "other_operation_same_scope",
        "other_tenant",
        "other_sub_id_same_tenant",
        "other_regime_same_tenant",
        "other_model_same_identity",
        "other_policy_same_identity",
        "runtime_other_model_digest",
        "runtime_other_regime",
        "runtime_wrong_dimensions",
        "runtime_empty_authority",
    )
    rows = [
        {
            "target": tenant,
            "target_sub_id": sub_id,
            "attack": attack,
            "rejected": True,
            "model_calls": 0,
        }
        for tenant, sub_id in (
            ("customer-alpha", "claims-team"),
            ("customer-beta", "renewals-team"),
        )
        for attack in attacks
    ]
    rows.append(
        {
            "target": "customer-alpha-same-regime-alias",
            "target_sub_id": "claims-team",
            "attack": "other_partition_same_regime_valid_origin_key",
            "rejected": True,
            "model_calls": 0,
        }
    )
    result = {
        "schema_version": 3,
        "experiment": "cargo_transformer_corpus_authorization",
        "status": "pass",
        "source": verified_source(),
        "assets": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_model_revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            "generation_model": "google/flan-t5-small",
            "generation_model_revision": "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab",
            "generation_model_digest": hashlib.sha256(
                b"google/flan-t5-small@0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
            ).hexdigest(),
            "model_digest": hashlib.sha256(
                (
                    "sentence-transformers/all-MiniLM-L6-v2@"
                    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41|"
                    "google/flan-t5-small@"
                    "0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab"
                ).encode()
            ).hexdigest(),
        },
        "owning_exact_recall": 1.0,
        "all_unauthorized_docks_rejected": True,
        "unauthorized_model_calls": 0,
        "load_receipts_valid": True,
        "retrieval_receipts_valid": True,
        "owning_rows": owners,
        "rejection_rows": rows,
    }
    artifact = {
        "schema_version": 3,
        "experiment": "cargo_transformer_corpus_authorization_combined",
        "result": result,
    }
    return artifact, job(
        "cargo-transformer-authorization",
        "cargo_transformer_corpus_authorization_combined",
        [],
        variant="compatibility",
    )


def orthogonal_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    config = {
        "seed": 0,
        "ratios": [8, 128],
        "train_examples": 8000,
        "test_examples": 7600,
        "max_length": 128,
        "batch_size": 64,
        "epochs": 2,
        "learning_rate": 2e-5,
        "smoke": False,
    }
    ratios = [
        {
            "R": ratio,
            "evaluated_regimes": ratio,
            "status": "pass",
            "accuracy_zero_loss": True,
            "maximum_absolute_accuracy_gap": 0.0,
            "maximum_absolute_logit_difference": 0.001,
            "baseline_accuracy": 0.8,
            "mean_accuracy": 0.8,
            "minimum_accuracy": 0.8,
            "maximum_accuracy": 0.8,
            "runtime": runtime(),
        }
        for ratio in (8, 128)
    ]
    result = {
        "schema_version": 1,
        "experiment": "orthogonal_superposition",
        "status": "pass",
        "config": config,
        "source": verified_source(),
        "assets": AG_ASSETS,
        "gate_placement": "gate-authorized-whole-model-activation",
        "scope": (
            "Whole-model permutation conjugation and serial addressed use; "
            "not sparse FFN capacity partitioning or concurrent cotenancy."
        ),
        "ratios": ratios,
    }
    artifact = {
        "schema_version": 1,
        "experiment": "orthogonal_superposition_combined",
        "source": local_source(),
        "result": result,
    }
    return artifact, job(
        "runtime-orthogonal-superposition",
        "orthogonal_superposition_combined",
        ["--ratios", "8,128"],
        variant="compatibility",
    )


def generative_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    common = {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "r": 8,
        "epochs": 1,
        "lr": 2e-5,
        "seq_len": 256,
        "batch_size": 2,
        "eval_batch_size": 4,
        "max_train_texts": 128,
        "max_eval_texts": 64,
        "max_train_blocks": 8,
        "max_eval_blocks": 4,
        "max_steps": 2,
        "canaries_per_regime": 1,
        "canary_repeats": 2,
        "canary_max_new_tokens": 12,
        "max_grad_norm": 1.0,
        "smoke": True,
    }
    probe = {
        "passed": True,
        "all_inactive_exact_zero": True,
        "any_active_delta_nonzero": True,
        "gradient_checks": {
            "gate_proj_inactive_rows_max_abs": 0.0,
            "up_proj_inactive_rows_max_abs": 0.0,
            "down_proj_inactive_columns_max_abs": 0.0,
        },
        "parameter_delta_checks": {
            "gate_proj_inactive_rows_max_abs": 0.0,
            "up_proj_inactive_rows_max_abs": 0.0,
            "down_proj_inactive_columns_max_abs": 0.0,
            "gate_proj_active_rows_max_abs": 0.1,
        },
    }
    result = {
        "schema_version": 1,
        "experiment": "tinyllama_intermediate_ffn_gate",
        "status": "pass",
        "config": {**common, "seed": 0},
        "source": verified_source(),
        "environment": {
            "determinism": {
                "deterministic_algorithms_enabled": True,
                "deterministic_algorithms_warn_only": False,
                "cublas_workspace_config": ":4096:8",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "sdpa_policy": ("pytorch_strict_deterministic_with_math_fallback"),
                "sdpa_backends": {
                    "math": True,
                    "flash": True,
                    "memory_efficient": True,
                    "cudnn": True,
                },
            }
        },
        "assets": {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_revision": "77e23968eed12d195bd46c519aa679cc22a27ddc",
            "dataset": "Salesforce/wikitext",
            "dataset_revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
        },
        "summary": {"confinement_probe_passed": True},
        "gate": {
            "semantic_location": "post-SwiGLU expanded activation, pre-down-projection",
            "intermediate_size": 5632,
            "active_dimensions_per_regime": 704,
            "partition_disjoint": True,
            "partition_complete": True,
            "mask_key_sha256": "f" * 64,
        },
        "matched_initialization": {
            "exact_match": True,
            "gated_fingerprint": FINGERPRINT,
            "ungated_control_fingerprint": FINGERPRINT,
        },
        "conditions": {
            "gated": {
                "training": {
                    "initialization_fingerprint": FINGERPRINT,
                    "optimizer_steps": 2,
                    "history": [{"optimizer_steps": 2}],
                    "confinement_probe": probe,
                }
            },
            "ungated_control": {
                "training": {
                    "initialization_fingerprint": FINGERPRINT,
                    "optimizer_steps": 2,
                    "history": [{"optimizer_steps": 2}],
                    "confinement_probe": None,
                }
            },
        },
        "runtime": runtime(),
    }
    artifact = {
        "schema_version": 1,
        "experiment": "tinyllama_intermediate_ffn_gate_combined",
        "source": local_source(),
        "config": {**common, "seeds": [0]},
        "all_jobs_ok": True,
        "jobs": [result],
    }
    return artifact, job(
        "generative-intermediate",
        "tinyllama_intermediate_ffn_gate_combined",
        ["--seeds", "0", "--r", "8", "--smoke"],
    )


def service_condition() -> dict[str, Any]:
    return {
        "checkpoint_tensor_bytes": 100,
        "resident_cuda_parameter_and_buffer_bytes": 100,
        "peak_cuda_allocated_bytes_during_load": 100,
        "warm_cache_serial_construction_to_ready_seconds": 0.1,
        "active_parameter_bytes_per_request": 100,
        "utility_accuracy": 0.75,
        "utility_correct": 48,
        "utility_total": 64,
        "timing": {
            "requests": 4,
            "samples": 32,
            "p50_latency_ms": 1.0,
            "p95_latency_ms": 2.0,
            "mean_latency_ms": 1.2,
            "throughput_samples_per_second": 100.0,
        },
    }


def service_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    config = {
        "seed": 42,
        "R": 8,
        "examples": 64,
        "max_length": 32,
        "batch_size": 8,
        "warmups": 2,
        "repetitions": 4,
        "smoke": True,
    }
    result = {
        "schema_version": 1,
        "experiment": "distilbert_runtime_service_consolidation",
        "status": "pass",
        "config": config,
        "source": verified_source(),
        "assets": {
            "model": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
            "model_revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
            "dataset": "nyu-mll/glue",
            "dataset_config": "sst2",
            "dataset_revision": "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        },
        "conditions": {
            name: service_condition()
            for name in (
                "separate_services",
                "shared_backbone_private_adapters",
                "shared_ffn_authorized_slices",
                "physically_extracted_authorized_slice",
            )
        },
        "extraction_equivalence": {
            "max_absolute_logit_difference": 0.001,
            "tolerance": 0.02,
            "within_tolerance": True,
        },
        "runtime": runtime(),
    }
    artifact = {
        "schema_version": 1,
        "experiment": "distilbert_runtime_service_consolidation_combined",
        "source": local_source(),
        "result": result,
    }
    return artifact, job(
        "distilbert-service-consolidation",
        "distilbert_runtime_service_consolidation_combined",
        ["--smoke"],
    )


def validate(tmp_path: Path, artifact: dict[str, Any], selected_job: dict[str, Any]) -> str:
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return validate_artifact(path, selected_job, plan())


@pytest.mark.parametrize(
    "factory",
    [
        dense_fixture,
        private_fixture,
        factorial_fixture,
        cargo_fixture,
        orthogonal_fixture,
        generative_fixture,
        service_fixture,
    ],
)
def test_all_canonical_artifact_schemas_pass(tmp_path: Path, factory: Any) -> None:
    artifact, selected_job = factory()
    assert len(validate(tmp_path, artifact, selected_job)) == 64


def test_exact_matrix_rejects_smoke_artifact_for_full_orthogonal_plan(
    tmp_path: Path,
) -> None:
    artifact, selected_job = orthogonal_fixture()
    artifact["result"]["config"]["ratios"] = [8]
    artifact["result"]["ratios"] = artifact["result"]["ratios"][:1]
    with pytest.raises(RecertificationError, match="config"):
        validate(tmp_path, artifact, selected_job)


def test_each_remote_record_requires_its_own_provenance(tmp_path: Path) -> None:
    artifact, selected_job = private_fixture()
    selected_job.update(
        {
            "variant": "compatibility",
            "arguments": ["--designs", "adapter,expert", "--seeds", "42"],
            "expected_artifact_records": 2,
            "remote_invocations": 2,
        }
    )
    artifact["results"] = [
        private_result("adapter", smoke=False),
        private_result("expert", smoke=False),
    ]
    del artifact["results"][1]["source"]
    with pytest.raises(RecertificationError, match="own source provenance"):
        validate(tmp_path, artifact, selected_job)


def test_remote_provenance_binds_version_repository_and_tree(tmp_path: Path) -> None:
    artifact, selected_job = dense_fixture()
    artifact["results"][0]["source"]["remote_verification"]["gate_version"] = "1.0.1"
    with pytest.raises(RecertificationError, match="remote Gate provenance"):
        validate(tmp_path, artifact, selected_job)


def test_full_factorial_requires_and_validates_pilot(tmp_path: Path) -> None:
    artifact, selected_job = factorial_fixture(full=True)
    assert len(validate(tmp_path, artifact, selected_job)) == 64
    artifact["preflight_pilot"] = None
    with pytest.raises(RecertificationError, match="pilot"):
        validate(tmp_path, artifact, selected_job)


def test_factorial_rejects_raw_nonzero_condition_delta(tmp_path: Path) -> None:
    artifact, selected_job = factorial_fixture()
    artifact["results"][0]["conditions"]["all_mask_distillation"][
        "maximum_off_partition_optimizer_moment"
    ] = 0.1
    with pytest.raises(RecertificationError, match="exact-zero"):
        validate(tmp_path, artifact, selected_job)


@pytest.mark.parametrize("field", ["partition_complete", "partition_disjoint"])
def test_generative_rejects_incomplete_gate_partition(tmp_path: Path, field: str) -> None:
    artifact, selected_job = generative_fixture()
    artifact["jobs"][0]["gate"][field] = False
    with pytest.raises(RecertificationError, match="partition evidence"):
        validate(tmp_path, artifact, selected_job)


def test_generative_rejects_unmatched_initialization(tmp_path: Path) -> None:
    artifact, selected_job = generative_fixture()
    artifact["jobs"][0]["matched_initialization"]["exact_match"] = False
    with pytest.raises(RecertificationError, match="matched initialization"):
        validate(tmp_path, artifact, selected_job)


@pytest.mark.parametrize(
    ("path", "noncanonical_value"),
    [
        (("deterministic_algorithms_enabled",), False),
        (("deterministic_algorithms_warn_only",), True),
        (("cublas_workspace_config",), ":16:8"),
        (("cudnn_benchmark",), True),
        (("cudnn_deterministic",), False),
        (("sdpa_policy",), "unrestricted"),
        (("sdpa_backends", "math"), False),
    ],
)
def test_generative_rejects_relaxed_determinism_evidence(
    tmp_path: Path, path: tuple[str, ...], noncanonical_value: object
) -> None:
    artifact, selected_job = generative_fixture()
    evidence = artifact["jobs"][0]["environment"]["determinism"]
    target = evidence
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = noncanonical_value
    with pytest.raises(RecertificationError, match="deterministic execution"):
        validate(tmp_path, artifact, selected_job)


def test_generative_rejects_missing_determinism_evidence(tmp_path: Path) -> None:
    artifact, selected_job = generative_fixture()
    del artifact["jobs"][0]["environment"]["determinism"]
    with pytest.raises(RecertificationError, match="evidence is missing"):
        validate(tmp_path, artifact, selected_job)


def test_cargo_rejects_boolean_numeric_aggregates_and_missing_attack(
    tmp_path: Path,
) -> None:
    artifact, selected_job = cargo_fixture()
    artifact["result"]["owning_exact_recall"] = True
    with pytest.raises(RecertificationError, match="acceptance contract"):
        validate(tmp_path, artifact, selected_job)
    artifact, selected_job = cargo_fixture()
    artifact["result"]["rejection_rows"].pop()
    with pytest.raises(RecertificationError, match="rejection rows"):
        validate(tmp_path, artifact, selected_job)


def test_orthogonal_rejects_boolean_zero_gap(tmp_path: Path) -> None:
    artifact, selected_job = orthogonal_fixture()
    artifact["result"]["ratios"][0]["maximum_absolute_accuracy_gap"] = False
    with pytest.raises(RecertificationError, match="exact-zero"):
        validate(tmp_path, artifact, selected_job)


def test_service_requires_all_conditions_and_consistent_extraction(
    tmp_path: Path,
) -> None:
    artifact, selected_job = service_fixture()
    del artifact["result"]["conditions"]["separate_services"]
    with pytest.raises(RecertificationError, match="conditions"):
        validate(tmp_path, artifact, selected_job)
    artifact, selected_job = service_fixture()
    artifact["result"]["extraction_equivalence"]["max_absolute_logit_difference"] = 0.03
    with pytest.raises(RecertificationError, match="exceeds tolerance"):
        validate(tmp_path, artifact, selected_job)
