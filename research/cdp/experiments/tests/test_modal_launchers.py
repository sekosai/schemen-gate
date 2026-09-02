"""Offline contracts shared by every Modal launcher in the public bundle."""

from __future__ import annotations

import ast
import re
from pathlib import Path

EXPERIMENTS = Path(__file__).parents[1]
HELPERS = {"modal_schemen_image.py"}
LEGACY_PRIVATE_MARKERS = (
    "/Users/",
    "Path.home()",
    "schemen-workspace",
    "poc.gate_crypto",
    "schemen-git:",
)


def launchers() -> list[Path]:
    return sorted(
        path
        for path in EXPERIMENTS.glob("modal_*.py")
        if path.name not in HELPERS
    )


def test_all_fifteen_modal_launchers_are_included() -> None:
    assert len(launchers()) == 15


def test_every_remote_function_has_a_three_container_ceiling() -> None:
    for path in launchers():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        decorators = [
            decorator
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            for decorator in node.decorator_list
            if isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "function"
        ]
        assert decorators, f"{path.name} defines no Modal remote function"
        for decorator in decorators:
            values = {
                keyword.arg: keyword.value
                for keyword in decorator.keywords
                if keyword.arg is not None
            }
            ceiling = values.get("max_containers")
            assert isinstance(ceiling, ast.Constant), path.name
            assert ceiling.value == 3, path.name


def test_every_image_dependency_is_exactly_pinned() -> None:
    for path in launchers():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pip_install"
        ]
        assert package_calls, f"{path.name} defines no image dependencies"
        for call in package_calls:
            for argument in call.args:
                assert isinstance(argument, ast.Constant), path.name
                assert isinstance(argument.value, str), path.name
                assert "==" in argument.value, (
                    f"{path.name} has floating dependency {argument.value!r}"
                )


def test_generative_protobuf_pins_match_reproduction_lock() -> None:
    requirements = (EXPERIMENTS / "requirements.txt").read_text(encoding="utf-8")
    match = re.search(r"^protobuf==([^\s]+)$", requirements, flags=re.MULTILINE)
    assert match is not None
    expected = f"protobuf=={match.group(1)}"
    for name in (
        "modal_generative_ffn.py",
        "modal_generative_full.py",
        "modal_generative_intermediate.py",
    ):
        assert f'"{expected}"' in (EXPERIMENTS / name).read_text(encoding="utf-8")


def test_generative_intermediate_enforces_deterministic_gpu_execution() -> None:
    path = EXPERIMENTS / "modal_generative_intermediate.py"
    source = path.read_text(encoding="utf-8")
    cublas_setting = 'os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"'
    strict_setting = "torch.use_deterministic_algorithms(True, warn_only=False)"
    assert cublas_setting in source
    assert source.index(cublas_setting) < source.index("import torch")
    assert strict_setting in source
    assert source.index(strict_setting) < source.index(
        '"gpu": torch.cuda.get_device_name(0)'
    )
    assert "torch.backends.cudnn.benchmark = False" in source
    assert "torch.backends.cudnn.deterministic = True" in source
    assert "torch.backends.cuda.enable_math_sdp(True)" in source
    assert "enable_flash_sdp(False)" not in source
    assert "enable_mem_efficient_sdp(False)" not in source
    assert "enable_cudnn_sdp(False)" not in source
    assert '"deterministic_algorithms_enabled"' in source
    assert '"deterministic_algorithms_warn_only"' in source
    assert '"sdpa_policy"' in source
    assert '"sdpa_backends"' in source


def test_generative_intermediate_model_calls_supply_attention_masks() -> None:
    path = EXPERIMENTS / "modal_generative_intermediate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forward_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "model"
        and any(keyword.arg == "input_ids" for keyword in node.keywords)
    ]
    assert len(forward_calls) == 4
    for call in forward_calls:
        assert any(keyword.arg == "attention_mask" for keyword in call.keywords)

    generate_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "model"
        and node.func.attr == "generate"
    ]
    assert len(generate_calls) == 1
    generate_keywords = {keyword.arg for keyword in generate_calls[0].keywords}
    assert "attention_mask" in generate_keywords
    assert "max_length" in generate_keywords
    assert "max_new_tokens" not in generate_keywords
    max_length = next(
        keyword.value
        for keyword in generate_calls[0].keywords
        if keyword.arg == "max_length"
    )
    expected_max_length = ast.parse(
        'len(prompt_ids) + int(config["canary_max_new_tokens"])',
        mode="eval",
    ).body
    assert ast.dump(max_length, include_attributes=False) == ast.dump(
        expected_max_length, include_attributes=False
    )


def test_generative_intermediate_uses_current_transformers_dtype_api() -> None:
    path = EXPERIMENTS / "modal_generative_intermediate.py"
    source = path.read_text(encoding="utf-8")
    assert "torch_dtype=" not in source
    tree = ast.parse(source, filename=str(path))
    model_loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "AutoModelForCausalLM"
    ]
    assert len(model_loads) == 2
    for call in model_loads:
        keywords = {keyword.arg for keyword in call.keywords}
        assert "dtype" in keywords
        assert "torch_dtype" not in keywords


def test_partition_algorithm_metadata_uses_verified_gate_version() -> None:
    for name in (
        "modal_dense_ffn_cotenancy.py",
        "modal_public_gate_adaptation_factorial.py",
    ):
        source = (EXPERIMENTS / name).read_text(encoding="utf-8")
        assert (
            '"partition_algorithm": verified_gate_mask_algorithm_identity(source)'
            in source
        )
        assert "schemen_gate.GateMask.derive@1.0.1" not in source


def test_canonical_recertification_launchers_have_bounded_remote_timeouts() -> None:
    expected = {
        "modal_dense_ffn_cotenancy.py": "timeout=10 * 60",
        "modal_private_transformer_lanes.py": "timeout=10 * 60",
        "modal_public_gate_adaptation_factorial.py": "timeout=20 * 60",
        "modal_cargo_transformer_authorization.py": "timeout=5 * 60",
        "modal_orthogonal_superposition.py": "timeout=30 * 60",
        "modal_generative_intermediate.py": "timeout=50 * 60",
        "modal_distilbert_service_consolidation.py": "timeout=5 * 60",
    }
    for name, timeout in expected.items():
        assert timeout in (EXPERIMENTS / name).read_text(encoding="utf-8")


def test_launchers_have_no_private_workspace_dependency() -> None:
    for path in launchers():
        source = path.read_text(encoding="utf-8")
        for marker in LEGACY_PRIVATE_MARKERS:
            assert marker not in source, f"{path.name} contains {marker!r}"


def test_every_launcher_has_a_local_entrypoint() -> None:
    for path in launchers():
        assert "@app.local_entrypoint()" in path.read_text(encoding="utf-8"), path.name


def test_gate_using_launchers_install_current_repository_source() -> None:
    gate_launchers = {
        "modal_cargo_transformer_authorization.py",
        "modal_dense_ffn_cotenancy.py",
        "modal_distilbert_service_consolidation.py",
        "modal_generative_ffn.py",
        "modal_generative_full.py",
        "modal_generative_intermediate.py",
        "modal_kv_cache_pollution.py",
        "modal_private_transformer_lanes.py",
        "modal_public_gate_adaptation_factorial.py",
        "modal_orthogonal_superposition.py",
    }
    for path in launchers():
        source = path.read_text(encoding="utf-8")
        if path.name in gate_launchers:
            assert "install_current_schemen(" in source, path.name
            assert "launcher=Path(__file__)" in source, path.name
        if "collect_experiment_provenance" in source:
            assert "source = assert_remote_schemen_versions(" in source, path.name
            assert "launcher_name=Path(__file__).name" in source, path.name


def test_no_unresolved_main_revision_defaults_remain() -> None:
    for path in launchers():
        source = path.read_text(encoding="utf-8")
        assert 'revision: str = "main"' not in source, path.name
        assert 'revision="main"' not in source, path.name


def test_external_model_and_dataset_loads_are_revision_pinned() -> None:
    for path in launchers():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            function_name = (
                call.func.attr
                if isinstance(call.func, ast.Attribute)
                else call.func.id
                if isinstance(call.func, ast.Name)
                else None
            )
            if function_name not in {"from_pretrained", "load_dataset"}:
                continue
            owner = (
                call.func.value.id
                if isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                else None
            )
            # PEFT adapter paths are local artifacts produced in the same job.
            if function_name == "from_pretrained" and owner == "PeftModel":
                continue
            keywords = {keyword.arg for keyword in call.keywords}
            assert "revision" in keywords, (
                f"{path.name}:{call.lineno} has an unpinned {function_name} call"
            )


def test_declared_remote_revisions_are_immutable_git_commits() -> None:
    for path in launchers():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id.endswith("REVISION")
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                assert re.fullmatch(r"[0-9a-f]{40}", node.value.value), (
                    f"{path.name}:{node.lineno} has a mutable revision"
                )


def test_revision_override_launchers_fail_closed_on_non_commits() -> None:
    adaptation = (EXPERIMENTS / "modal_matched_adaptation.py").read_text(
        encoding="utf-8"
    )
    deposition = (EXPERIMENTS / "modal_matched_deposition.py").read_text(
        encoding="utf-8"
    )
    assert 're.fullmatch(r"[0-9a-f]{40}", revision_value)' in adaptation
    assert 're.fullmatch(r"[0-9a-f]{40}", model_revision)' in deposition
