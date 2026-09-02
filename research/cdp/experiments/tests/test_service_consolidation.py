from __future__ import annotations

from pathlib import Path

import pytest
import torch
from service_consolidation import (
    model_parameter_bytes,
    percentile,
    require_complete_result,
    state_dict_bytes,
    summarize_timings,
)


def test_percentile_interpolates_and_rejects_invalid_input() -> None:
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([1, 2, 3, 4], 0.95) == pytest.approx(3.85)
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1], 1.1)


def test_tensor_byte_counts_do_not_double_count_tied_parameters() -> None:
    model = torch.nn.Linear(4, 3, bias=False)
    model.alias = model.weight
    expected = model.weight.numel() * model.weight.element_size()
    assert model_parameter_bytes(model) == expected
    assert state_dict_bytes(model.state_dict()) == expected


def test_timing_summary_uses_full_denominator() -> None:
    summary = summarize_timings([0.01, 0.02, 0.03], samples_per_request=4)
    assert summary["requests"] == 3
    assert summary["samples"] == 12
    assert summary["p50_latency_ms"] == pytest.approx(20.0)
    assert summary["throughput_samples_per_second"] == pytest.approx(200.0)


def test_result_contract_fails_closed() -> None:
    valid = {
        "conditions": {
            name: {}
            for name in (
                "separate_services",
                "shared_backbone_private_adapters",
                "shared_ffn_authorized_slices",
                "physically_extracted_authorized_slice",
            )
        },
        "runtime": {
            "rejection_probe": {
                "all_rejected": True,
                "unauthorized_model_calls": 0,
            }
        },
        "extraction_equivalence": {"within_tolerance": True},
    }
    require_complete_result(valid)
    valid["runtime"]["rejection_probe"]["unauthorized_model_calls"] = 1
    with pytest.raises(ValueError, match="unauthorized"):
        require_complete_result(valid)


def test_distilbert_runner_has_all_runtime_conditions_and_pinned_assets() -> None:
    source = (
        Path(__file__).parents[1]
        / "modal_distilbert_service_consolidation.py"
    ).read_text()
    for condition in (
        "separate_services",
        "shared_backbone_private_adapters",
        "shared_ffn_authorized_slices",
        "physically_extracted_authorized_slice",
    ):
        assert condition in source
    assert "GateExecutionPreflight(" in source
    assert "runtime.rejection_probe()" in source
    assert "revision=MODEL_REVISION" in source
    assert "revision=DATASET_REVISION" in source
    assert "require_complete_result(result)" in source
    assert 'with_name("service_consolidation.py")' in source
    assert "self.shared_bias = nn.Parameter(" in source
    assert "bias=False, dtype=dtype" in source
