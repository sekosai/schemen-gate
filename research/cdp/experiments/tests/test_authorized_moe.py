from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from authorized_moe import (
    AuthorizedExpertBank,
    LearnedTop1MoE,
    maximum_parameter_delta,
    snapshot,
    unsafe_zero_logit_route,
)
from train_authorized_moe import RegimeDataset, synthetic_datasets, train_lane


def make_lane(seed: int) -> LearnedTop1MoE:
    torch.manual_seed(seed)
    return LearnedTop1MoE(
        input_dimensions=6,
        hidden_dimensions=5,
        classes=2,
        experts=3,
    )


def test_packing_preserves_logits_predictions_and_routes() -> None:
    lanes = [make_lane(10), make_lane(11)]
    bank = AuthorizedExpertBank.pack(lanes)
    inputs = torch.randn(9, 6, generator=torch.Generator().manual_seed(12))

    for regime, lane in enumerate(lanes):
        separate, separate_trace = lane(inputs)
        packed, packed_trace = bank(inputs, regime)
        assert torch.equal(separate, packed)
        assert torch.equal(separate.argmax(dim=-1), packed.argmax(dim=-1))
        assert torch.equal(separate_trace.local_experts, packed_trace.local_experts)
        allowed = set(bank.allowed_experts(regime))
        assert set(packed_trace.global_experts.tolist()) <= allowed


def test_dense_negative_infinity_mask_matches_candidate_restriction() -> None:
    bank = AuthorizedExpertBank.pack([make_lane(20), make_lane(21)])
    inputs = torch.randn(7, 6, generator=torch.Generator().manual_seed(22))
    _, trace = bank(inputs, 1)
    dense_probabilities, dense_selected = bank.dense_masked_route(inputs, 1)

    assert torch.equal(dense_selected, trace.global_experts)
    assert torch.allclose(
        dense_probabilities[:, 3:6],
        trace.probabilities,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.count_nonzero(dense_probabilities[:, :3]) == 0


def test_invalid_regime_and_empty_pack_fail_closed() -> None:
    bank = AuthorizedExpertBank.pack([make_lane(30)])
    inputs = torch.zeros(2, 6)
    with pytest.raises(PermissionError):
        bank(inputs, -1)
    with pytest.raises(PermissionError):
        bank(inputs, 1)
    with pytest.raises(ValueError):
        AuthorizedExpertBank.pack([])


def test_training_one_regime_leaves_every_inactive_lane_unchanged() -> None:
    bank = AuthorizedExpertBank.pack([make_lane(40), make_lane(41)])
    inactive_before = snapshot(bank.lanes[1])
    optimizer = torch.optim.AdamW(bank.parameters(), lr=0.01)
    inputs = torch.randn(12, 6, generator=torch.Generator().manual_seed(42))
    labels = torch.arange(12) % 2

    optimizer.zero_grad(set_to_none=True)
    logits, trace = bank(inputs, 0)
    loss = F.cross_entropy(logits, labels) + 0.01 * bank.lanes[0].load_balance_loss(trace)
    loss.backward()
    assert all(parameter.grad is None for parameter in bank.lanes[1].parameters())
    optimizer.step()
    assert maximum_parameter_delta(inactive_before, bank.lanes[1]) == 0.0
    assert all(parameter not in optimizer.state for parameter in bank.lanes[1].parameters())


def test_zeroing_pre_softmax_logits_is_an_invalid_authorization_mask() -> None:
    logits = torch.tensor([[-2.0, -1.0, 8.0]])
    allowed = torch.tensor([[True, True, False]])
    selected = unsafe_zero_logit_route(logits, allowed)
    assert selected.item() == 2


def test_pack_copies_instead_of_aliasing_source_parameters() -> None:
    lane = make_lane(50)
    original = copy.deepcopy(lane.state_dict())
    bank = AuthorizedExpertBank.pack([lane])
    with torch.no_grad():
        next(bank.parameters()).add_(1.0)
    for name, tensor in lane.state_dict().items():
        assert torch.equal(tensor, original[name])


def test_synthetic_dataset_is_deterministic_and_regime_scoped() -> None:
    first = synthetic_datasets(
        regimes=2,
        dimensions=6,
        train_examples=12,
        test_examples=8,
        seed=60,
    )
    second = synthetic_datasets(
        regimes=2,
        dimensions=6,
        train_examples=12,
        test_examples=8,
        seed=60,
    )
    assert len(first) == 2
    assert first[0].train_inputs.shape == (12, 6)
    assert torch.equal(first[1].test_inputs, second[1].test_inputs)
    assert torch.equal(first[0].train_labels, torch.arange(12) % 2)


def test_load_balance_loss_is_finite_and_differentiable() -> None:
    lane = make_lane(70)
    inputs = torch.randn(16, 6, generator=torch.Generator().manual_seed(71))
    _, trace = lane(inputs)
    loss = lane.load_balance_loss(trace)
    loss.backward()
    assert torch.isfinite(loss)
    assert lane.router.weight.grad is not None
    assert torch.isfinite(lane.router.weight.grad).all()


def test_training_records_learned_router_parameter_movement() -> None:
    lane = make_lane(80)
    inputs = torch.randn(32, 6, generator=torch.Generator().manual_seed(81))
    labels = (inputs[:, 0] > 0).to(torch.int64)
    dataset = RegimeDataset("unit", inputs, labels, inputs, labels)
    result = train_lane(
        lane,
        dataset,
        epochs=2,
        batch_size=16,
        learning_rate=0.01,
        balance_weight=0.1,
        seed=82,
    )
    assert result["router_maximum_parameter_delta"] > 0.0


def test_canonical_artifact_satisfies_publication_contract() -> None:
    artifact_path = (
        Path(__file__).parents[1]
        / "results"
        / "authorized_learned_moe_20260831T182431_055309Z.json"
    )
    artifact = json.loads(artifact_path.read_text())
    evaluation = artifact["evaluation"]
    isolation = artifact["training_isolation"]
    runtime = artifact["runtime"]

    assert artifact["status"] == "pass"
    assert artifact["source"]["dirty"] is False
    assert evaluation["macro_accuracy"] >= 0.75
    assert evaluation["prediction_differences"] == 0
    assert evaluation["maximum_logit_difference"] == 0.0
    assert evaluation["unauthorized_dispatches"] == 0
    assert evaluation["minimum_expert_utilization_fraction"] >= 0.10
    assert artifact["minimum_router_maximum_parameter_delta"] > 0.0
    assert isolation["inactive_gradient_tensors"] == 0
    assert isolation["inactive_optimizer_states"] == 0
    assert isolation["inactive_maximum_parameter_delta"] == 0.0
    assert len(runtime["authorities"]) == 8
    assert [authority["authorized_regime_ids"] for authority in runtime["authorities"]] == [
        [regime] for regime in range(8)
    ]
    assert runtime["rejection_probe"]["all_rejected"] is True
    assert runtime["rejection_probe"]["unauthorized_model_calls"] == 0
    assert artifact["negative_control"]["unauthorized_expert_selected"] is True
