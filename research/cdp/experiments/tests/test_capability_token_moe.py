from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from capability_token_moe import (
    CapabilityTokenBank,
    CapabilityTokenLane,
    maximum_parameter_delta,
    snapshot,
)
from train_capability_token_moe import encode_texts, token_id


def make_lane(seed: int) -> CapabilityTokenLane:
    torch.manual_seed(seed)
    return CapabilityTokenLane(
        vocabulary_size=32,
        embedding_dimensions=8,
        hidden_dimensions=6,
        classes=2,
        experts=2,
    )


def test_token_routes_never_leave_runtime_selected_expert_set() -> None:
    bank = CapabilityTokenBank([make_lane(1), make_lane(2)])
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long)
    _, trace = bank(tokens, 1)
    assert set(trace.global_experts.tolist()) <= {2, 3}


def test_packing_preserves_token_routes_logits_and_predictions_exactly() -> None:
    lanes = [make_lane(3), make_lane(4)]
    bank = CapabilityTokenBank(lanes)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    for regime, lane in enumerate(lanes):
        separate, separate_trace = lane(tokens)
        packed, packed_trace = bank(tokens, regime)
        assert torch.equal(separate, packed)
        assert torch.equal(separate.argmax(dim=-1), packed.argmax(dim=-1))
        assert torch.equal(separate_trace.local_experts, packed_trace.local_experts)


def test_dense_authorization_mask_matches_candidate_restriction() -> None:
    bank = CapabilityTokenBank([make_lane(5), make_lane(6)])
    tokens = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]], dtype=torch.long)
    _, trace = bank(tokens, 0)
    dense_probabilities, dense_selected = bank.dense_masked_route(tokens, 0)
    assert torch.equal(dense_selected, trace.global_experts)
    assert torch.allclose(
        dense_probabilities[:, :2],
        trace.probabilities,
        atol=1e-7,
        rtol=0.0,
    )
    assert torch.count_nonzero(dense_probabilities[:, 2:]) == 0


def test_user_tokens_cannot_select_another_capability_prefix() -> None:
    bank = CapabilityTokenBank([make_lane(7), make_lane(8)])
    # Token 31 stands in for a literal user-authored "CAPABILITY_REGIME_1".
    spoofed_user_tokens = torch.tensor([[31, 31, 4, 5]], dtype=torch.long)
    _, trace = bank(spoofed_user_tokens, 0)
    assert set(trace.global_experts.tolist()) <= {0, 1}


def test_invalid_runtime_regime_fails_before_prefix_selection() -> None:
    bank = CapabilityTokenBank([make_lane(9)])
    tokens = torch.tensor([[1, 2]], dtype=torch.long)
    try:
        bank(tokens, 1)
    except PermissionError:
        pass
    else:
        raise AssertionError("unauthorized regime did not fail closed")

    for malformed in (True, 0.0, "0"):
        try:
            bank(tokens, malformed)  # type: ignore[arg-type]
        except PermissionError:
            pass
        else:
            raise AssertionError("malformed regime authority did not fail closed")


def test_one_training_step_leaves_other_prefix_router_and_experts_unchanged() -> None:
    bank = CapabilityTokenBank([make_lane(10), make_lane(11)])
    inactive_before = snapshot(bank.lanes[1])
    optimizer = torch.optim.AdamW(bank.parameters(), lr=0.01)
    tokens = torch.tensor(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        dtype=torch.long,
    )
    labels = torch.tensor([0, 1, 0, 1])
    optimizer.zero_grad(set_to_none=True)
    logits, trace = bank(tokens, 0)
    loss = F.cross_entropy(logits, labels) + 0.1 * bank.lanes[0].load_balance_loss(trace)
    loss.backward()
    assert all(parameter.grad is None for parameter in bank.lanes[1].parameters())
    optimizer.step()
    assert maximum_parameter_delta(inactive_before, bank.lanes[1]) == 0.0
    assert all(parameter not in optimizer.state for parameter in bank.lanes[1].parameters())


def test_empty_documents_are_rejected() -> None:
    lane = make_lane(12)
    tokens = torch.zeros((1, 4), dtype=torch.long)
    try:
        lane(tokens)
    except ValueError as exc:
        assert "at least one user token" in str(exc)
    else:
        raise AssertionError("empty token input was accepted")


def test_out_of_vocabulary_user_tokens_are_rejected() -> None:
    lane = make_lane(13)
    for tokens in (
        torch.tensor([[-1]], dtype=torch.long),
        torch.tensor([[lane.vocabulary_size]], dtype=torch.long),
    ):
        try:
            lane(tokens)
        except ValueError as exc:
            assert "configured vocabulary" in str(exc)
        else:
            raise AssertionError("out-of-vocabulary token input was accepted")


def test_text_tokenization_is_deterministic_and_reserves_padding() -> None:
    first = encode_texts(
        ["Capability prefix is user text", ""],
        vocabulary_size=64,
        maximum_tokens=6,
    )
    second = encode_texts(
        ["Capability prefix is user text", ""],
        vocabulary_size=64,
        maximum_tokens=6,
    )
    assert torch.equal(first, second)
    assert first[0, 0].item() == token_id("capability", 64)
    assert first[1, 0].item() != 0


def test_canonical_artifact_satisfies_publication_contract() -> None:
    artifact_path = (
        Path(__file__).parents[1]
        / "results"
        / "capability_prefix_token_moe_20260831T182453_013039Z.json"
    )
    artifact = json.loads(artifact_path.read_text())
    evaluation = artifact["evaluation"]
    isolation = artifact["training_isolation"]
    runtime = artifact["runtime"]

    assert artifact["status"] == "pass"
    assert artifact["source"]["dirty"] is False
    assert artifact["git_revision"] == artifact["source"]["commit"]
    assert evaluation["macro_accuracy"] >= 0.72
    assert evaluation["examples"] == 6231
    assert evaluation["routed_tokens"] == 311477
    assert evaluation["prediction_differences"] == 0
    assert evaluation["exact_logit_mismatched_regimes"] == 0
    assert evaluation["maximum_logit_difference"] == 0.0
    assert evaluation["maximum_dense_probability_difference"] == 0.0
    assert evaluation["unauthorized_dispatches"] == 0
    assert evaluation["minimum_expert_utilization_fraction"] >= 0.10
    assert artifact["minimum_prefix_maximum_parameter_delta"] > 0.0
    assert artifact["minimum_router_maximum_parameter_delta"] > 0.0
    assert artifact["spoofing_control"]["unauthorized_dispatches"] == 0
    assert isolation["inactive_gradient_tensors"] == 0
    assert isolation["inactive_optimizer_states"] == 0
    assert isolation["inactive_maximum_parameter_delta"] == 0.0
    assert len(runtime["authorities"]) == 8
    assert [authority["authorized_regime_ids"] for authority in runtime["authorities"]] == [
        [regime] for regime in range(8)
    ]
    assert runtime["rejection_probe"]["all_rejected"] is True
    assert runtime["rejection_probe"]["unauthorized_model_calls"] == 0
    assert artifact["storage"]["packed_bank_parameter_bytes"] == artifact["storage"][
        "separate_lane_parameter_bytes"
    ]
