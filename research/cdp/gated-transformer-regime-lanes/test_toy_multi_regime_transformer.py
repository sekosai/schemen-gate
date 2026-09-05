from __future__ import annotations

import math
import unittest
from dataclasses import asdict

from toy_multi_regime_transformer import (
    GateRefusal,
    Grant,
    LaneState,
    LaneStateStore,
    MultiRegimeTransformer,
    SchemenGate,
    SharedStatelessWeights,
    attention_scores,
    build_model,
    example_states,
)


class MultiRegimeTransformerTests(unittest.TestCase):
    def grant(self, *regimes: str, allow_join: bool = False) -> Grant:
        return Grant(
            owner="researcher",
            model_id="toy-transformer-v1",
            regimes=tuple(regimes),
            allow_join=allow_join,
        )

    def assertVectorAlmostEqual(self, left, right) -> None:  # noqa: N802
        self.assertEqual(len(left), len(right))
        for left_value, right_value in zip(left, right, strict=True):
            self.assertAlmostEqual(left_value, right_value, places=12)

    def test_union_execution_equals_separate_execution_per_lane(self) -> None:
        union = build_model().execute(self.grant("research", "public"))
        outcomes = {item.regime: item for item in union.outcomes}
        for regime in ("public", "research"):
            separate = build_model().execute(self.grant(regime)).outcomes[0]
            self.assertVectorAlmostEqual(
                outcomes[regime].residual_after_block,
                separate.residual_after_block,
            )

    def test_unauthorized_lane_mutation_cannot_change_authorized_output(self) -> None:
        ordinary = build_model().execute(self.grant("public")).outcomes[0]
        states = example_states()
        private = states["private"]
        states["private"] = LaneState(
            **{
                **asdict(private),
                "residual": (-1.0e300, 1.0e300),
                "values": ((1.0e300, 1.0e300),),
            }
        )
        mutated = build_model(states).execute(self.grant("public")).outcomes[0]
        self.assertEqual(ordinary, mutated)

    def test_only_authorized_lanes_are_read_and_called(self) -> None:
        model = build_model()
        receipt = model.execute(self.grant("public", "research"))
        self.assertEqual(receipt.state_reads, ("public", "research"))
        self.assertEqual(receipt.attention_calls, receipt.state_reads)
        self.assertEqual(receipt.ffn_calls, receipt.state_reads)
        self.assertNotIn("private", receipt.state_reads)

    def test_invalid_grants_refuse_before_any_state_read(self) -> None:
        invalid = (
            Grant("researcher", "toy-transformer-v1", (), signed=True),
            Grant("other", "toy-transformer-v1", ("public",), signed=True),
            Grant("researcher", "wrong", ("public",), signed=True),
            Grant("researcher", "toy-transformer-v1", ("public",), signed=False),
            Grant("researcher", "toy-transformer-v1", ("missing",), signed=True),
        )
        for grant in invalid:
            states = example_states()
            store = LaneStateStore(states)
            model = MultiRegimeTransformer(
                gate=SchemenGate(
                    owner="researcher",
                    model_id="toy-transformer-v1",
                    registered=states,
                ),
                store=store,
            )
            with self.assertRaises(GateRefusal):
                model.execute(grant)
            self.assertEqual(store.read_log, [])

    def test_common_rope_offset_cancels_from_attention_scores(self) -> None:
        state = example_states()["public"]
        shifted = LaneState(**{**asdict(state), "common_phase": 1.234567})
        query = SharedStatelessWeights().query_of(state.residual)
        self.assertVectorAlmostEqual(
            attention_scores(state, query), attention_scores(shifted, query)
        )

    def test_capability_relative_phase_leaks_and_has_periodic_collisions(self) -> None:
        state = example_states()["public"]
        query = SharedStatelessWeights().query_of(state.residual)
        wrong_bank = attention_scores(
            state,
            query,
            query_capability_phase=0.0,
            bank_capability_phase=0.7,
        )
        self.assertTrue(any(abs(score) > 1e-12 for score in wrong_bank))
        self.assertVectorAlmostEqual(
            attention_scores(state, query, query_capability_phase=2.0 * math.pi),
            attention_scores(state, query),
        )

    def test_join_is_a_separate_authorized_operation(self) -> None:
        model = build_model()
        denied = self.grant("public", "research")
        receipt = model.execute(denied)
        with self.assertRaises(GateRefusal):
            model.join(denied, receipt)

        allowed = self.grant("public", "research", allow_join=True)
        allowed_receipt = model.execute(allowed)
        joined = model.join(allowed, allowed_receipt)
        expected = tuple(
            sum(outcome.residual_after_block[index] for outcome in allowed_receipt.outcomes)
            / len(allowed_receipt.outcomes)
            for index in range(2)
        )
        self.assertVectorAlmostEqual(joined, expected)

    def test_shared_weight_object_does_not_merge_lane_state(self) -> None:
        states = example_states()
        weights = SharedStatelessWeights()
        model = MultiRegimeTransformer(
            gate=SchemenGate(
                owner="researcher",
                model_id="toy-transformer-v1",
                registered=states,
            ),
            store=LaneStateStore(states),
            weights=weights,
        )
        receipt = model.execute(self.grant("public", "research"))
        self.assertEqual(receipt.ffn_calls, ("public", "research"))
        self.assertNotEqual(
            receipt.outcomes[0].residual_after_block,
            receipt.outcomes[1].residual_after_block,
        )


if __name__ == "__main__":
    unittest.main()
