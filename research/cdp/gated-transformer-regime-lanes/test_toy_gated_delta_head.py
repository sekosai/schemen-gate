from __future__ import annotations

import unittest
from dataclasses import replace

from toy_gated_delta_head import (
    AuthorizationRefusal,
    DeltaSource,
    RecordingScorer,
    StaticGrant,
    execute_authorized_delta_head,
    naive_zero_logit_softmax,
    run_toy,
    verify_receipt,
)


class GatedDeltaHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = (
            DeltaSource("r0:a", 0, (1.0, 0.0)),
            DeltaSource("r0:b", 0, (0.0, 1.0)),
            DeltaSource("r1:a", 1, (10.0, 10.0)),
        )
        self.grant = StaticGrant("g0", 0, ("r0:a", "r0:b"))
        self.logits = {"r0:a": -2.0, "r0:b": -3.0, "r1:a": 9.0}

    def execute(self):
        scorer = RecordingScorer(self.logits)
        receipt = execute_authorized_delta_head(
            (0.5, -0.5),
            self.sources,
            self.grant,
            scorer,
            query_id="q",
        )
        return scorer, receipt

    def test_candidate_restriction_precedes_scoring(self) -> None:
        scorer, receipt = self.execute()
        self.assertEqual(scorer.accessed, ["r0:a", "r0:b"])
        self.assertEqual(receipt.scored_source_ids, self.grant.allowed_source_ids)
        self.assertTrue(verify_receipt(receipt, self.sources, self.grant))

    def test_unauthorized_delta_mutation_cannot_change_output(self) -> None:
        _, receipt = self.execute()
        mutated = (*self.sources[:2], DeltaSource("r1:a", 1, (1e15, -1e15)))
        scorer = RecordingScorer(self.logits)
        changed = execute_authorized_delta_head(
            (0.5, -0.5), mutated, self.grant, scorer, query_id="q-mutated"
        )
        self.assertEqual(receipt.output, changed.output)
        self.assertNotIn("r1:a", scorer.accessed)

    def test_zeroing_logits_before_softmax_is_not_authorization(self) -> None:
        weights = naive_zero_logit_softmax((-2.0, -3.0, 9.0), (True, True, False))
        self.assertGreater(weights[2], 0.0)

    def test_cross_owner_grant_refuses_before_scoring(self) -> None:
        scorer = RecordingScorer(self.logits)
        with self.assertRaises(AuthorizationRefusal):
            execute_authorized_delta_head(
                (0.5, -0.5),
                self.sources,
                StaticGrant("bad", 0, ("r1:a",)),
                scorer,
                query_id="q-bad",
            )
        self.assertEqual(scorer.accessed, [])

    def test_empty_grant_is_refused_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            StaticGrant("empty", 0, ())

    def test_receipt_mutation_is_rejected(self) -> None:
        _, receipt = self.execute()
        self.assertFalse(
            verify_receipt(replace(receipt, output=(999.0, 999.0)), self.sources, self.grant)
        )

    def test_upstream_residual_is_explicitly_outside_the_local_gate(self) -> None:
        scorer_a = RecordingScorer(self.logits)
        scorer_b = RecordingScorer(self.logits)
        first = execute_authorized_delta_head(
            (0.5, -0.5), self.sources, self.grant, scorer_a, query_id="q-a"
        )
        second = execute_authorized_delta_head(
            (0.5, 99.5), self.sources, self.grant, scorer_b, query_id="q-b"
        )
        self.assertNotEqual(first.output, second.output)

    def test_preregistered_toy_accepts(self) -> None:
        result = run_toy()
        self.assertEqual(result["acceptance"]["status"], "PASS")
        self.assertTrue(result["controls"]["empty_grant_refused_at_construction"])


if __name__ == "__main__":
    unittest.main()
