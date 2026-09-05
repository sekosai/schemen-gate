"""Small falsifiable model for an externally gated depth-attention head.

This is a mathematical research harness, not a replacement for Schemen Gate or
Substrate authority.  It isolates one question: if earlier Transformer block
deltas are the candidate values, can an external allow-set prevent an
unauthorized delta from being scored or transported while a learned scorer
chooses among the authorized candidates?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

Vector = tuple[float, ...]


class AuthorizationRefusal(ValueError):
    """Raised before scoring when a toy grant cannot authorize a request."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _seal(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _finite_vector(values: Iterable[float], *, name: str) -> Vector:
    result = tuple(float(value) for value in values)
    if not result or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must be a non-empty finite vector")
    return result


@dataclass(frozen=True)
class DeltaSource:
    """One exact block/sublayer write offered to the depth router."""

    source_id: str
    owner_regime: int
    delta: Vector

    def __post_init__(self) -> None:
        if not self.source_id or self.owner_regime < 0:
            raise ValueError("source identity and owner regime are required")
        object.__setattr__(
            self,
            "delta",
            _finite_vector(self.delta, name=f"delta {self.source_id}"),
        )

    @property
    def seal(self) -> str:
        return _seal(asdict(self))


@dataclass(frozen=True)
class StaticGrant:
    """Frozen allow-set for the toy; real deployments must use Gate authority."""

    grant_id: str
    regime_id: int
    allowed_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        allowed = tuple(self.allowed_source_ids)
        if (
            not self.grant_id
            or self.regime_id < 0
            or not allowed
            or len(set(allowed)) != len(allowed)
            or any(not source_id for source_id in allowed)
        ):
            raise ValueError("grant must name one regime and a non-empty unique allow-set")
        object.__setattr__(self, "allowed_source_ids", allowed)

    @property
    def seal(self) -> str:
        return _seal(asdict(self))


@dataclass(frozen=True)
class GatedDeltaHeadReceipt:
    """Replayable receipt for one authorized candidate-restricted depth read."""

    schema: str
    query_id: str
    grant_id: str
    grant_seal: str
    regime_id: int
    catalog_seal: str
    allowed_source_ids: tuple[str, ...]
    scored_source_ids: tuple[str, ...]
    source_seals: tuple[str, ...]
    logits: Vector
    weights: Vector
    current_residual: Vector
    routed_delta: Vector
    output: Vector
    seal: str

    def payload(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("seal")
        return value


class RecordingScorer:
    """Deterministic learned-score stand-in that records every source access."""

    def __init__(self, logits: Mapping[str, float]) -> None:
        self._logits = {key: float(value) for key, value in logits.items()}
        if not all(math.isfinite(value) for value in self._logits.values()):
            raise ValueError("all logits must be finite")
        self.accessed: list[str] = []

    def __call__(self, source: DeltaSource) -> float:
        self.accessed.append(source.source_id)
        try:
            return self._logits[source.source_id]
        except KeyError as error:
            raise ValueError(f"missing score for {source.source_id}") from error


def catalog_seal(sources: Sequence[DeltaSource]) -> str:
    """Seal source identities and ownership without reading excluded payloads."""

    ordered = tuple(sources)
    if not ordered or len({source.source_id for source in ordered}) != len(ordered):
        raise ValueError("catalog must contain uniquely identified sources")
    return _seal(tuple((source.source_id, source.owner_regime) for source in ordered))


def _softmax(logits: Sequence[float]) -> Vector:
    values = _finite_vector(logits, name="logits")
    maximum = max(values)
    exponentials = tuple(math.exp(value - maximum) for value in values)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def execute_authorized_delta_head(
    current_residual: Sequence[float],
    sources: Sequence[DeltaSource],
    grant: StaticGrant,
    scorer: Callable[[DeltaSource], float],
    *,
    query_id: str,
) -> GatedDeltaHeadReceipt:
    """Score and transport only candidates admitted by the external allow-set.

    Candidate restriction happens before the learned scorer.  This is stronger
    than multiplying already-computed logits by zero and then normalizing.
    """

    if not query_id:
        raise ValueError("query_id is required")
    current = _finite_vector(current_residual, name="current residual")
    ordered = tuple(sources)
    sealed_catalog = catalog_seal(ordered)
    by_id = {source.source_id: source for source in ordered}

    try:
        allowed = tuple(by_id[source_id] for source_id in grant.allowed_source_ids)
    except KeyError as error:
        raise AuthorizationRefusal("grant refers to an unknown source") from error
    if any(source.owner_regime != grant.regime_id for source in allowed):
        raise AuthorizationRefusal("grant crosses a declared source owner")
    if any(len(source.delta) != len(current) for source in allowed):
        raise ValueError("current residual and source deltas must align")

    logits = tuple(float(scorer(source)) for source in allowed)
    if not all(math.isfinite(value) for value in logits):
        raise ValueError("scorer returned a non-finite logit")
    weights = _softmax(logits)
    routed = tuple(
        sum(
            weight * source.delta[index]
            for weight, source in zip(weights, allowed, strict=True)
        )
        for index in range(len(current))
    )
    output = tuple(
        base + addition for base, addition in zip(current, routed, strict=True)
    )
    payload: dict[str, object] = {
        "schema": "toy-gated-delta-head-receipt-v1",
        "query_id": query_id,
        "grant_id": grant.grant_id,
        "grant_seal": grant.seal,
        "regime_id": grant.regime_id,
        "catalog_seal": sealed_catalog,
        "allowed_source_ids": grant.allowed_source_ids,
        "scored_source_ids": tuple(source.source_id for source in allowed),
        "source_seals": tuple(source.seal for source in allowed),
        "logits": logits,
        "weights": weights,
        "current_residual": current,
        "routed_delta": routed,
        "output": output,
    }
    return GatedDeltaHeadReceipt(**payload, seal=_seal(payload))


def verify_receipt(
    receipt: GatedDeltaHeadReceipt,
    sources: Sequence[DeltaSource],
    grant: StaticGrant,
) -> bool:
    """Reconstruct every claimed value and refuse any changed field."""

    ordered = tuple(sources)
    by_id = {source.source_id: source for source in ordered}
    if (
        receipt.schema != "toy-gated-delta-head-receipt-v1"
        or receipt.query_id == ""
        or receipt.grant_id != grant.grant_id
        or receipt.grant_seal != grant.seal
        or receipt.regime_id != grant.regime_id
        or receipt.catalog_seal != catalog_seal(ordered)
        or receipt.allowed_source_ids != grant.allowed_source_ids
        or receipt.scored_source_ids != grant.allowed_source_ids
        or len(receipt.logits) != len(grant.allowed_source_ids)
        or len(receipt.weights) != len(grant.allowed_source_ids)
        or not math.isclose(sum(receipt.weights), 1.0, rel_tol=0.0, abs_tol=1e-15)
        or any(weight <= 0.0 for weight in receipt.weights)
    ):
        return False
    try:
        allowed = tuple(by_id[source_id] for source_id in grant.allowed_source_ids)
    except KeyError:
        return False
    if tuple(source.seal for source in allowed) != receipt.source_seals:
        return False
    expected_weights = _softmax(receipt.logits)
    if expected_weights != receipt.weights:
        return False
    expected_routed = tuple(
        sum(
            weight * source.delta[index]
            for weight, source in zip(receipt.weights, allowed, strict=True)
        )
        for index in range(len(receipt.current_residual))
    )
    expected_output = tuple(
        base + addition
        for base, addition in zip(receipt.current_residual, expected_routed, strict=True)
    )
    return (
        expected_routed == receipt.routed_delta
        and expected_output == receipt.output
        and _seal(receipt.payload()) == receipt.seal
    )


def naive_zero_logit_softmax(
    logits: Sequence[float],
    allowed: Sequence[bool],
) -> Vector:
    """Deliberately broken control: zero logits, then softmax all candidates."""

    if len(logits) != len(allowed) or not any(allowed):
        raise ValueError("logits and a non-empty mask must align")
    return _softmax(
        tuple(
            float(logit) * float(is_allowed)
            for logit, is_allowed in zip(logits, allowed, strict=True)
        )
    )


def _max_abs_difference(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must align")
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def run_toy() -> dict[str, object]:
    sources = (
        DeltaSource("r0:block0:attention", 0, (1.0, 0.0, 0.25)),
        DeltaSource("r0:block0:ffn", 0, (0.0, 1.0, -0.25)),
        DeltaSource("r1:block0:attention", 1, (4.0, 4.0, 4.0)),
        DeltaSource("r1:block0:ffn", 1, (-4.0, -4.0, -4.0)),
    )
    grant = StaticGrant(
        "toy-grant-r0",
        0,
        ("r0:block0:attention", "r0:block0:ffn"),
    )
    logits = {
        "r0:block0:attention": -4.0,
        "r0:block0:ffn": -5.0,
        "r1:block0:attention": 4.0,
        "r1:block0:ffn": 3.0,
    }
    current = (0.25, -0.25, 0.5)
    scorer = RecordingScorer(logits)
    receipt = execute_authorized_delta_head(
        current,
        sources,
        grant,
        scorer,
        query_id="toy-query-001",
    )

    mutated_sources = (
        *sources[:2],
        DeltaSource("r1:block0:attention", 1, (1e12, -1e12, 1e12)),
        DeltaSource("r1:block0:ffn", 1, (-1e12, 1e12, -1e12)),
    )
    mutation_scorer = RecordingScorer(logits)
    mutated = execute_authorized_delta_head(
        current,
        mutated_sources,
        grant,
        mutation_scorer,
        query_id="toy-query-001-mutated-unauthorized",
    )

    naive_weights = naive_zero_logit_softmax(
        tuple(logits[source.source_id] for source in sources),
        tuple(source.owner_regime == grant.regime_id for source in sources),
    )
    unauthorized_naive_mass = sum(naive_weights[2:])

    bypass_current = (0.25, -0.25, 99.5)
    bypass_scorer = RecordingScorer(logits)
    bypass = execute_authorized_delta_head(
        bypass_current,
        sources,
        grant,
        bypass_scorer,
        query_id="toy-query-001-upstream-bypass",
    )

    refused_before_scoring = False
    refusal_scorer = RecordingScorer(logits)
    try:
        execute_authorized_delta_head(
            current,
            sources,
            StaticGrant("invalid-cross-owner", 0, ("r1:block0:attention",)),
            refusal_scorer,
            query_id="toy-query-refused",
        )
    except AuthorizationRefusal:
        refused_before_scoring = refusal_scorer.accessed == []

    empty_grant_refused_at_construction = False
    try:
        StaticGrant("invalid-empty", 0, ())
    except ValueError:
        empty_grant_refused_at_construction = True

    result = {
        "schema": "toy-gated-delta-head-result-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version()},
        "claim_scope": (
            "candidate-restricted depth-delta routing in a deterministic toy; "
            "not full Transformer isolation, semantic routing, or production authority"
        ),
        "source_count": len(sources),
        "authorized_source_count": len(grant.allowed_source_ids),
        "authorized_scored_source_ids": scorer.accessed,
        "unauthorized_sources_scored": sorted(
            set(scorer.accessed) - set(grant.allowed_source_ids)
        ),
        "receipt_verified": verify_receipt(receipt, sources, grant),
        "receipt": asdict(receipt),
        "controls": {
            "unauthorized_delta_mutation_max_abs_output_difference": (
                _max_abs_difference(receipt.output, mutated.output)
            ),
            "unauthorized_delta_mutation_scored_source_ids": mutation_scorer.accessed,
            "naive_zero_logit_unauthorized_probability_mass": unauthorized_naive_mass,
            "naive_zero_logit_weights": naive_weights,
            "upstream_shared_residual_mutation_max_abs_output_difference": (
                _max_abs_difference(receipt.output, bypass.output)
            ),
            "upstream_shared_residual_scored_source_ids": bypass_scorer.accessed,
            "cross_owner_grant_refused_before_scoring": refused_before_scoring,
            "empty_grant_refused_at_construction": empty_grant_refused_at_construction,
        },
    }
    controls = result["controls"]
    assert isinstance(controls, dict)
    result["acceptance"] = {
        "status": "PASS" if (
            result["unauthorized_sources_scored"] == []
            and result["receipt_verified"] is True
            and controls["unauthorized_delta_mutation_max_abs_output_difference"] == 0.0
            and controls["naive_zero_logit_unauthorized_probability_mass"] > 0.0
            and controls["upstream_shared_residual_mutation_max_abs_output_difference"] > 0.0
            and controls["cross_owner_grant_refused_before_scoring"] is True
            and controls["empty_grant_refused_at_construction"] is True
        ) else "FAIL",
        "meaning": (
            "The positive local gate and both required negative controls behaved "
            "as preregistered. PASS does not promote a model-scale claim."
        ),
    }
    result["result_seal"] = _seal(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run_toy()
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["acceptance"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
