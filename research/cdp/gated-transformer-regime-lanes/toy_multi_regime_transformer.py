"""Small falsifiable model of Schemen-gated complete Transformer regime lanes.

The toy deliberately models topology and authority, not language quality.  A
grant is resolved before the state store is touched.  Every authorized regime
then receives a complete attention calculation and an independent residual
update.  The FFN weights are shared and stateless; lane state is not.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

Vector = tuple[float, float]


class GateRefusal(RuntimeError):
    """Raised before state access when a grant is invalid."""


def add(left: Vector, right: Vector) -> Vector:
    return (left[0] + right[0], left[1] + right[1])


def matvec(matrix: tuple[Vector, Vector], value: Vector) -> Vector:
    return (
        matrix[0][0] * value[0] + matrix[0][1] * value[1],
        matrix[1][0] * value[0] + matrix[1][1] * value[1],
    )


def dot(left: Vector, right: Vector) -> float:
    return left[0] * right[0] + left[1] * right[1]


def rotate(value: Vector, angle: float) -> Vector:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        cosine * value[0] - sine * value[1],
        sine * value[0] + cosine * value[1],
    )


def stable_softmax(values: Sequence[float]) -> tuple[float, ...]:
    peak = max(values)
    exponentials = tuple(math.exp(value - peak) for value in values)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


@dataclass(frozen=True)
class Grant:
    owner: str
    model_id: str
    regimes: tuple[str, ...]
    signed: bool = True
    allow_join: bool = False


@dataclass(frozen=True)
class LaneState:
    residual: Vector
    keys: tuple[Vector, ...]
    values: tuple[Vector, ...]
    key_positions: tuple[int, ...]
    position: int
    rope_frequency: float
    common_phase: float = 0.0

    def __post_init__(self) -> None:
        if not self.keys or len(self.keys) != len(self.values):
            raise ValueError("keys and values must be nonempty and equal length")
        if len(self.keys) != len(self.key_positions):
            raise ValueError("each key requires a position")


@dataclass(frozen=True)
class LaneOutcome:
    regime: str
    attention: Vector
    residual_after_attention: Vector
    ffn: Vector
    residual_after_block: Vector


@dataclass(frozen=True)
class ExecutionReceipt:
    owner: str
    model_id: str
    authorized_regimes: tuple[str, ...]
    state_reads: tuple[str, ...]
    attention_calls: tuple[str, ...]
    ffn_calls: tuple[str, ...]
    outcomes: tuple[LaneOutcome, ...]
    receipt_seal: str


class SchemenGate:
    """Fail-closed registry check with no reference to lane state."""

    def __init__(self, *, owner: str, model_id: str, registered: Iterable[str]):
        self._owner = owner
        self._model_id = model_id
        self._registered = frozenset(registered)

    def resolve(self, grant: Grant) -> tuple[str, ...]:
        if not grant.signed:
            raise GateRefusal("grant is unsigned")
        if grant.owner != self._owner:
            raise GateRefusal("owner mismatch")
        if grant.model_id != self._model_id:
            raise GateRefusal("model mismatch")
        if not grant.regimes:
            raise GateRefusal("empty regime set")
        if len(set(grant.regimes)) != len(grant.regimes):
            raise GateRefusal("duplicate regime")
        unknown = set(grant.regimes) - self._registered
        if unknown:
            raise GateRefusal(f"unregistered regime: {sorted(unknown)}")
        return tuple(sorted(grant.regimes))


class LaneStateStore:
    """Instrumented state boundary used to prove pre-access authorization."""

    def __init__(self, states: Mapping[str, LaneState]):
        self._states = dict(states)
        self.read_log: list[str] = []

    def read(self, regime: str) -> LaneState:
        self.read_log.append(regime)
        return self._states[regime]


@dataclass(frozen=True)
class SharedStatelessWeights:
    query: tuple[Vector, Vector] = ((1.0, 0.2), (-0.1, 0.9))
    ffn: tuple[Vector, Vector] = ((0.4, -0.3), (0.2, 0.5))

    def query_of(self, residual: Vector) -> Vector:
        return matvec(self.query, residual)

    def ffn_of(self, value: Vector) -> Vector:
        hidden = matvec(self.ffn, value)
        return (max(0.0, hidden[0]), max(0.0, hidden[1]))


def attention_scores(
    state: LaneState,
    query: Vector,
    *,
    query_capability_phase: float = 0.0,
    bank_capability_phase: float = 0.0,
) -> tuple[float, ...]:
    """Return full-width RoPE scores for one complete lane.

    ``common_phase`` is applied to both Q and K and therefore cancels.  The two
    capability phases are deliberately distinct to expose focusing and leak.
    """

    q_angle = (
        state.position * state.rope_frequency
        + state.common_phase
        + query_capability_phase
    )
    rotated_query = rotate(query, q_angle)
    return tuple(
        dot(
            rotated_query,
            rotate(
                key,
                key_position * state.rope_frequency
                + state.common_phase
                + bank_capability_phase,
            ),
        )
        for key, key_position in zip(state.keys, state.key_positions, strict=True)
    )


def attend(state: LaneState, query: Vector) -> Vector:
    weights = stable_softmax(attention_scores(state, query))
    return (
        sum(weight * value[0] for weight, value in zip(weights, state.values, strict=True)),
        sum(weight * value[1] for weight, value in zip(weights, state.values, strict=True)),
    )


class MultiRegimeTransformer:
    def __init__(
        self,
        *,
        gate: SchemenGate,
        store: LaneStateStore,
        weights: SharedStatelessWeights | None = None,
    ):
        self._gate = gate
        self._store = store
        self._weights = weights or SharedStatelessWeights()

    def execute(self, grant: Grant) -> ExecutionReceipt:
        authorized = self._gate.resolve(grant)
        read_start = len(self._store.read_log)
        attention_calls: list[str] = []
        ffn_calls: list[str] = []
        outcomes: list[LaneOutcome] = []

        for regime in authorized:
            state = self._store.read(regime)
            query = self._weights.query_of(state.residual)
            attention = attend(state, query)
            attention_calls.append(regime)
            after_attention = add(state.residual, attention)
            ffn = self._weights.ffn_of(after_attention)
            ffn_calls.append(regime)
            outcomes.append(
                LaneOutcome(
                    regime=regime,
                    attention=attention,
                    residual_after_attention=after_attention,
                    ffn=ffn,
                    residual_after_block=add(after_attention, ffn),
                )
            )

        unsigned = {
            "owner": grant.owner,
            "model_id": grant.model_id,
            "authorized_regimes": authorized,
            "state_reads": tuple(self._store.read_log[read_start:]),
            "attention_calls": tuple(attention_calls),
            "ffn_calls": tuple(ffn_calls),
            "outcomes": [asdict(outcome) for outcome in outcomes],
        }
        seal = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return ExecutionReceipt(
            owner=grant.owner,
            model_id=grant.model_id,
            authorized_regimes=authorized,
            state_reads=tuple(self._store.read_log[read_start:]),
            attention_calls=tuple(attention_calls),
            ffn_calls=tuple(ffn_calls),
            outcomes=tuple(outcomes),
            receipt_seal=seal,
        )

    def join(self, grant: Grant, receipt: ExecutionReceipt) -> Vector:
        authorized = self._gate.resolve(grant)
        if not grant.allow_join:
            raise GateRefusal("join is not authorized")
        if authorized != receipt.authorized_regimes:
            raise GateRefusal("join receipt does not match grant")
        count = len(receipt.outcomes)
        return (
            sum(item.residual_after_block[0] for item in receipt.outcomes) / count,
            sum(item.residual_after_block[1] for item in receipt.outcomes) / count,
        )


def example_states() -> dict[str, LaneState]:
    return {
        "public": LaneState(
            residual=(0.8, -0.2),
            keys=((1.0, 0.0), (0.2, 0.9)),
            values=((0.4, 0.1), (-0.2, 0.7)),
            key_positions=(0, 2),
            position=3,
            rope_frequency=0.35,
        ),
        "research": LaneState(
            residual=(-0.1, 0.9),
            keys=((0.0, 1.0), (0.8, -0.1)),
            values=((0.1, 0.8), (0.9, -0.3)),
            key_positions=(1, 4),
            position=5,
            rope_frequency=0.22,
            common_phase=0.4,
        ),
        "private": LaneState(
            residual=(9.0, 9.0),
            keys=((9.0, 9.0),),
            values=((99.0, -99.0),),
            key_positions=(0,),
            position=1,
            rope_frequency=0.5,
        ),
    }


def build_model(states: Mapping[str, LaneState] | None = None) -> MultiRegimeTransformer:
    state_map = dict(states or example_states())
    return MultiRegimeTransformer(
        gate=SchemenGate(
            owner="researcher",
            model_id="toy-transformer-v1",
            registered=state_map,
        ),
        store=LaneStateStore(state_map),
    )


def _close(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12) for a, b in zip(left, right, strict=True))


def self_check_result() -> dict[str, object]:
    checks: dict[str, bool] = {}

    one = build_model().execute(
        Grant("researcher", "toy-transformer-v1", ("public",))
    )
    union = build_model().execute(
        Grant("researcher", "toy-transformer-v1", ("research", "public"))
    )
    public_alone = one.outcomes[0].residual_after_block
    public_in_union = next(
        outcome.residual_after_block
        for outcome in union.outcomes
        if outcome.regime == "public"
    )
    checks["union_equals_separate_per_lane"] = _close(public_alone, public_in_union)
    checks["one_complete_attention_call_per_lane"] = (
        union.attention_calls == ("public", "research")
    )
    checks["one_shared_ffn_invocation_per_lane"] = union.ffn_calls == (
        "public",
        "research",
    )
    checks["unauthorized_private_lane_not_read"] = "private" not in union.state_reads

    baseline_state = example_states()["public"]
    shifted_state = LaneState(**{**asdict(baseline_state), "common_phase": 1.234})
    query = SharedStatelessWeights().query_of(baseline_state.residual)
    checks["common_rope_phase_cancels"] = _close(
        attention_scores(baseline_state, query), attention_scores(shifted_state, query)
    )
    mismatch = attention_scores(
        baseline_state,
        query,
        query_capability_phase=0.0,
        bank_capability_phase=0.7,
    )
    checks["capability_phase_is_not_hard_isolation"] = any(
        not math.isclose(value, 0.0, abs_tol=1e-12) for value in mismatch
    )
    checks["capability_phase_is_periodic"] = _close(
        attention_scores(
            baseline_state, query, query_capability_phase=2.0 * math.pi
        ),
        attention_scores(baseline_state, query),
    )

    refusal_model = build_model()
    try:
        refusal_model.execute(
            Grant("researcher", "wrong-model", ("public",), signed=True)
        )
        refused_without_read = False
    except GateRefusal:
        refused_without_read = refusal_model._store.read_log == []
    checks["invalid_grant_refused_before_state_read"] = refused_without_read

    join_model = build_model()
    join_grant = Grant(
        "researcher", "toy-transformer-v1", ("public", "research"), allow_join=True
    )
    join_receipt = join_model.execute(join_grant)
    joined = join_model.join(join_grant, join_receipt)
    checks["join_requires_and_accepts_explicit_authority"] = len(joined) == 2

    unsigned = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "topology": "gate_before_state_access_complete_regime_lanes_shared_stateless_ffn",
        "scope": "deterministic topology toy; no learned language-model claim",
    }
    result_seal = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {**unsigned, "result_seal": result_seal}


def main() -> None:
    output = Path(__file__).parent / "results" / "toy_multi_regime_transformer_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = self_check_result()
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
