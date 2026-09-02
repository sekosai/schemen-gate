"""Reproduce the formative capacity-preserving wide-classifier experiment.

This is a publication-safe reimplementation of the repository predecessor's
``wide_regime`` proof of concept.  It intentionally contains no production
key custody, lockbox, token, IdP, or serving-runtime code.

The experiment holds per-regime hidden capacity constant at ``q`` dimensions
and scales the hidden width as ``d = q * R``.  It therefore asks a different
question from a fixed-width R-sweep: can one gated, block-sparse classifier
retain a fixed amount of capacity for every regime as the number of regimes
grows?  The synthetic task has two observation/answer pairs per regime and is
a deterministic memorization/scalability check, not a held-out utility study.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from execution_preflight import GateExecutionPreflight
from library_provenance import collect_library_provenance, collect_source_provenance

from schemen_gate import GateMask


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class WideModel:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def initialize_model(
    *,
    observations: int,
    hidden: int,
    outputs: int,
    seed: int,
) -> WideModel:
    rng = np.random.RandomState(seed)
    w1_scale = np.sqrt(2.0 / (observations + hidden))
    w2_scale = np.sqrt(2.0 / (hidden + outputs))
    return WideModel(
        w1=rng.randn(observations, hidden) * w1_scale,
        b1=np.zeros(hidden, dtype=np.float64),
        w2=rng.randn(hidden, outputs) * w2_scale,
        b2=np.zeros(outputs, dtype=np.float64),
    )


def train(
    model: WideModel,
    supports: np.ndarray,
    *,
    observations_per_regime: int,
    epochs: int,
    learning_rate: float,
) -> tuple[float, float]:
    """Train the same gated MLP as the formative PoC, exploiting sparsity.

    Each synthetic input is one-hot, so only one row of ``w1`` is used.  The
    binary gate then retains only the owning regime's support.  Computing that
    support directly is algebraically equivalent to materializing the full
    hidden vector and multiplying by a binary mask, and avoids an otherwise
    wasteful dense matrix multiply during each epoch.
    """

    regimes, width = supports.shape
    observations = regimes * observations_per_regime
    sample_indices = np.arange(observations)
    sample_regimes = np.repeat(np.arange(regimes), observations_per_regime)
    sample_supports = supports[sample_regimes]
    final_loss = float("nan")
    initial_loss = float("nan")

    for epoch in range(epochs):
        z1 = (
            model.w1[sample_indices[:, None], sample_supports]
            + model.b1[sample_supports]
        )
        hidden = np.maximum(0.0, z1)
        selected_w2 = model.w2[sample_supports, :]
        logits = np.einsum("nq,nqo->no", hidden, selected_w2) + model.b2
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        losses = -np.log(probabilities[sample_indices, sample_indices] + 1e-12)
        final_loss = float(losses.mean())
        if epoch == 0:
            initial_loss = final_loss

        d_logits = probabilities
        d_logits[sample_indices, sample_indices] -= 1.0
        d_logits /= observations

        d_hidden = np.einsum("no,nqo->nq", d_logits, selected_w2)
        d_z1 = d_hidden * (z1 > 0.0)

        # All gradients above were computed from the pre-update state.  The
        # loop only scatters each regime's q-row block back into dense storage.
        for regime in range(regimes):
            start = regime * observations_per_regime
            stop = start + observations_per_regime
            rows = slice(start, stop)
            support = supports[regime]
            model.w2[support, :] -= learning_rate * (
                hidden[rows].T @ d_logits[rows]
            )
            model.b1[support] -= learning_rate * d_z1[rows].sum(axis=0)

        model.w1[sample_indices[:, None], sample_supports] -= (
            learning_rate * d_z1
        )
        model.b2 -= learning_rate * d_logits.sum(axis=0)

    return initial_loss, final_loss


def selected_logits(
    model: WideModel,
    supports: np.ndarray,
    observation: int,
    regime: int,
) -> np.ndarray:
    support = supports[regime]
    hidden = np.maximum(0.0, model.w1[observation, support] + model.b1[support])
    return hidden @ model.w2[support, :] + model.b2


def dense_gated_logits(
    model: WideModel,
    supports: np.ndarray,
    observation: int,
    regime: int,
) -> np.ndarray:
    hidden = np.maximum(0.0, model.w1[observation] + model.b1)
    gate = np.zeros_like(hidden)
    gate[supports[regime]] = 1.0
    return (hidden * gate) @ model.w2 + model.b2


def evaluate(
    model: WideModel,
    supports: np.ndarray,
    *,
    observations_per_regime: int,
    runtime: GateExecutionPreflight,
) -> dict:
    regimes = supports.shape[0]
    observations = regimes * observations_per_regime

    owning_correct = 0
    wrong_key_correct = 0
    wrong_key_probes = 0
    extraction_max_difference = 0.0

    for regime in range(regimes):
        def evaluate_authorized_regime(
            authorized_regime: int = regime,
        ) -> tuple[int, int, int, float]:
            local_owning = 0
            local_wrong = 0
            local_wrong_probes = 0
            local_extraction_difference = 0.0
            for observation in range(observations):
                owner = observation // observations_per_regime
                direct = selected_logits(model, supports, observation, authorized_regime)
                if authorized_regime == owner:
                    dense = dense_gated_logits(
                        model,
                        supports,
                        observation,
                        authorized_regime,
                    )
                    local_extraction_difference = max(
                        local_extraction_difference,
                        float(np.max(np.abs(direct - dense))),
                    )
                    local_owning += int(np.argmax(direct) == observation)
                else:
                    local_wrong_probes += 1
                    local_wrong += int(np.argmax(direct) == observation)
            return (
                local_owning,
                local_wrong,
                local_wrong_probes,
                local_extraction_difference,
            )

        regime_result = runtime.invoke(regime, evaluate_authorized_regime)
        owning_correct += regime_result[0]
        wrong_key_correct += regime_result[1]
        wrong_key_probes += regime_result[2]
        extraction_max_difference = max(
            extraction_max_difference,
            regime_result[3],
        )

    full_hidden = np.maximum(0.0, model.w1 + model.b1)
    full_predictions = np.argmax(full_hidden @ model.w2 + model.b2, axis=1)
    full_correct = int(np.sum(full_predictions == np.arange(observations)))

    return {
        "owning_correct": owning_correct,
        "owning_total": observations,
        "owning_accuracy": owning_correct / observations,
        "wrong_key_correct": wrong_key_correct,
        "wrong_key_probes": wrong_key_probes,
        "wrong_key_empirical_rate": (
            wrong_key_correct / wrong_key_probes if wrong_key_probes else None
        ),
        "ungated_control_correct": full_correct,
        "ungated_control_total": observations,
        "ungated_control_accuracy": full_correct / observations,
        "selected_vs_dense_max_absolute_difference": extraction_max_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", type=int, default=128)
    parser.add_argument("--observations-per-regime", type=int, default=2)
    parser.add_argument("--dimensions-per-regime", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-seed", type=int, default=128042)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if min(
        args.regimes,
        args.observations_per_regime,
        args.dimensions_per_regime,
        args.epochs,
    ) <= 0:
        raise ValueError("regimes, observations, dimensions, and epochs must be positive")

    observations = args.regimes * args.observations_per_regime
    hidden = args.regimes * args.dimensions_per_regime
    mask_key = __import__("hashlib").sha256(
        f"cdp-wide-classifier:{args.mask_seed}".encode()
    ).digest()
    supports = np.asarray(
        [
            np.flatnonzero(
                GateMask.derive(
                    mask_key,
                    regime_id,
                    n_dims=hidden,
                    n_regimes=args.regimes,
                ).to_numpy()
            )
            for regime_id in range(args.regimes)
        ],
        dtype=np.int64,
    )
    model = initialize_model(
        observations=observations,
        hidden=hidden,
        outputs=observations,
        seed=args.seed,
    )
    runtime = GateExecutionPreflight(
        model_id="capacity-preserving-wide-classifier",
        dimensions=hidden,
        authorized_regime_ids=list(range(args.regimes)),
    )

    started = time.time()
    initial_loss, final_loss = train(
        model,
        supports,
        observations_per_regime=args.observations_per_regime,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
    )
    summary = evaluate(
        model,
        supports,
        observations_per_regime=args.observations_per_regime,
        runtime=runtime,
    )
    summary.update(
        {
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "hidden_dimensions": hidden,
            "dimensions_per_regime": args.dimensions_per_regime,
        }
    )

    passed = (
        summary["owning_correct"] == summary["owning_total"]
        and summary["wrong_key_correct"] == 0
        and summary["selected_vs_dense_max_absolute_difference"] <= 1e-12
    )
    runtime_rejections = runtime.rejection_probe()
    passed = (
        passed
        and runtime_rejections["all_rejected"]
        and runtime_rejections["unauthorized_model_calls"] == 0
    )
    artifact = {
        "schema_version": 3,
        "experiment": "capacity_preserving_wide_classifier",
        "status": "pass" if passed else "failure",
        "scope": (
            "Synthetic memorization and capacity-scaling check. This is not a "
            "held-out generalization, privacy, or fixed-width parity result."
        ),
        "verdict_basis": (
            "Owning gated accuracy, wrong-key rejection, selected-vs-dense "
            "equivalence, and execution authorization. The ungated control is "
            "diagnostic only because it is not a serving path."
        ),
        "started_at_unix": started,
        "elapsed_seconds": time.time() - started,
        "git_revision": git_revision(),
        "source": collect_source_provenance(),
        "runtime": {
            **runtime.evidence(),
            "rejection_probe": runtime_rejections,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "libraries": collect_library_provenance(),
        },
        "mask_contract": {
            "implementation": "schemen_gate.GateMask.derive",
            "key_derivation": "SHA-256('cdp-wide-classifier:' + mask_seed)",
            "n_dims": hidden,
            "n_regimes": args.regimes,
        },
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir) if args.output_dir else None,
        },
        "summary": summary,
    }

    output_dir = args.output_dir or (Path(__file__).resolve().parent / "results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"capacity_preserving_wide_{timestamp}.json"
    output_path.write_text(json.dumps(artifact, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Results saved to {output_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
