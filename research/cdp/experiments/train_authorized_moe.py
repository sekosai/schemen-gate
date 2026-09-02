"""Train learned MoE routers inside fail-closed authorized expert sets.

The publication run uses eight disjoint binary tasks from the standard
20 Newsgroups train/test split.  Each task trains a conventional top-1 router
over two private experts.  The trained lanes are then copied unchanged into
one packed bank and evaluated through the source-visible research preflight.

Examples:

    python experiments/train_authorized_moe.py --synthetic --epochs 8
    python experiments/train_authorized_moe.py --epochs 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from authorized_moe import (
    AuthorizedExpertBank,
    LearnedTop1MoE,
    maximum_parameter_delta,
    snapshot,
    unsafe_zero_logit_route,
)
from execution_preflight import GateExecutionPreflight
from library_provenance import collect_library_provenance, collect_source_provenance

CATEGORY_PAIRS = (
    ("alt.atheism", "soc.religion.christian"),
    ("comp.graphics", "comp.windows.x"),
    ("comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"),
    ("comp.sys.mac.hardware", "misc.forsale"),
    ("rec.autos", "rec.motorcycles"),
    ("rec.sport.baseball", "rec.sport.hockey"),
    ("sci.crypt", "sci.electronics"),
    ("sci.med", "sci.space"),
)


@dataclass(frozen=True)
class RegimeDataset:
    name: str
    train_inputs: torch.Tensor
    train_labels: torch.Tensor
    test_inputs: torch.Tensor
    test_labels: torch.Tensor


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synthetic_datasets(
    *,
    regimes: int,
    dimensions: int,
    train_examples: int,
    test_examples: int,
    seed: int,
) -> list[RegimeDataset]:
    datasets = []
    for regime in range(regimes):
        rng = np.random.default_rng(seed + regime)
        direction = rng.normal(size=dimensions)
        direction /= np.linalg.norm(direction)

        def split(
            examples: int,
            *,
            local_rng: np.random.Generator = rng,
            local_direction: np.ndarray = direction,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            labels = np.arange(examples, dtype=np.int64) % 2
            noise = local_rng.normal(scale=0.55, size=(examples, dimensions))
            inputs = noise + (2 * labels[:, None] - 1) * local_direction * 1.5
            return (
                torch.from_numpy(inputs.astype(np.float32)),
                torch.from_numpy(labels),
            )

        train_inputs, train_labels = split(train_examples)
        test_inputs, test_labels = split(test_examples)
        datasets.append(
            RegimeDataset(
                name=f"synthetic-{regime}",
                train_inputs=train_inputs,
                train_labels=train_labels,
                test_inputs=test_inputs,
                test_labels=test_labels,
            )
        )
    return datasets


def newsgroups_datasets(
    *,
    dimensions: int,
    seed: int,
) -> tuple[list[RegimeDataset], dict[str, Any]]:
    from sklearn import __version__ as sklearn_version
    from sklearn.datasets import fetch_20newsgroups, get_data_home
    from sklearn.feature_extraction.text import HashingVectorizer

    categories = sorted({name for pair in CATEGORY_PAIRS for name in pair})
    fetch_arguments = {
        "categories": categories,
        "remove": ("headers", "footers", "quotes"),
        "shuffle": True,
        "random_state": seed,
    }
    train = fetch_20newsgroups(subset="train", **fetch_arguments)
    test = fetch_20newsgroups(subset="test", **fetch_arguments)
    vectorizer = HashingVectorizer(
        n_features=dimensions,
        alternate_sign=False,
        norm="l2",
        stop_words="english",
    )
    train_features = vectorizer.transform(train.data)
    test_features = vectorizer.transform(test.data)
    train_names = np.asarray([train.target_names[index] for index in train.target])
    test_names = np.asarray([test.target_names[index] for index in test.target])

    datasets = []
    for first, second in CATEGORY_PAIRS:
        train_mask = np.logical_or(train_names == first, train_names == second)
        test_mask = np.logical_or(test_names == first, test_names == second)
        datasets.append(
            RegimeDataset(
                name=f"{first}__vs__{second}",
                train_inputs=torch.from_numpy(
                    train_features[train_mask].toarray().astype(np.float32)
                ),
                train_labels=torch.from_numpy(
                    (train_names[train_mask] == second).astype(np.int64)
                ),
                test_inputs=torch.from_numpy(
                    test_features[test_mask].toarray().astype(np.float32)
                ),
                test_labels=torch.from_numpy(
                    (test_names[test_mask] == second).astype(np.int64)
                ),
            )
        )

    archive = Path(get_data_home()) / "20news-bydate_py3.pkz"
    provenance = {
        "dataset": "scikit-learn 20 Newsgroups by-date split",
        "sklearn_version": sklearn_version,
        "archive_sha256": sha256_file(archive),
        "categories": categories,
        "pairs": [list(pair) for pair in CATEGORY_PAIRS],
        "remove": list(fetch_arguments["remove"]),
        "shuffle": True,
        "random_state": seed,
        "vectorizer": {
            "class": "sklearn.feature_extraction.text.HashingVectorizer",
            "n_features": dimensions,
            "alternate_sign": False,
            "norm": "l2",
            "stop_words": "english",
        },
    }
    return datasets, provenance


def train_lane(
    lane: LearnedTop1MoE,
    dataset: RegimeDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    balance_weight: float,
    seed: int,
) -> dict[str, Any]:
    router_before = snapshot(lane.router)
    optimizer = torch.optim.AdamW(lane.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    initial_loss = None
    final_loss = None
    utilization = torch.zeros(lane.expert_count, dtype=torch.int64)
    lane.train()
    for _ in range(epochs):
        order = torch.randperm(
            dataset.train_inputs.shape[0],
            generator=generator,
        )
        utilization.zero_()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            inputs = dataset.train_inputs[rows]
            labels = dataset.train_labels[rows]
            optimizer.zero_grad(set_to_none=True)
            logits, trace = lane(inputs)
            task_loss = F.cross_entropy(logits, labels)
            balance_loss = lane.load_balance_loss(trace)
            loss = task_loss + balance_weight * balance_loss
            loss.backward()
            optimizer.step()
            utilization += torch.bincount(
                trace.local_experts.detach(),
                minlength=lane.expert_count,
            )
            if initial_loss is None:
                initial_loss = float(task_loss.detach())
            final_loss = float(task_loss.detach())
    return {
        "initial_task_loss": initial_loss,
        "final_task_loss": final_loss,
        "router_maximum_parameter_delta": maximum_parameter_delta(
            router_before,
            lane.router,
        ),
        "final_epoch_expert_utilization": utilization.tolist(),
    }


@torch.no_grad()
def evaluate(
    lanes: list[LearnedTop1MoE],
    bank: AuthorizedExpertBank,
    datasets: list[RegimeDataset],
    runtimes: list[GateExecutionPreflight],
) -> dict[str, Any]:
    rows = []
    total_correct = 0
    total_examples = 0
    prediction_differences = 0
    exact_logit_mismatches = 0
    maximum_logit_difference = 0.0
    maximum_dense_probability_difference = 0.0
    unauthorized_dispatches = 0

    for regime, (lane, dataset) in enumerate(zip(lanes, datasets, strict=True)):
        runtime = runtimes[regime]
        lane.eval()
        bank.eval()

        def scored_evaluation(
            active_lane: LearnedTop1MoE = lane,
            active_dataset: RegimeDataset = dataset,
            active_regime: int = regime,
        ) -> dict[str, Any]:
            separate_logits, separate_trace = active_lane(active_dataset.test_inputs)
            packed_logits, packed_trace = bank(active_dataset.test_inputs, active_regime)
            dense_probabilities, dense_selected = bank.dense_masked_route(
                active_dataset.test_inputs,
                active_regime,
            )
            allowed = set(bank.allowed_experts(active_regime))
            local_difference = float(
                (separate_logits - packed_logits).abs().max()
            )
            probability_difference = float(
                (
                    dense_probabilities[
                        :, min(allowed) : max(allowed) + 1
                    ]
                    - packed_trace.probabilities
                ).abs().max()
            )
            separate_predictions = separate_logits.argmax(dim=-1)
            packed_predictions = packed_logits.argmax(dim=-1)
            selected_global = packed_trace.global_experts.tolist()
            return {
                "name": active_dataset.name,
                "train_examples": int(active_dataset.train_inputs.shape[0]),
                "test_examples": int(active_dataset.test_inputs.shape[0]),
                "correct": int(
                    (packed_predictions == active_dataset.test_labels).sum()
                ),
                "prediction_differences": int(
                    (separate_predictions != packed_predictions).sum()
                ),
                "exact_logit_equal": bool(
                    torch.equal(separate_logits, packed_logits)
                ),
                "maximum_logit_difference": local_difference,
                "dense_mask_maximum_probability_difference": probability_difference,
                "dense_mask_route_differences": int(
                    (dense_selected != packed_trace.global_experts).sum()
                ),
                "unauthorized_dispatches": sum(
                    expert not in allowed for expert in selected_global
                ),
                "expert_utilization": torch.bincount(
                    separate_trace.local_experts,
                    minlength=active_lane.expert_count,
                ).tolist(),
            }

        row = runtime.invoke(regime, scored_evaluation)
        rows.append(row)
        total_correct += row["correct"]
        total_examples += row["test_examples"]
        prediction_differences += row["prediction_differences"]
        exact_logit_mismatches += int(not row["exact_logit_equal"])
        maximum_logit_difference = max(
            maximum_logit_difference,
            row["maximum_logit_difference"],
        )
        maximum_dense_probability_difference = max(
            maximum_dense_probability_difference,
            row["dense_mask_maximum_probability_difference"],
        )
        unauthorized_dispatches += row["unauthorized_dispatches"]
        unauthorized_dispatches += row["dense_mask_route_differences"]

    return {
        "per_regime": rows,
        "macro_accuracy": float(
            np.mean([row["correct"] / row["test_examples"] for row in rows])
        ),
        "micro_accuracy": total_correct / total_examples,
        "correct": total_correct,
        "examples": total_examples,
        "prediction_differences": prediction_differences,
        "exact_logit_mismatched_regimes": exact_logit_mismatches,
        "maximum_logit_difference": maximum_logit_difference,
        "maximum_dense_probability_difference": maximum_dense_probability_difference,
        "unauthorized_dispatches": unauthorized_dispatches,
        "minimum_expert_utilization_fraction": min(
            min(row["expert_utilization"]) / row["test_examples"]
            for row in rows
        ),
    }


def audit_training_isolation(
    bank: AuthorizedExpertBank,
    dataset: RegimeDataset,
    *,
    regime: int,
    learning_rate: float,
    balance_weight: float,
    batch_size: int,
) -> dict[str, Any]:
    audited = AuthorizedExpertBank.pack(bank.lanes)
    active_before = snapshot(audited.lanes[regime])
    inactive_before = {
        other: snapshot(audited.lanes[other])
        for other in range(audited.regimes)
        if other != regime
    }
    optimizer = torch.optim.AdamW(audited.parameters(), lr=learning_rate)
    inputs = dataset.train_inputs[:batch_size]
    labels = dataset.train_labels[:batch_size]
    optimizer.zero_grad(set_to_none=True)
    logits, trace = audited(inputs, regime)
    loss = F.cross_entropy(logits, labels) + balance_weight * audited.lanes[
        regime
    ].load_balance_loss(trace)
    loss.backward()
    inactive_gradient_tensors = sum(
        parameter.grad is not None
        for other in inactive_before
        for parameter in audited.lanes[other].parameters()
    )
    optimizer.step()
    inactive_optimizer_states = sum(
        parameter in optimizer.state
        for other in inactive_before
        for parameter in audited.lanes[other].parameters()
    )
    return {
        "regime": regime,
        "active_maximum_parameter_delta": maximum_parameter_delta(
            active_before,
            audited.lanes[regime],
        ),
        "inactive_maximum_parameter_delta": max(
            maximum_parameter_delta(before, audited.lanes[other])
            for other, before in inactive_before.items()
        ),
        "inactive_gradient_tensors": inactive_gradient_tensors,
        "inactive_optimizer_states": inactive_optimizer_states,
    }


def parameter_bytes(module: torch.nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in module.parameters():
        pointer = parameter.untyped_storage().data_ptr()
        if pointer not in seen:
            seen.add(pointer)
            total += parameter.numel() * parameter.element_size()
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--regimes", type=int, default=8)
    parser.add_argument("--input-dimensions", type=int, default=2048)
    parser.add_argument("--hidden-dimensions", type=int, default=32)
    parser.add_argument("--experts-per-regime", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--balance-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.regimes != 8 and not args.synthetic:
        raise ValueError("the publication dataset defines exactly eight regimes")
    if min(
        args.regimes,
        args.input_dimensions,
        args.hidden_dimensions,
        args.experts_per_regime,
        args.epochs,
        args.batch_size,
        args.threads,
    ) <= 0:
        raise ValueError("all counts and dimensions must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()

    if args.synthetic:
        datasets = synthetic_datasets(
            regimes=args.regimes,
            dimensions=args.input_dimensions,
            train_examples=128,
            test_examples=64,
            seed=args.seed,
        )
        data_provenance = {
            "dataset": "deterministic synthetic Gaussian binary tasks",
            "train_examples_per_regime": 128,
            "test_examples_per_regime": 64,
        }
    else:
        datasets, data_provenance = newsgroups_datasets(
            dimensions=args.input_dimensions,
            seed=args.seed,
        )

    lanes = []
    training_rows = []
    for regime, dataset in enumerate(datasets):
        torch.manual_seed(args.seed + regime)
        lane = LearnedTop1MoE(
            input_dimensions=args.input_dimensions,
            hidden_dimensions=args.hidden_dimensions,
            classes=2,
            experts=args.experts_per_regime,
        )
        training_rows.append(
            {
                "regime": regime,
                "name": dataset.name,
                **train_lane(
                    lane,
                    dataset,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    balance_weight=args.balance_weight,
                    seed=args.seed + 10_000 + regime,
                ),
            }
        )
        lanes.append(lane)

    bank = AuthorizedExpertBank.pack(lanes)
    runtimes = [
        GateExecutionPreflight(
            model_id="authorized-learned-moe-20newsgroups",
            dimensions=args.input_dimensions,
            authorized_regime_ids=[regime],
        )
        for regime in range(args.regimes)
    ]
    evaluation = evaluate(lanes, bank, datasets, runtimes)
    isolation = audit_training_isolation(
        bank,
        datasets[0],
        regime=0,
        learning_rate=args.learning_rate,
        balance_weight=args.balance_weight,
        batch_size=args.batch_size,
    )
    per_regime_rejections = [
        {
            "regime": regime,
            **runtime.rejection_probe(),
        }
        for regime, runtime in enumerate(runtimes)
    ]
    rejection_probe = {
        "per_regime": per_regime_rejections,
        "all_rejected": all(
            row["all_rejected"] for row in per_regime_rejections
        ),
        "unauthorized_model_calls": sum(
            row["unauthorized_model_calls"] for row in per_regime_rejections
        ),
    }

    negative_logits = torch.tensor([[-2.0, -1.0, 8.0]])
    negative_allowed = torch.tensor([[True, True, False]])
    negative_selected = int(
        unsafe_zero_logit_route(negative_logits, negative_allowed).item()
    )
    invalid_mask_detected = negative_selected == 2
    utility_floor = 0.90 if args.synthetic else 0.75
    utilization_floor = 0.10
    minimum_router_delta = min(
        row["router_maximum_parameter_delta"] for row in training_rows
    )
    passed = (
        evaluation["macro_accuracy"] >= utility_floor
        and evaluation["prediction_differences"] == 0
        and evaluation["exact_logit_mismatched_regimes"] == 0
        and evaluation["maximum_logit_difference"] == 0.0
        and evaluation["maximum_dense_probability_difference"] <= 1e-7
        and evaluation["unauthorized_dispatches"] == 0
        and evaluation["minimum_expert_utilization_fraction"]
        >= utilization_floor
        and minimum_router_delta > 0.0
        and isolation["active_maximum_parameter_delta"] > 0.0
        and isolation["inactive_maximum_parameter_delta"] == 0.0
        and isolation["inactive_gradient_tensors"] == 0
        and isolation["inactive_optimizer_states"] == 0
        and rejection_probe["all_rejected"]
        and rejection_probe["unauthorized_model_calls"] == 0
        and invalid_mask_detected
    )

    artifact = {
        "schema_version": 1,
        "experiment": "authorized_learned_moe",
        "status": "pass" if passed else "failure",
        "scope": (
            "Learned top-1 routing within fixed execution-authorized expert sets; "
            "held-out binary classification and exact separate-to-packed "
            "equivalence. This does not partition a shared attention path."
        ),
        "verdict_basis": {
            "utility_macro_accuracy_floor": utility_floor,
            "minimum_expert_utilization_fraction": utilization_floor,
            "every_learned_router_parameter_delta_positive": True,
            "exact_separate_to_packed_predictions": True,
            "exact_separate_to_packed_logits": True,
            "zero_unauthorized_dispatches": True,
            "zero_inactive_training_state": True,
            "zero_unauthorized_runtime_calls": True,
            "invalid_zero_logit_mask_detected": True,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "git_revision": git_revision(),
        "source": collect_source_provenance(),
        "libraries": collect_library_provenance(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "threads": args.threads,
        },
        "config": {
            **vars(args),
            "output_dir": str(args.output_dir) if args.output_dir else None,
        },
        "data": data_provenance,
        "training": training_rows,
        "minimum_router_maximum_parameter_delta": minimum_router_delta,
        "evaluation": evaluation,
        "training_isolation": isolation,
        "runtime": {
            "surface": "schemen-gate/research-execution-preflight-v1",
            "authorities": [runtime.evidence() for runtime in runtimes],
            "authorized_model_calls": sum(
                runtime.model_calls for runtime in runtimes
            ),
            "rejection_probe": rejection_probe,
        },
        "negative_control": {
            "construction": "multiply unauthorized pre-softmax logits by zero",
            "selected_expert": negative_selected,
            "unauthorized_expert_selected": invalid_mask_detected,
        },
        "storage": {
            "separate_lane_parameter_bytes": sum(
                parameter_bytes(lane) for lane in lanes
            ),
            "packed_bank_parameter_bytes": parameter_bytes(bank),
            "note": (
                "The experiment isolates router/expert packing. A frozen "
                "feature transform has no duplicated learned parameters here."
            ),
        },
        "claim_boundaries": [
            "The authority gate is fixed; only the semantic router is learned.",
            "Exact zero dip is a copy/packing equivalence result.",
            "Held-out utility is measured on eight binary 20 Newsgroups tasks.",
            "The experiment has no shared attention, cache, or residual path.",
            "The zero-logit construction is deliberately invalid.",
        ],
    }

    output_dir = args.output_dir or Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output_dir / f"authorized_learned_moe_{timestamp}.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": artifact["status"],
        "macro_accuracy": evaluation["macro_accuracy"],
        "micro_accuracy": evaluation["micro_accuracy"],
        "prediction_differences": evaluation["prediction_differences"],
        "maximum_logit_difference": evaluation["maximum_logit_difference"],
        "unauthorized_dispatches": evaluation["unauthorized_dispatches"],
        "minimum_expert_utilization_fraction": evaluation[
            "minimum_expert_utilization_fraction"
        ],
        "inactive_maximum_parameter_delta": isolation[
            "inactive_maximum_parameter_delta"
        ],
        "unauthorized_runtime_calls": rejection_probe[
            "unauthorized_model_calls"
        ],
        "output": str(output),
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
