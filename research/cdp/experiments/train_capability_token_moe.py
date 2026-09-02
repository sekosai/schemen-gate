"""Train token-level MoE routing under a trusted capability-prefix channel.

execution authority selects a regime. The selected private lane internally
injects a learned continuous prefix and routes every non-padding user token
among that regime's authorized experts. Capability-looking text remains
ordinary user input and cannot select the trusted prefix or another expert set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from capability_token_moe import (
    CapabilityTokenBank,
    CapabilityTokenLane,
    maximum_parameter_delta,
    snapshot,
)
from execution_preflight import GateExecutionPreflight
from library_provenance import collect_library_provenance, collect_source_provenance
from train_authorized_moe import CATEGORY_PAIRS, sha256_file

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


@dataclass(frozen=True)
class TokenRegimeDataset:
    name: str
    train_tokens: torch.Tensor
    train_labels: torch.Tensor
    test_tokens: torch.Tensor
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


def token_id(token: str, vocabulary_size: int) -> int:
    if vocabulary_size < 2:
        raise ValueError("vocabulary must reserve padding plus one token")
    digest = hashlib.blake2b(
        token.lower().encode("utf-8"),
        digest_size=8,
        person=b"cdp-tok-v1",
    ).digest()
    return 1 + int.from_bytes(digest, "big") % (vocabulary_size - 1)


def encode_texts(
    texts: list[str],
    *,
    vocabulary_size: int,
    maximum_tokens: int,
) -> torch.Tensor:
    encoded = torch.zeros((len(texts), maximum_tokens), dtype=torch.long)
    empty_id = token_id("__empty_document__", vocabulary_size)
    for row, text in enumerate(texts):
        pieces = TOKEN_PATTERN.findall(text.lower())[:maximum_tokens]
        if not pieces:
            pieces = ["__empty_document__"]
        encoded[row, : len(pieces)] = torch.tensor(
            [token_id(piece, vocabulary_size) for piece in pieces],
            dtype=torch.long,
        )
        if len(pieces) == 1 and pieces[0] == "__empty_document__":
            encoded[row, 0] = empty_id
    return encoded


def synthetic_datasets(
    *,
    regimes: int,
    vocabulary_size: int,
    maximum_tokens: int,
    train_examples: int,
    test_examples: int,
    seed: int,
) -> list[TokenRegimeDataset]:
    datasets = []
    for regime in range(regimes):
        rng = np.random.default_rng(seed + regime)
        class_zero = np.arange(1, 17)
        class_one = np.arange(17, 33)
        noise = np.arange(33, min(vocabulary_size, 97))

        def split(
            examples: int,
            *,
            local_rng: np.random.Generator = rng,
            local_class_zero: np.ndarray = class_zero,
            local_class_one: np.ndarray = class_one,
            local_noise: np.ndarray = noise,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            labels = np.arange(examples, dtype=np.int64) % 2
            tokens = np.zeros((examples, maximum_tokens), dtype=np.int64)
            for row, label in enumerate(labels):
                source = local_class_zero if label == 0 else local_class_one
                signal_count = max(1, int(maximum_tokens * 0.75))
                tokens[row, :signal_count] = local_rng.choice(source, signal_count)
                tokens[row, signal_count:] = local_rng.choice(
                    local_noise,
                    maximum_tokens - signal_count,
                )
                local_rng.shuffle(tokens[row])
            return torch.from_numpy(tokens), torch.from_numpy(labels)

        train_tokens, train_labels = split(train_examples)
        test_tokens, test_labels = split(test_examples)
        datasets.append(
            TokenRegimeDataset(
                name=f"synthetic-token-{regime}",
                train_tokens=train_tokens,
                train_labels=train_labels,
                test_tokens=test_tokens,
                test_labels=test_labels,
            )
        )
    return datasets


def newsgroups_datasets(
    *,
    vocabulary_size: int,
    maximum_tokens: int,
    seed: int,
) -> tuple[list[TokenRegimeDataset], dict[str, Any]]:
    from sklearn import __version__ as sklearn_version
    from sklearn.datasets import fetch_20newsgroups, get_data_home

    categories = sorted({name for pair in CATEGORY_PAIRS for name in pair})
    arguments = {
        "categories": categories,
        "remove": ("headers", "footers", "quotes"),
        "shuffle": True,
        "random_state": seed,
    }
    train = fetch_20newsgroups(subset="train", **arguments)
    test = fetch_20newsgroups(subset="test", **arguments)
    train_names = np.asarray([train.target_names[index] for index in train.target])
    test_names = np.asarray([test.target_names[index] for index in test.target])
    datasets = []
    for first, second in CATEGORY_PAIRS:
        train_mask = np.logical_or(train_names == first, train_names == second)
        test_mask = np.logical_or(test_names == first, test_names == second)
        datasets.append(
            TokenRegimeDataset(
                name=f"{first}__vs__{second}",
                train_tokens=encode_texts(
                    [text for text, keep in zip(train.data, train_mask, strict=True) if keep],
                    vocabulary_size=vocabulary_size,
                    maximum_tokens=maximum_tokens,
                ),
                train_labels=torch.from_numpy(
                    (train_names[train_mask] == second).astype(np.int64)
                ),
                test_tokens=encode_texts(
                    [text for text, keep in zip(test.data, test_mask, strict=True) if keep],
                    vocabulary_size=vocabulary_size,
                    maximum_tokens=maximum_tokens,
                ),
                test_labels=torch.from_numpy(
                    (test_names[test_mask] == second).astype(np.int64)
                ),
            )
        )
    archive = Path(get_data_home()) / "20news-bydate_py3.pkz"
    return datasets, {
        "dataset": "scikit-learn 20 Newsgroups by-date split",
        "sklearn_version": sklearn_version,
        "archive_sha256": sha256_file(archive),
        "categories": categories,
        "pairs": [list(pair) for pair in CATEGORY_PAIRS],
        "remove": list(arguments["remove"]),
        "shuffle": True,
        "random_state": seed,
        "tokenizer": {
            "pattern": TOKEN_PATTERN.pattern,
            "hash": "BLAKE2b-64(person='cdp-tok-v1')",
            "vocabulary_size": vocabulary_size,
            "maximum_tokens": maximum_tokens,
            "padding_id": 0,
        },
    }


def train_lane(
    lane: CapabilityTokenLane,
    dataset: TokenRegimeDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    balance_weight: float,
    seed: int,
) -> dict[str, Any]:
    prefix_before = lane.capability_prefix.detach().clone()
    router_before = snapshot(lane.router)
    optimizer = torch.optim.AdamW(lane.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    initial_loss = None
    final_loss = None
    utilization = torch.zeros(lane.expert_count, dtype=torch.int64)
    lane.train()
    for _ in range(epochs):
        order = torch.randperm(len(dataset.train_tokens), generator=generator)
        utilization.zero_()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, trace = lane(dataset.train_tokens[rows])
            task_loss = F.cross_entropy(logits, dataset.train_labels[rows])
            loss = task_loss + balance_weight * lane.load_balance_loss(trace)
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
        "prefix_maximum_parameter_delta": float(
            (lane.capability_prefix.detach() - prefix_before).abs().max()
        ),
        "router_maximum_parameter_delta": maximum_parameter_delta(
            router_before,
            lane.router,
        ),
        "final_epoch_token_expert_utilization": utilization.tolist(),
    }


@torch.no_grad()
def evaluate(
    lanes: list[CapabilityTokenLane],
    bank: CapabilityTokenBank,
    datasets: list[TokenRegimeDataset],
    runtimes: list[GateExecutionPreflight],
) -> dict[str, Any]:
    rows = []
    total_correct = total_examples = 0
    prediction_differences = exact_logit_mismatches = 0
    maximum_logit_difference = maximum_dense_probability_difference = 0.0
    unauthorized_dispatches = 0
    for regime, (lane, dataset) in enumerate(zip(lanes, datasets, strict=True)):
        lane.eval()
        bank.eval()

        def scored(
            active_lane: CapabilityTokenLane = lane,
            active_dataset: TokenRegimeDataset = dataset,
            active_regime: int = regime,
        ) -> dict[str, Any]:
            separate_logits, separate_trace = active_lane(active_dataset.test_tokens)
            packed_logits, packed_trace = bank(active_dataset.test_tokens, active_regime)
            dense_probabilities, dense_selected = bank.dense_masked_route(
                active_dataset.test_tokens,
                active_regime,
            )
            allowed = set(bank.allowed_experts(active_regime))
            predictions = packed_logits.argmax(dim=-1)
            return {
                "name": active_dataset.name,
                "train_examples": len(active_dataset.train_tokens),
                "test_examples": len(active_dataset.test_tokens),
                "routed_tokens": int(len(packed_trace.global_experts)),
                "correct": int((predictions == active_dataset.test_labels).sum()),
                "prediction_differences": int(
                    (separate_logits.argmax(dim=-1) != predictions).sum()
                ),
                "exact_logit_equal": bool(torch.equal(separate_logits, packed_logits)),
                "maximum_logit_difference": float(
                    (separate_logits - packed_logits).abs().max()
                ),
                "dense_mask_maximum_probability_difference": float(
                    (
                        dense_probabilities[:, min(allowed) : max(allowed) + 1]
                        - packed_trace.probabilities
                    ).abs().max()
                ),
                "dense_mask_route_differences": int(
                    (dense_selected != packed_trace.global_experts).sum()
                ),
                "unauthorized_dispatches": sum(
                    expert not in allowed
                    for expert in packed_trace.global_experts.tolist()
                ),
                "token_expert_utilization": torch.bincount(
                    separate_trace.local_experts,
                    minlength=active_lane.expert_count,
                ).tolist(),
            }

        row = runtimes[regime].invoke(regime, scored)
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
        "routed_tokens": sum(row["routed_tokens"] for row in rows),
        "prediction_differences": prediction_differences,
        "exact_logit_mismatched_regimes": exact_logit_mismatches,
        "maximum_logit_difference": maximum_logit_difference,
        "maximum_dense_probability_difference": maximum_dense_probability_difference,
        "unauthorized_dispatches": unauthorized_dispatches,
        "minimum_expert_utilization_fraction": min(
            min(row["token_expert_utilization"]) / row["routed_tokens"]
            for row in rows
        ),
    }


def spoof_probe(
    bank: CapabilityTokenBank,
    *,
    vocabulary_size: int,
    maximum_tokens: int,
) -> dict[str, Any]:
    marker_text = " ".join(
        f"CAPABILITY_REGIME_{regime}" for regime in range(bank.regimes)
    )
    tokens = encode_texts(
        [marker_text],
        vocabulary_size=vocabulary_size,
        maximum_tokens=maximum_tokens,
    )
    rows = []
    for regime in range(bank.regimes):
        _, trace = bank(tokens, regime)
        allowed = set(bank.allowed_experts(regime))
        rows.append(
            {
                "runtime_regime": regime,
                "selected_global_experts": sorted(
                    set(trace.global_experts.tolist())
                ),
                "unauthorized_dispatches": sum(
                    expert not in allowed for expert in trace.global_experts.tolist()
                ),
            }
        )
    return {
        "user_text": marker_text,
        "control_channel": (
            "capability prefix is selected internally from execution regime; "
            "user text is tokenized only as ordinary data"
        ),
        "per_regime": rows,
        "unauthorized_dispatches": sum(
            row["unauthorized_dispatches"] for row in rows
        ),
    }


def audit_training_isolation(
    bank: CapabilityTokenBank,
    dataset: TokenRegimeDataset,
    *,
    regime: int,
    batch_size: int,
    learning_rate: float,
    balance_weight: float,
) -> dict[str, Any]:
    audited = CapabilityTokenBank(bank.lanes)
    active_before = snapshot(audited.lanes[regime])
    inactive_before = {
        other: snapshot(audited.lanes[other])
        for other in range(audited.regimes)
        if other != regime
    }
    optimizer = torch.optim.AdamW(audited.parameters(), lr=learning_rate)
    optimizer.zero_grad(set_to_none=True)
    logits, trace = audited(dataset.train_tokens[:batch_size], regime)
    loss = F.cross_entropy(
        logits,
        dataset.train_labels[:batch_size],
    ) + balance_weight * audited.lanes[regime].load_balance_loss(trace)
    loss.backward()
    inactive_gradient_tensors = sum(
        parameter.grad is not None
        for other in inactive_before
        for parameter in audited.lanes[other].parameters()
    )
    optimizer.step()
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
        "inactive_optimizer_states": sum(
            parameter in optimizer.state
            for other in inactive_before
            for parameter in audited.lanes[other].parameters()
        ),
    }


def parameter_bytes(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--regimes", type=int, default=8)
    parser.add_argument("--vocabulary-size", type=int, default=4096)
    parser.add_argument("--maximum-tokens", type=int, default=64)
    parser.add_argument("--embedding-dimensions", type=int, default=64)
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
        args.vocabulary_size,
        args.maximum_tokens,
        args.embedding_dimensions,
        args.hidden_dimensions,
        args.experts_per_regime,
        args.epochs,
        args.batch_size,
        args.threads,
    ) <= 0:
        raise ValueError("all counts and dimensions must be positive")
    if args.synthetic and args.vocabulary_size < 97:
        raise ValueError("synthetic token motifs require vocabulary size >= 97")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    started = time.time()
    if args.synthetic:
        datasets = synthetic_datasets(
            regimes=args.regimes,
            vocabulary_size=args.vocabulary_size,
            maximum_tokens=args.maximum_tokens,
            train_examples=128,
            test_examples=64,
            seed=args.seed,
        )
        data_provenance = {
            "dataset": "deterministic synthetic token motifs",
            "train_examples_per_regime": 128,
            "test_examples_per_regime": 64,
        }
    else:
        datasets, data_provenance = newsgroups_datasets(
            vocabulary_size=args.vocabulary_size,
            maximum_tokens=args.maximum_tokens,
            seed=args.seed,
        )

    lanes = []
    training = []
    for regime, dataset in enumerate(datasets):
        torch.manual_seed(args.seed + regime)
        lane = CapabilityTokenLane(
            vocabulary_size=args.vocabulary_size,
            embedding_dimensions=args.embedding_dimensions,
            hidden_dimensions=args.hidden_dimensions,
            classes=2,
            experts=args.experts_per_regime,
        )
        training.append(
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
                    seed=args.seed + 20_000 + regime,
                ),
            }
        )
        lanes.append(lane)
    bank = CapabilityTokenBank(lanes)
    runtimes = [
        GateExecutionPreflight(
            model_id="capability-prefix-token-moe-20newsgroups",
            dimensions=args.embedding_dimensions,
            authorized_regime_ids=[regime],
        )
        for regime in range(args.regimes)
    ]
    evaluation = evaluate(lanes, bank, datasets, runtimes)
    spoofing = spoof_probe(
        bank,
        vocabulary_size=args.vocabulary_size,
        maximum_tokens=args.maximum_tokens,
    )
    isolation = audit_training_isolation(
        bank,
        datasets[0],
        regime=0,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        balance_weight=args.balance_weight,
    )
    per_regime_rejections = [
        {"regime": regime, **runtime.rejection_probe()}
        for regime, runtime in enumerate(runtimes)
    ]
    rejection_probe = {
        "per_regime": per_regime_rejections,
        "all_rejected": all(row["all_rejected"] for row in per_regime_rejections),
        "unauthorized_model_calls": sum(
            row["unauthorized_model_calls"] for row in per_regime_rejections
        ),
    }
    utility_floor = 0.90 if args.synthetic else 0.72
    utilization_floor = 0.10
    minimum_prefix_delta = min(
        row["prefix_maximum_parameter_delta"] for row in training
    )
    minimum_router_delta = min(
        row["router_maximum_parameter_delta"] for row in training
    )
    passed = (
        evaluation["macro_accuracy"] >= utility_floor
        and evaluation["prediction_differences"] == 0
        and evaluation["exact_logit_mismatched_regimes"] == 0
        and evaluation["maximum_logit_difference"] == 0.0
        and evaluation["maximum_dense_probability_difference"] <= 1e-6
        and evaluation["unauthorized_dispatches"] == 0
        and evaluation["minimum_expert_utilization_fraction"] >= utilization_floor
        and minimum_prefix_delta > 0.0
        and minimum_router_delta > 0.0
        and spoofing["unauthorized_dispatches"] == 0
        and isolation["active_maximum_parameter_delta"] > 0.0
        and isolation["inactive_maximum_parameter_delta"] == 0.0
        and isolation["inactive_gradient_tensors"] == 0
        and isolation["inactive_optimizer_states"] == 0
        and rejection_probe["all_rejected"]
        and rejection_probe["unauthorized_model_calls"] == 0
    )
    artifact = {
        "schema_version": 1,
        "experiment": "capability_prefix_token_moe",
        "status": "pass" if passed else "failure",
        "scope": (
            "Token-level learned top-1 routing conditioned on an internal "
            "soft prefix selected after singleton execution authorization. "
            "User text is not authority and cannot select the prefix."
        ),
        "verdict_basis": {
            "utility_macro_accuracy_floor": utility_floor,
            "minimum_token_expert_utilization_fraction": utilization_floor,
            "exact_separate_to_packed_logits_and_predictions": True,
            "zero_unauthorized_token_dispatch": True,
            "zero_spoofed_prefix_dispatch": True,
            "zero_inactive_training_state": True,
            "zero_unauthorized_runtime_calls": True,
            "every_prefix_and_router_parameter_delta_positive": True,
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
        "training": training,
        "minimum_prefix_maximum_parameter_delta": minimum_prefix_delta,
        "minimum_router_maximum_parameter_delta": minimum_router_delta,
        "evaluation": evaluation,
        "spoofing_control": spoofing,
        "training_isolation": isolation,
        "runtime": {
            "surface": "schemen-gate/research-execution-preflight-v1",
            "authorities": [runtime.evidence() for runtime in runtimes],
            "authorized_model_calls": sum(runtime.model_calls for runtime in runtimes),
            "rejection_probe": rejection_probe,
        },
        "storage": {
            "separate_lane_parameter_bytes": sum(parameter_bytes(lane) for lane in lanes),
            "packed_bank_parameter_bytes": parameter_bytes(bank),
            "note": "Packing copies identical lane state; this artifact does not claim compression.",
        },
        "claim_boundaries": [
            "A plaintext or user-authored prefix is not a capability.",
            "execution authority selects the internal continuous prefix and expert set.",
            "User tokens may affect routing only within the selected expert set.",
            "Exact zero dip is a copy-and-pack equivalence result.",
            "This small classifier has no shared attention, cache, or language-model generation.",
        ],
    }
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output_dir / f"capability_prefix_token_moe_{timestamp}.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": artifact["status"],
        "macro_accuracy": evaluation["macro_accuracy"],
        "micro_accuracy": evaluation["micro_accuracy"],
        "routed_tokens": evaluation["routed_tokens"],
        "prediction_differences": evaluation["prediction_differences"],
        "maximum_logit_difference": evaluation["maximum_logit_difference"],
        "unauthorized_dispatches": evaluation["unauthorized_dispatches"],
        "spoofed_prefix_unauthorized_dispatches": spoofing["unauthorized_dispatches"],
        "minimum_expert_utilization_fraction": evaluation[
            "minimum_expert_utilization_fraction"
        ],
        "inactive_maximum_parameter_delta": isolation[
            "inactive_maximum_parameter_delta"
        ],
        "unauthorized_runtime_calls": rejection_probe["unauthorized_model_calls"],
        "output": str(output),
    }, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
