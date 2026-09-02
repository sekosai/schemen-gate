"""Matched DistilBERT adaptation experiment for Modal (PLAN Priority 3).

The unit of remote work is one paired ``(R, seed)`` experiment.  Every job
uses the same initial checkpoint, tokenized data, partition masks, label
permutations, optimizer recipe, and evaluator for these conditions:

* ``ungated_separate``: one independently fine-tuned model per regime.
* ``frozen_posthoc_gate``: the trained separate models evaluated with a gate.
* ``continued_gate_aware_{10,25,50}pct``: independent continuations from each
  ungated checkpoint and its optimizer state.
* ``cotrained_gate_aware``: one model trained with every gate from step zero.

Mask construction is deliberately non-proprietary and crypto-free: a seeded
``torch.randperm`` is split into equal disjoint blocks. No external workspace
files are uploaded.

Each remote job checkpoints a timestamped JSON artifact to a named Modal
Volume after every condition (and on failure) before returning.  The local
entrypoint writes a combined timestamped JSON under ``experiments/results``.

Examples (these commands are documentation; this file never launches itself):

    modal run experiments/modal_matched_adaptation.py --smoke
    modal run experiments/modal_matched_adaptation.py
    modal run experiments/modal_matched_adaptation.py \
        --r-values 4,8,16 --seeds 42,123,256,789,1337
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import platform
import random
import re
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cdp-matched-distilbert-adaptation"
VOLUME_NAME = "cdp-matched-adaptation-results"
VOLUME_PATH = "/results"
MODEL_NAME = "distilbert/distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
DEFAULT_SEEDS = (42, 123, 256, 789, 1337)
DEFAULT_R_VALUES = (4, 8, 16)
ADAPTATION_FRACTIONS = (0.10, 0.25, 0.50)
HIDDEN_DIM = 768
N_CLASSES = 4

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.13.0",
    "transformers==5.10.1",
    "datasets==5.0.1",
    "numpy==2.4.6",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_int_csv(value: str, *, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated list of integers") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} must not contain duplicates")
    return parsed


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


@app.function(
    max_containers=3,
    gpu="L4",
    image=gpu_image,
    timeout=6 * 60 * 60,
    volumes={VOLUME_PATH: results_volume},
)
def run_paired_job(config: dict[str, Any]) -> dict[str, Any]:
    """Run and durably archive one paired R/seed condition matrix."""
    import gc

    import datasets
    import numpy as np
    import torch
    import torch.nn as nn
    import transformers
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import DistilBertModel, DistilBertTokenizerFast

    started = time.time()
    started_at = _utc_now()
    R = int(config["R"])
    seed = int(config["seed"])
    artifact_name = (
        f"matched_adaptation_R{R}_seed{seed}_{config['run_id']}.json"
    )
    artifact_path = Path(VOLUME_PATH) / artifact_name

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "matched_distilbert_adaptation",
        "job": {"R": R, "seed": seed},
        "status": "running",
        "started_at": started_at,
        "updated_at": started_at,
        "git_revision": config["git_revision"],
        "config": config,
        "environment": {},
        "protocol": {
            "remote_unit": "one paired R/seed matrix",
            "mask_logic": (
                "crypto-free seeded torch.randperm split into disjoint blocks"
            ),
            "checkpoint_matching": (
                "all models begin from one in-memory initial state; post-hoc and "
                "continued conditions begin from each trained ungated state"
            ),
            "data_matching": (
                "all conditions use the same tokenized train/evaluation tensors "
                "and deterministic epoch/batch order"
            ),
            "optimizer_matching": (
                "AdamW parameter groups and hyperparameters are identical; "
                "continued runs restore the ungated optimizer state"
            ),
            "evaluation_matching": (
                "all conditions use the same full evaluation tensor and inverse "
                "label-permutation accuracy"
            ),
        },
        "masks": {},
        "label_permutations": {},
        "conditions": {},
    }

    def persist() -> None:
        artifact["updated_at"] = _utc_now()
        artifact["elapsed_seconds"] = time.time() - started
        temporary = artifact_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(_json_safe(artifact), indent=2) + "\n")
        temporary.replace(artifact_path)
        results_volume.commit()

    def set_seed(value: int) -> None:
        random.seed(value)
        np.random.seed(value % (2**32))
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def reset_device() -> None:
        gc.collect()
        torch.cuda.empty_cache()

    try:
        if R not in DEFAULT_R_VALUES:
            raise ValueError(f"R must be one of {DEFAULT_R_VALUES}, got {R}")
        if HIDDEN_DIM % R:
            raise ValueError(f"{HIDDEN_DIM=} must be divisible by {R=}")

        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        properties = torch.cuda.get_device_properties(0)
        artifact["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "datasets": datasets.__version__,
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "gpu": properties.name,
            "gpu_memory_bytes": properties.total_memory,
            "modal_task_id": os.environ.get("MODAL_TASK_ID", "unknown"),
            "modal_region": os.environ.get("MODAL_REGION", "unknown"),
        }

        # Purely benchmark-oriented masks.  No secrets, hashes, HMAC, HKDF,
        # rejection sampling, or proprietary gate derivation are used.
        mask_generator = torch.Generator(device="cpu").manual_seed(
            int(config["mask_seed"]) + R * 1_000_003 + seed
        )
        partition = torch.randperm(HIDDEN_DIM, generator=mask_generator)
        blocks = partition.reshape(R, HIDDEN_DIM // R)
        masks = torch.zeros(R, HIDDEN_DIM, dtype=torch.float32)
        for regime in range(R):
            masks[regime, blocks[regime]] = 1.0
        if not torch.equal(masks.sum(dim=0), torch.ones(HIDDEN_DIM)):
            raise AssertionError("masks must be disjoint and exhaustive")
        masks = masks.to(device)
        artifact["masks"] = {
            "seed": int(config["mask_seed"]) + R * 1_000_003 + seed,
            "dimensions_per_regime": HIDDEN_DIM // R,
            "disjoint": True,
            "exhaustive": True,
            "active_indices": {
                str(regime): sorted(blocks[regime].tolist())
                for regime in range(R)
            },
        }

        all_permutations = list(itertools.permutations(range(N_CLASSES)))
        # Reproducible experiment shuffling, never key or nonce generation.
        permutation_rng = random.Random(  # nosec B311
            int(config["permutation_seed"]) + R * 10_007 + seed
        )
        permutation_rng.shuffle(all_permutations)
        permutations = [
            list(all_permutations[regime % len(all_permutations)])
            for regime in range(R)
        ]
        permutation_tensors = [
            torch.tensor(item, dtype=torch.long, device=device)
            for item in permutations
        ]
        inverse_permutations = [
            torch.tensor(
                [item.index(label) for label in range(N_CLASSES)],
                dtype=torch.long,
                device=device,
            )
            for item in permutations
        ]
        artifact["label_permutations"] = {
            str(regime): permutation
            for regime, permutation in enumerate(permutations)
        }

        tokenizer = DistilBertTokenizerFast.from_pretrained(
            config["model_name"], revision=config["model_revision"]
        )
        raw = datasets.load_dataset(
            config["dataset_name"], revision=config["dataset_revision"]
        )
        train_count = min(int(config["train_samples"]), len(raw["train"]))
        eval_count = min(int(config["eval_samples"]), len(raw["test"]))
        train_split = (
            raw["train"]
            .shuffle(seed=int(config["data_seed"]))
            .select(range(train_count))
        )
        eval_split = raw["test"].select(range(eval_count))

        def tokenize(split: Any) -> TensorDataset:
            encoded = tokenizer(
                split["text"],
                truncation=True,
                padding="max_length",
                max_length=int(config["max_length"]),
                return_tensors="pt",
            )
            return TensorDataset(
                encoded["input_ids"],
                encoded["attention_mask"],
                torch.tensor(split["label"], dtype=torch.long),
            )

        train_data = tokenize(train_split)
        eval_data = tokenize(eval_split)
        artifact["data"] = {
            "train_samples": len(train_data),
            "eval_samples": len(eval_data),
            "data_seed": config["data_seed"],
        }

        class Classifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bert = DistilBertModel.from_pretrained(
                    config["model_name"], revision=config["model_revision"]
                )
                self.dropout = nn.Dropout(float(config["dropout"]))
                self.classifier = nn.Linear(HIDDEN_DIM, N_CLASSES)

            def forward(
                self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                gate: torch.Tensor | None = None,
            ) -> torch.Tensor:
                pooled = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state[:, 0, :]
                if gate is not None:
                    pooled = pooled * gate
                return self.classifier(self.dropout(pooled))

        def make_optimizer(model: nn.Module) -> torch.optim.AdamW:
            return torch.optim.AdamW(
                [
                    {
                        "params": list(model.bert.parameters()),
                        "lr": float(config["backbone_lr"]),
                    },
                    {
                        "params": list(model.classifier.parameters()),
                        "lr": float(config["head_lr"]),
                    },
                ],
                weight_decay=float(config["weight_decay"]),
            )

        criterion = nn.CrossEntropyLoss()

        def loader_for_epoch(epoch: int) -> DataLoader:
            return DataLoader(
                train_data,
                batch_size=int(config["batch_size"]),
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                generator=torch.Generator().manual_seed(
                    seed + int(config["loader_seed"]) + epoch
                ),
            )

        @torch.no_grad()
        def evaluate(
            model: nn.Module,
            regime: int,
            *,
            gated: bool,
        ) -> dict[str, float | int]:
            model.eval()
            loader = DataLoader(
                eval_data,
                batch_size=int(config["eval_batch_size"]),
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
            correct = 0
            total = 0
            loss_sum = 0.0
            gate = masks[regime].unsqueeze(0) if gated else None
            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                targets = permutation_tensors[regime][labels]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids, attention_mask, gate)
                    loss = criterion(logits, targets)
                predictions = inverse_permutations[regime][logits.argmax(dim=-1)]
                correct += int((predictions == labels).sum().item())
                total += labels.numel()
                loss_sum += float(loss.item()) * labels.numel()
            return {
                "accuracy": correct / max(total, 1),
                "loss": loss_sum / max(total, 1),
                "correct": correct,
                "total": total,
            }

        def train_separate(
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            regime: int,
            steps: int,
            *,
            gated: bool,
            start_epoch: int = 0,
        ) -> dict[str, Any]:
            model.train()
            losses: list[float] = []
            completed = 0
            epoch = start_epoch
            gate = masks[regime].unsqueeze(0) if gated else None
            while completed < steps:
                for input_ids, attention_mask, labels in loader_for_epoch(epoch):
                    if completed >= steps:
                        break
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    targets = permutation_tensors[regime][labels]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(input_ids, attention_mask, gate)
                        loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.item()))
                    completed += 1
                epoch += 1
            return {
                "steps": completed,
                "mean_training_loss": sum(losses) / max(len(losses), 1),
            }

        def train_cotrained(
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            steps: int,
        ) -> dict[str, Any]:
            model.train()
            losses: list[float] = []
            completed = 0
            epoch = 0
            while completed < steps:
                for input_ids, attention_mask, labels in loader_for_epoch(epoch):
                    if completed >= steps:
                        break
                    input_ids = input_ids.to(device, non_blocking=True)
                    attention_mask = attention_mask.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    batch_loss = torch.zeros((), device=device)
                    for regime in range(R):
                        targets = permutation_tensors[regime][labels]
                        gate = masks[regime].unsqueeze(0)
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            logits = model(input_ids, attention_mask, gate)
                            batch_loss = batch_loss + criterion(logits, targets) / R
                    batch_loss.backward()
                    optimizer.step()
                    losses.append(float(batch_loss.item()))
                    completed += 1
                epoch += 1
            return {
                "optimizer_steps": completed,
                "regime_forwards": completed * R,
                "mean_training_loss": sum(losses) / max(len(losses), 1),
            }

        def summarize(metrics: list[dict[str, Any]]) -> dict[str, float]:
            accuracies = [float(item["accuracy"]) for item in metrics]
            return {
                "mean_accuracy": sum(accuracies) / len(accuracies),
                "min_accuracy": min(accuracies),
                "max_accuracy": max(accuracies),
            }

        set_seed(seed)
        initial_model = Classifier().to(device)
        resolved_model_revision = getattr(initial_model.bert.config, "_commit_hash", None)
        artifact["resolved_revisions"] = {
            "model": resolved_model_revision or config["model_revision"],
            "dataset": config["dataset_revision"],
        }
        initial_state = copy.deepcopy(initial_model.state_dict())
        del initial_model
        reset_device()

        steps_per_epoch = math.ceil(len(train_data) / int(config["batch_size"]))
        base_steps = steps_per_epoch * int(config["epochs"])
        artifact["training_budget"] = {
            "steps_per_epoch": steps_per_epoch,
            "base_optimizer_steps": base_steps,
            "continued_steps": {
                str(int(fraction * 100)): max(1, math.ceil(base_steps * fraction))
                for fraction in ADAPTATION_FRACTIONS
            },
        }
        persist()

        # Ungated controls are retained in CPU memory so every post-hoc and
        # continued condition starts from the exact same trained checkpoint.
        ungated_checkpoints: list[dict[str, Any]] = []
        ungated_metrics: list[dict[str, Any]] = []
        condition_started = time.time()
        for regime in range(R):
            set_seed(seed)
            model = Classifier().to(device)
            model.load_state_dict(initial_state)
            optimizer = make_optimizer(model)
            training = train_separate(
                model, optimizer, regime, base_steps, gated=False
            )
            metric = evaluate(model, regime, gated=False)
            metric.update({"regime": regime, "training": training})
            ungated_metrics.append(metric)
            ungated_checkpoints.append(
                {
                    "model": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                    "optimizer": copy.deepcopy(optimizer.state_dict()),
                }
            )
            del model, optimizer
            reset_device()
        artifact["conditions"]["ungated_separate"] = {
            "runtime_seconds": time.time() - condition_started,
            "per_regime": ungated_metrics,
            "summary": summarize(ungated_metrics),
        }
        persist()

        condition_started = time.time()
        frozen_metrics: list[dict[str, Any]] = []
        for regime, checkpoint in enumerate(ungated_checkpoints):
            model = Classifier().to(device)
            model.load_state_dict(checkpoint["model"])
            metric = evaluate(model, regime, gated=True)
            metric["regime"] = regime
            frozen_metrics.append(metric)
            del model
            reset_device()
        artifact["conditions"]["frozen_posthoc_gate"] = {
            "runtime_seconds": time.time() - condition_started,
            "per_regime": frozen_metrics,
            "summary": summarize(frozen_metrics),
        }
        persist()

        for fraction in ADAPTATION_FRACTIONS:
            percent = int(fraction * 100)
            condition_name = f"continued_gate_aware_{percent}pct"
            condition_started = time.time()
            continued_metrics: list[dict[str, Any]] = []
            continuation_steps = max(1, math.ceil(base_steps * fraction))
            for regime, checkpoint in enumerate(ungated_checkpoints):
                set_seed(seed)
                model = Classifier().to(device)
                model.load_state_dict(checkpoint["model"])
                optimizer = make_optimizer(model)
                optimizer.load_state_dict(checkpoint["optimizer"])
                training = train_separate(
                    model,
                    optimizer,
                    regime,
                    continuation_steps,
                    gated=True,
                    start_epoch=int(config["epochs"]),
                )
                metric = evaluate(model, regime, gated=True)
                metric.update({"regime": regime, "training": training})
                continued_metrics.append(metric)
                del model, optimizer
                reset_device()
            artifact["conditions"][condition_name] = {
                "adaptation_fraction": fraction,
                "runtime_seconds": time.time() - condition_started,
                "per_regime": continued_metrics,
                "summary": summarize(continued_metrics),
            }
            persist()

        # Free CPU copies before the final co-training condition.
        del ungated_checkpoints
        reset_device()

        condition_started = time.time()
        set_seed(seed)
        cotrained_model = Classifier().to(device)
        cotrained_model.load_state_dict(initial_state)
        cotrained_optimizer = make_optimizer(cotrained_model)
        cotrained_training = train_cotrained(
            cotrained_model, cotrained_optimizer, base_steps
        )
        cotrained_metrics = [
            {
                "regime": regime,
                **evaluate(cotrained_model, regime, gated=True),
            }
            for regime in range(R)
        ]
        artifact["conditions"]["cotrained_gate_aware"] = {
            "runtime_seconds": time.time() - condition_started,
            "training": cotrained_training,
            "per_regime": cotrained_metrics,
            "summary": summarize(cotrained_metrics),
        }

        artifact["status"] = "complete"
        artifact["completed_at"] = _utc_now()
        persist()
        return json.loads(json.dumps(_json_safe({
            "status": artifact["status"],
            "artifact_name": artifact_name,
            "job": artifact["job"],
            "elapsed_seconds": artifact["elapsed_seconds"],
            "conditions": artifact["conditions"],
            "environment": artifact["environment"],
            "resolved_revisions": artifact["resolved_revisions"],
        }), default=str))
    except BaseException as exc:
        artifact["status"] = "failed"
        artifact["failed_at"] = _utc_now()
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            persist()
        except BaseException as persist_exc:
            print(
                f"CRITICAL: failed to persist {artifact_name}: {persist_exc}",
                flush=True,
            )
        raise


def _t_critical_95(sample_count: int) -> float:
    # Two-sided 95% Student-t critical values for df=1..30.
    values = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
    }
    return values.get(sample_count - 1, 1.96)


def _paired_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    condition_names = (
        "frozen_posthoc_gate",
        "continued_gate_aware_10pct",
        "continued_gate_aware_25pct",
        "continued_gate_aware_50pct",
        "cotrained_gate_aware",
    )
    by_r: dict[str, Any] = {}
    for R in sorted({int(result["job"]["R"]) for result in results}):
        rows = [result for result in results if int(result["job"]["R"]) == R]
        control = [
            float(row["conditions"]["ungated_separate"]["summary"]["mean_accuracy"])
            for row in rows
        ]
        condition_summary: dict[str, Any] = {}
        for name in condition_names:
            values = [
                float(row["conditions"][name]["summary"]["mean_accuracy"])
                for row in rows
            ]
            differences = [
                condition - baseline
                for condition, baseline in zip(values, control, strict=True)
            ]
            mean = sum(differences) / len(differences)
            if len(differences) > 1:
                variance = sum((item - mean) ** 2 for item in differences) / (
                    len(differences) - 1
                )
                half_width = _t_critical_95(len(differences)) * math.sqrt(
                    variance / len(differences)
                )
                confidence_interval: list[float] | None = [
                    mean - half_width,
                    mean + half_width,
                ]
            else:
                confidence_interval = None
            condition_summary[name] = {
                "paired_difference_definition": "condition - ungated_separate",
                "n_seeds": len(differences),
                "mean_difference": mean,
                "confidence_interval_95": confidence_interval,
                "per_seed_differences": differences,
            }
        by_r[str(R)] = {
            "seeds": [int(row["job"]["seed"]) for row in rows],
            "ungated_separate_mean_accuracy": sum(control) / len(control),
            "conditions": condition_summary,
        }
    return by_r


@app.local_entrypoint()
def main(
    r_values: str = "4,8,16",
    seeds: str = "42,123,256,789,1337",
    smoke: bool = False,
    train_samples: int = 8_000,
    eval_samples: int = 7_600,
    epochs: int = 2,
    batch_size: int = 48,
    eval_batch_size: int = 96,
    max_length: int = 128,
    backbone_lr: float = 2e-5,
    head_lr: float = 1e-3,
    weight_decay: float = 0.01,
    dropout: float = 0.1,
    model_revision: str = MODEL_REVISION,
    dataset_revision: str = DATASET_REVISION,
) -> None:
    """Dispatch paired jobs and save one local combined artifact."""
    for name, revision_value in {
        "model_revision": model_revision,
        "dataset_revision": dataset_revision,
    }.items():
        if re.fullmatch(r"[0-9a-f]{40}", revision_value) is None:
            raise ValueError(f"{name} must be an immutable lowercase Git commit")
    selected_r = _parse_int_csv(r_values, name="r_values")
    selected_seeds = _parse_int_csv(seeds, name="seeds")
    if smoke:
        selected_r = selected_r[:1]
        selected_seeds = selected_seeds[:1]
        train_samples = min(train_samples, 192)
        eval_samples = min(eval_samples, 192)
        epochs = 1
        batch_size = min(batch_size, 16)
        eval_batch_size = min(eval_batch_size, 32)

    unsupported = sorted(set(selected_r) - set(DEFAULT_R_VALUES))
    if unsupported:
        raise ValueError(
            f"r_values must be drawn from {DEFAULT_R_VALUES}; got {unsupported}"
        )
    for name, value in {
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "max_length": max_length,
    }.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    revision = _git_revision()
    run_id = _timestamp()
    common_config: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "model_revision": model_revision,
        "dataset_name": DATASET_NAME,
        "dataset_revision": dataset_revision,
        "train_samples": train_samples,
        "eval_samples": eval_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "max_length": max_length,
        "backbone_lr": backbone_lr,
        "head_lr": head_lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "mask_seed": 91_337,
        "permutation_seed": 27_271,
        "data_seed": 42,
        "loader_seed": 81_919,
        "adaptation_fractions": list(ADAPTATION_FRACTIONS),
        "git_revision": revision,
        "run_id": run_id,
        "smoke": smoke,
    }
    job_configs = [
        {**common_config, "R": R, "seed": seed}
        for R in selected_r
        for seed in selected_seeds
    ]

    launched_at = _utc_now()
    started = time.time()
    print(
        f"Dispatching {len(job_configs)} paired jobs on L4: "
        f"R={selected_r}, seeds={selected_seeds}, smoke={smoke}",
        flush=True,
    )
    handles = [
        (job_config, run_paired_job.spawn(job_config))
        for job_config in job_configs
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job_config, handle in handles:
        try:
            results.append(handle.get())
        except BaseException as exc:
            failure = {
                "status": "failed",
                "job": {"R": job_config["R"], "seed": job_config["seed"]},
                "artifact_name": (
                    f"matched_adaptation_R{job_config['R']}_"
                    f"seed{job_config['seed']}_{run_id}.json"
                ),
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
            failures.append(failure)
            print(
                f"Job R={job_config['R']} seed={job_config['seed']} failed; "
                f"remote failure artifact: {failure['artifact_name']}",
                flush=True,
            )
    results.sort(key=lambda item: (int(item["job"]["R"]), int(item["job"]["seed"])))

    combined = {
        "schema_version": 1,
        "experiment": "matched_distilbert_adaptation",
        "status": "partial_failure" if failures else "complete",
        "launched_at": launched_at,
        "completed_at": _utc_now(),
        "elapsed_seconds": time.time() - started,
        "git_revision": revision,
        "volume": {
            "name": VOLUME_NAME,
            "path": VOLUME_PATH,
            "remote_artifacts": [
                result["artifact_name"] for result in results + failures
            ],
        },
        "config": {
            **common_config,
            "r_values": selected_r,
            "seeds": selected_seeds,
        },
        "paired_summary": _paired_summary(results),
        "jobs": results + failures,
        "local_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"matched_adaptation_{_timestamp()}.json"
    output_path.write_text(json.dumps(_json_safe(combined), indent=2) + "\n")
    print(f"Combined results saved to {output_path}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} remote job(s) failed")
