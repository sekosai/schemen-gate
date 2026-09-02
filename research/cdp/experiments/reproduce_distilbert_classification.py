"""Reproduce the DistilBERT classification experiments without runtime IP.

Two gate surfaces are deliberately separated:

``legacy-post-encoder``
    Reproduces the formative Series 1 protocol that generated the archived
    five-seed results.  It masks the final 768-dimensional CLS vector after the
    encoder.  These numbers are not intermediate-FFN evidence.

``intermediate-ffn``
    Implements the paper-aligned surface by masking each DistilBERT FFN's
    expanded activation at the input to ``lin2``.  This is the corrected
    reproduction path and writes new artifacts; it must not be presented as a
    reproduction of the archived legacy numbers.

This file contains model-training code only. Balanced masks come from the
exact locked ``schemen-gate`` revision; production key custody, lockbox,
sidecar, and trusted-runtime code remain outside this training process.

Examples:
    python experiments/reproduce_distilbert_classification.py --smoke
    python experiments/reproduce_distilbert_classification.py \
        --surface legacy-post-encoder --r-values 8,16,24,32,64,96,128
    python experiments/reproduce_distilbert_classification.py \
        --surface intermediate-ffn --r-values 8,16,24,32,64,96,128
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import platform
import random
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from library_provenance import collect_library_provenance, collect_source_provenance
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from schemen_gate import GateMask

MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
N_CLASSES = 4


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def label_permutations(regimes: int) -> list[tuple[int, ...]]:
    permutations = list(itertools.permutations(range(N_CLASSES)))
    np.random.default_rng(42).shuffle(permutations)
    return [permutations[r % len(permutations)] for r in range(regimes)]


def load_ag_news(
    *,
    train_examples: int,
    test_examples: int,
    max_length: int,
) -> tuple[TensorDataset, TensorDataset]:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION)
    train = dataset["train"].shuffle(seed=42).select(range(train_examples))
    test_count = min(test_examples, len(dataset["test"]))
    test = dataset["test"].select(range(test_count))

    def encode(split) -> TensorDataset:
        encoded = tokenizer(
            list(split["text"]),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(list(split["label"]), dtype=torch.long),
        )

    return encode(train), encode(test)


class SeparateDistilBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, N_CLASSES))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        return self.classifier(cls)


class GatedDistilBert(nn.Module):
    def __init__(self, surface: str) -> None:
        super().__init__()
        from transformers import AutoModel

        if surface not in {"legacy-post-encoder", "intermediate-ffn"}:
            raise ValueError(f"unknown gate surface: {surface}")
        self.surface = surface
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
        )
        self.classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, N_CLASSES))
        self._active_mask: torch.Tensor | None = None
        self._hooks: list[Any] = []

        if surface == "intermediate-ffn":
            for layer in self.encoder.transformer.layer:
                self._hooks.append(
                    layer.ffn.lin2.register_forward_pre_hook(self._mask_lin2_input)
                )

    @property
    def gate_dimensions(self) -> int:
        if self.surface == "legacy-post-encoder":
            return int(self.encoder.config.dim)
        return int(self.encoder.config.hidden_dim)

    def _mask_lin2_input(self, _module, inputs):
        if self._active_mask is None:
            raise RuntimeError("intermediate FFN hook invoked without an active mask")
        activation = inputs[0]
        mask = self._active_mask
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        return (activation * mask[:, None, :].to(activation.dtype),)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        gate_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._active_mask = gate_mask
        try:
            cls = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state[:, 0, :]
        finally:
            self._active_mask = None

        if self.surface == "legacy-post-encoder":
            cls = cls * gate_mask.to(cls.dtype)
        return self.classifier(cls)


def make_loaders(
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    *,
    batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def optimizer_for(model: nn.Module, *, encoder_lr: float, head_lr: float):
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ],
        weight_decay=0.01,
    )


def train_separate(
    model: SeparateDistilBert,
    loader: DataLoader,
    *,
    permutation: tuple[int, ...],
    epochs: int,
    encoder_lr: float,
    head_lr: float,
    device: torch.device,
) -> None:
    model.train()
    optimizer = optimizer_for(model, encoder_lr=encoder_lr, head_lr=head_lr)
    permutation_tensor = torch.tensor(permutation, device=device)
    for _ in range(epochs):
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = permutation_tensor[labels.to(device)]
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device):
                loss = F.cross_entropy(model(input_ids, attention_mask), labels)
            loss.backward()
            optimizer.step()


def train_gated(
    model: GatedDistilBert,
    loader: DataLoader,
    *,
    masks: list[torch.Tensor],
    permutations: list[tuple[int, ...]],
    epochs: int,
    encoder_lr: float,
    head_lr: float,
    device: torch.device,
) -> None:
    model.train()
    optimizer = optimizer_for(model, encoder_lr=encoder_lr, head_lr=head_lr)
    permutation_tensors = [
        torch.tensor(permutation, device=device) for permutation in permutations
    ]
    for _ in range(epochs):
        for input_ids, attention_mask, labels in loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            for regime, mask in enumerate(masks):
                batch_mask = mask.unsqueeze(0).expand(input_ids.shape[0], -1)
                with amp_context(device):
                    logits = model(input_ids, attention_mask, batch_mask)
                    loss = (
                        F.cross_entropy(logits, permutation_tensors[regime][labels])
                        / len(masks)
                    )
                loss.backward()
            optimizer.step()


@torch.no_grad()
def evaluate_separate(
    model: SeparateDistilBert,
    loader: DataLoader,
    *,
    permutation: tuple[int, ...],
    device: torch.device,
) -> float:
    model.eval()
    inverse = torch.tensor(
        [permutation.index(label) for label in range(N_CLASSES)],
        device=device,
    )
    correct = total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        with amp_context(device):
            predictions = inverse[model(input_ids, attention_mask).argmax(-1)]
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


@torch.no_grad()
def evaluate_gated(
    model: GatedDistilBert,
    loader: DataLoader,
    *,
    mask: torch.Tensor,
    owning_permutation: tuple[int, ...],
    device: torch.device,
) -> float:
    model.eval()
    inverse = torch.tensor(
        [owning_permutation.index(label) for label in range(N_CLASSES)],
        device=device,
    )
    correct = total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        batch_mask = mask.unsqueeze(0).expand(input_ids.shape[0], -1)
        with amp_context(device):
            predictions = inverse[
                model(input_ids, attention_mask, batch_mask).argmax(-1)
            ]
        correct += int((predictions == labels).sum())
        total += labels.numel()
    return correct / total


def summarize_pairs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    separate = np.asarray([row["separate_mean"] for row in rows], dtype=float)
    gated = np.asarray([row["gated_mean"] for row in rows], dtype=float)
    gaps = separate - gated
    if len(rows) < 2:
        return {
            "separate_mean": float(separate.mean()),
            "gated_mean": float(gated.mean()),
            "gap_fraction": float(gaps.mean()),
            "gap_percentage_points": float(100 * gaps.mean()),
            "paired_p_value": None,
            "student_t_95_ci_fraction": None,
        }

    t_statistic, p_value = stats.ttest_rel(separate, gated)
    standard_error = stats.sem(gaps)
    interval = stats.t.interval(
        0.95,
        df=len(rows) - 1,
        loc=float(gaps.mean()),
        scale=float(standard_error),
    )
    return {
        "separate_mean": float(separate.mean()),
        "gated_mean": float(gated.mean()),
        "gap_fraction": float(gaps.mean()),
        "gap_percentage_points": float(100 * gaps.mean()),
        "paired_t_statistic": float(t_statistic),
        "paired_p_value": float(p_value),
        "student_t_95_ci_fraction": [float(interval[0]), float(interval[1])],
        "student_t_95_ci_percentage_points": [
            float(100 * interval[0]),
            float(100 * interval[1]),
        ],
    }


def run_ratio(
    *,
    ratio: int,
    seeds: list[int],
    args: argparse.Namespace,
    train_dataset: TensorDataset,
    test_dataset: TensorDataset,
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        print(f"R={ratio} seed={seed}: starting", flush=True)
        set_seed(seed)
        permutations = label_permutations(ratio)

        probe = GatedDistilBert(args.surface).to(device)
        dimensions = probe.gate_dimensions
        del probe
        mask_key = hashlib.sha256(
            f"cdp-distilbert:{args.surface}:{ratio}:{seed}".encode()
        ).digest()
        masks = [
            GateMask.derive(
                mask_key,
                regime,
                n_dims=dimensions,
                n_regimes=ratio,
            ).to_torch(device=device, dtype=torch.float32)
            for regime in range(ratio)
        ]
        train_loader, test_loader = make_loaders(
            train_dataset,
            test_dataset,
            batch_size=args.batch_size,
            seed=seed,
        )

        separate_accuracies = []
        for regime in range(args.separate_controls):
            set_seed(seed + regime * 1000)
            model = SeparateDistilBert().to(device)
            train_separate(
                model,
                train_loader,
                permutation=permutations[regime],
                epochs=args.epochs,
                encoder_lr=args.encoder_lr,
                head_lr=args.head_lr,
                device=device,
            )
            separate_accuracies.append(
                evaluate_separate(
                    model,
                    test_loader,
                    permutation=permutations[regime],
                    device=device,
                )
            )
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        set_seed(seed)
        gated = GatedDistilBert(args.surface).to(device)
        train_gated(
            gated,
            train_loader,
            masks=masks,
            permutations=permutations,
            epochs=args.epochs,
            encoder_lr=args.encoder_lr,
            head_lr=args.head_lr,
            device=device,
        )
        owning_accuracies = [
            evaluate_gated(
                gated,
                test_loader,
                mask=masks[regime],
                owning_permutation=permutations[regime],
                device=device,
            )
            for regime in range(args.separate_controls)
        ]
        wrong_key_accuracies = [
            {
                "owner": owner,
                "applied_mask": wrong,
                "accuracy": evaluate_gated(
                    gated,
                    test_loader,
                    mask=masks[wrong],
                    owning_permutation=permutations[owner],
                    device=device,
                ),
            }
            for owner in range(args.separate_controls)
            for wrong in range(args.separate_controls)
            if owner != wrong
        ]
        row = {
            "seed": seed,
            "gate_dimensions": dimensions,
            "dimensions_per_regime": dimensions // ratio,
            "separate_accuracies": separate_accuracies,
            "gated_owning_accuracies": owning_accuracies,
            "wrong_key_accuracies": wrong_key_accuracies,
            "separate_mean": float(np.mean(separate_accuracies)),
            "gated_mean": float(np.mean(owning_accuracies)),
            "wrong_key_mean": float(
                np.mean([item["accuracy"] for item in wrong_key_accuracies])
            ),
        }
        rows.append(row)
        print(
            f"R={ratio} seed={seed}: separate={row['separate_mean']:.4f} "
            f"gated={row['gated_mean']:.4f} wrong={row['wrong_key_mean']:.4f}",
            flush=True,
        )
        del gated
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "R": ratio,
        "surface": args.surface,
        "paired_runs": rows,
        "summary": summarize_pairs(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--surface",
        choices=["legacy-post-encoder", "intermediate-ffn"],
        default="legacy-post-encoder",
    )
    parser.add_argument("--r-values", default="8,16,24,32,64,96,128")
    parser.add_argument("--seeds", default="42,123,256,456,789")
    parser.add_argument("--train-examples", type=int, default=8000)
    parser.add_argument("--test-examples", type=int, default=7600)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--separate-controls", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.r_values = "8"
        args.seeds = "42"
        args.train_examples = 128
        args.test_examples = 128
        args.batch_size = 16
        args.epochs = 1
        args.separate_controls = 2

    ratios = [int(value) for value in args.r_values.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    device = resolve_device(args.device)
    started = time.time()
    print(f"device={device} surface={args.surface}", flush=True)

    train_dataset, test_dataset = load_ag_news(
        train_examples=args.train_examples,
        test_examples=args.test_examples,
        max_length=args.max_length,
    )
    results = [
        run_ratio(
            ratio=ratio,
            seeds=seeds,
            args=args,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
        )
        for ratio in ratios
    ]
    artifact = {
        "schema_version": 2,
        "experiment": "distilbert_classification_gate_reproduction",
        "status": "complete",
        "surface": args.surface,
        "surface_evidence": (
            "formative legacy post-encoder classification gate"
            if args.surface == "legacy-post-encoder"
            else "paper-aligned intermediate FFN gate; new evidence"
        ),
        "started_at_utc": datetime.fromtimestamp(
            started, timezone.utc
        ).isoformat(),
        "elapsed_seconds": time.time() - started,
        "git_revision": git_revision(),
        "source": collect_source_provenance(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "libraries": collect_library_provenance(),
        },
        "mask_contract": {
            "implementation": "schemen_gate.GateMask.derive",
            "key_derivation": "SHA-256(surface, R, seed)",
        },
        "assets": {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
        },
        "config": vars(args),
        "results": results,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"distilbert_classification_{args.surface}_{utc_timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
