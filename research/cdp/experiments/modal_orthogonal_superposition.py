"""Execution-authorized reproduction of exact orthogonal model placement.

This is whole-model activation, not sparse FFN partitioning. A trained
DistilBERT classifier is conjugated into permutation bases and evaluated only
after the source-visible research preflight authorizes the model/dimension/regime
request. Accuracy must be exactly unchanged; logit drift is reported
separately as a finite-precision diagnostic.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal
from library_provenance import collect_experiment_provenance
from modal_schemen_image import (
    assert_remote_schemen_versions,
    install_current_schemen,
)

APP_NAME = "cdp-orthogonal-superposition"
VOLUME_NAME = "cdp-runtime-orthogonal-results"
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = install_current_schemen(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.13.0",
        "transformers==5.10.1",
        "datasets==5.0.1",
        "numpy==2.4.6",
    ),
    launcher=Path(__file__),
).add_local_file(
    Path(__file__).with_name("orthogonal_superposition_sweep.py"),
    "/root/orthogonal_superposition_sweep.py",
    copy=True,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def source_metadata() -> dict[str, Any]:
    return collect_experiment_provenance(Path(__file__))


@app.function(
    max_containers=3,
    image=image,
    gpu="A100",
    timeout=30 * 60,
    volumes={"/results": results_volume},
)
def run_reproduction(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    import copy
    import gc
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from execution_preflight import GateExecutionPreflight
    from orthogonal_superposition_sweep import conjugate_residual_basis
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    source = assert_remote_schemen_versions(
        source,
        launcher_name=Path(__file__).name,
    )
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION)
    train_split = dataset["train"].shuffle(seed=42).select(
        range(int(config["train_examples"]))
    )
    test_split = dataset["test"].select(range(int(config["test_examples"])))

    def encode(split) -> TensorDataset:
        encoded = tokenizer(
            list(split["text"]),
            padding="max_length",
            truncation=True,
            max_length=int(config["max_length"]),
            return_tensors="pt",
        )
        return TensorDataset(
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(list(split["label"]), dtype=torch.long),
        )

    train_data = encode(train_split)
    test_data = encode(test_split)

    def loader(data, *, shuffle: bool, salt: int = 0):
        return DataLoader(
            data,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle,
            generator=(torch.Generator().manual_seed(seed + salt) if shuffle else None),
        )

    backbone = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION).to(device)
    classifier = nn.Linear(768, 4).to(device)
    optimizer = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()),
        lr=float(config["learning_rate"]),
    )
    for _ in range(int(config["epochs"])):
        backbone.train()
        classifier.train()
        for ids, attention, labels in loader(train_data, shuffle=True, salt=17):
            ids = ids.to(device)
            attention = attention.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = classifier(
                backbone(input_ids=ids, attention_mask=attention)
                .last_hidden_state[:, 0, :]
            )
            F.cross_entropy(logits, labels).backward()
            optimizer.step()

    @torch.no_grad()
    def evaluate(eval_backbone, eval_classifier) -> tuple[float, torch.Tensor]:
        eval_backbone.eval()
        eval_classifier.eval()
        logits_rows = []
        correct = total = 0
        for ids, attention, labels in loader(test_data, shuffle=False):
            ids = ids.to(device)
            attention = attention.to(device)
            labels = labels.to(device)
            logits = eval_classifier(
                eval_backbone(input_ids=ids, attention_mask=attention)
                .last_hidden_state[:, 0, :]
            )
            logits_rows.append(logits.cpu())
            correct += int((logits.argmax(-1) == labels).sum())
            total += labels.numel()
        return correct / total, torch.cat(logits_rows)

    baseline_accuracy, baseline_logits = evaluate(backbone, classifier)
    ratio_results = []
    for ratio in config["ratios"]:
        ratio = int(ratio)
        runtime = GateExecutionPreflight(
            model_id=f"{MODEL_NAME}:orthogonal-conjugation",
            dimensions=768,
            authorized_regime_ids=list(range(ratio)),
        )
        accuracies = []
        maximum_logit_difference = 0.0
        for regime in range(ratio):
            permutation = np.random.default_rng(
                seed + ratio * 100_000 + regime
            ).permutation(768)
            placed_backbone = copy.deepcopy(backbone)
            placed_classifier = copy.deepcopy(classifier)
            conjugate_residual_basis(
                placed_backbone,
                placed_classifier,
                permutation,
                device,
            )
            accuracy, logits = runtime.invoke(
                regime,
                lambda placed_backbone=placed_backbone,
                placed_classifier=placed_classifier: evaluate(
                    placed_backbone,
                    placed_classifier,
                ),
            )
            accuracies.append(accuracy)
            maximum_logit_difference = max(
                maximum_logit_difference,
                float((logits - baseline_logits).abs().max()),
            )
            del placed_backbone, placed_classifier, logits
            gc.collect()
            torch.cuda.empty_cache()

        rejection_probe = runtime.rejection_probe()
        maximum_accuracy_gap = max(
            abs(accuracy - baseline_accuracy) for accuracy in accuracies
        )
        ratio_results.append(
            {
                "R": ratio,
                "evaluated_regimes": ratio,
                "baseline_accuracy": baseline_accuracy,
                "mean_accuracy": float(np.mean(accuracies)),
                "minimum_accuracy": min(accuracies),
                "maximum_accuracy": max(accuracies),
                "maximum_absolute_accuracy_gap": maximum_accuracy_gap,
                "maximum_absolute_logit_difference": maximum_logit_difference,
                "accuracy_zero_loss": maximum_accuracy_gap == 0.0,
                "finite_precision_logit_diagnostic": (
                    "Reported but not part of the pass predicate. Permuting "
                    "fp32 reduction order can change low-order logits even "
                    "when every prediction and the measured accuracy are exact."
                ),
                "runtime": {
                    **runtime.evidence(),
                    "rejection_probe": rejection_probe,
                },
                "status": (
                    "pass"
                    if maximum_accuracy_gap == 0.0
                    and rejection_probe["all_rejected"]
                    and rejection_probe["unauthorized_model_calls"] == 0
                    else "failure"
                ),
            }
        )

    result = {
        "schema_version": 1,
        "experiment": "orthogonal_superposition",
        "scope": (
            "Whole-model permutation conjugation and serial addressed use; "
            "not sparse FFN capacity partitioning or concurrent cotenancy."
        ),
        "status": (
            "pass"
            if all(row["status"] == "pass" for row in ratio_results)
            else "failure"
        ),
        "config": config,
        "source": source,
        "assets": {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
        },
        "gate_placement": "gate-authorized-whole-model-activation",
        "ratios": ratio_results,
    }
    remote_path = Path("/results") / f"orthogonal_{timestamp()}.json"
    remote_path.write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    result["remote_artifact"] = str(remote_path)
    return result


@app.local_entrypoint()
def main(
    ratios: str = "8,128",
    smoke: bool = False,
) -> None:
    selected_ratios = [int(value) for value in ratios.split(",") if value]
    if smoke:
        selected_ratios = [8]
    config = {
        "seed": 0,
        "ratios": selected_ratios,
        "train_examples": 256 if smoke else 8000,
        "test_examples": 256 if smoke else 7600,
        "max_length": 64 if smoke else 128,
        "batch_size": 64,
        "epochs": 1 if smoke else 2,
        "learning_rate": 2e-5,
        "smoke": smoke,
    }
    started = time.perf_counter()
    source = source_metadata()
    result = run_reproduction.remote(config, source)
    artifact = {
        "schema_version": 1,
        "experiment": "orthogonal_superposition_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "source": source,
        "result": result,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"orthogonal_superposition_{timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
