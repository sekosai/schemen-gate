"""Dense intermediate-FFN cotenancy on a real frozen Transformer.

Only the active rows of every DistilBERT ``lin1`` and matching columns of
``lin2`` may update for a tenant.  Attention, embeddings, normalization,
residual paths, and all other shared parameters are frozen.  Each tenant also
has a private classifier selected by the same regime authorization.

This differs materially from jointly fine-tuning the whole encoder under an
activation mask: joint fine-tuning lets tenant data update shared attention and
normalization parameters and therefore cannot guarantee state separation.

    modal run experiments/modal_dense_ffn_cotenancy.py --smoke
    modal run experiments/modal_dense_ffn_cotenancy.py \
      --r-values 1,2,4,8,16 --seeds 42,123,256
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
    verified_gate_mask_algorithm_identity,
)

APP_NAME = "cdp-dense-ffn-cotenancy"
VOLUME_NAME = "cdp-dense-ffn-cotenancy-results"
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
MAX_REMOTE_CONCURRENCY = 3

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
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def source_metadata() -> dict[str, Any]:
    return collect_experiment_provenance(Path(__file__))


@app.function(
    max_containers=3,
    image=image,
    gpu="A100",
    timeout=10 * 60,
    volumes={"/results": results_volume},
)
def run_ratio(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    import gc
    import itertools
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from execution_preflight import GateExecutionPreflight
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    from schemen_gate import GateMask

    source = assert_remote_schemen_versions(
        source,
        launcher_name=Path(__file__).name,
    )

    seed = int(config["seed"])
    regimes = int(config["R"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION)
    train = dataset["train"].shuffle(seed=42).select(
        range(int(config["train_examples"]))
    )
    test = dataset["test"].select(range(int(config["test_examples"])))

    def encode(split) -> TensorDataset:
        batch = tokenizer(
            list(split["text"]),
            padding="max_length",
            truncation=True,
            max_length=int(config["max_length"]),
            return_tensors="pt",
        )
        return TensorDataset(
            batch["input_ids"],
            batch["attention_mask"],
            torch.tensor(list(split["label"]), dtype=torch.long),
        )

    train_dataset = encode(train)
    test_dataset = encode(test)

    def balanced_masks(dimensions: int) -> list[torch.Tensor]:
        key = seed.to_bytes(32, "big", signed=False)
        return [
            torch.as_tensor(
                GateMask.derive(key, regime, dimensions, regimes).to_numpy(),
                dtype=torch.bool,
                device=device,
            )
            for regime in range(regimes)
        ]

    class DenseCotenancyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
            )
            for parameter in self.encoder.parameters():
                parameter.requires_grad_(False)
            for layer in self.encoder.transformer.layer:
                layer.ffn.lin1.weight.requires_grad_(True)
                layer.ffn.lin1.bias.requires_grad_(True)
                layer.ffn.lin2.weight.requires_grad_(True)
            self.classifiers = nn.ModuleList(
                [nn.Linear(768, 4) for _ in range(regimes)]
            )
            self._active_mask: torch.Tensor | None = None
            self._hooks = [
                layer.ffn.lin2.register_forward_pre_hook(self._mask_lin2_input)
                for layer in self.encoder.transformer.layer
            ]

        def _mask_lin2_input(self, _module, inputs):
            if self._active_mask is None:
                raise RuntimeError("FFN invoked without an active regime mask")
            return (
                inputs[0] * self._active_mask.to(inputs[0].dtype),
            )

        def forward(self, ids, attention, regime: int, mask: torch.Tensor):
            self._active_mask = mask
            try:
                cls = self.encoder(
                    input_ids=ids,
                    attention_mask=attention,
                ).last_hidden_state[:, 0, :]
            finally:
                self._active_mask = None
            return self.classifiers[regime](cls)

    class ScopedAdam:
        def __init__(self, entries) -> None:
            self.entries = list(entries)
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.steps = 0
            self.state = {
                id(parameter): (
                    torch.zeros_like(parameter),
                    torch.zeros_like(parameter),
                )
                for parameter, _authorized, _learning_rate in self.entries
            }

        def zero_grad(self) -> None:
            for parameter, _authorized, _learning_rate in self.entries:
                parameter.grad = None

        @torch.no_grad()
        def step(self) -> None:
            self.steps += 1
            correction1 = 1 - self.beta1**self.steps
            correction2 = 1 - self.beta2**self.steps
            for parameter, authorized, learning_rate in self.entries:
                if parameter.grad is None:
                    continue
                first, second = self.state[id(parameter)]
                gradient = parameter.grad
                first[authorized] = (
                    self.beta1 * first[authorized]
                    + (1 - self.beta1) * gradient[authorized]
                )
                second[authorized] = (
                    self.beta2 * second[authorized]
                    + (1 - self.beta2) * gradient[authorized].square()
                )
                estimate = first[authorized] / correction1
                variance = second[authorized] / correction2
                parameter[authorized] -= (
                    learning_rate
                    * estimate
                    / (variance.sqrt() + self.epsilon)
                )

    def authorized_entries(model, active, regime):
        entries = []
        for layer in model.encoder.transformer.layer:
            entries.extend(
                [
                    (
                        layer.ffn.lin1.weight,
                        active[:, None].expand_as(layer.ffn.lin1.weight),
                        float(config["ffn_learning_rate"]),
                    ),
                    (
                        layer.ffn.lin1.bias,
                        active,
                        float(config["ffn_learning_rate"]),
                    ),
                    (
                        layer.ffn.lin2.weight,
                        active[None, :].expand_as(layer.ffn.lin2.weight),
                        float(config["ffn_learning_rate"]),
                    ),
                ]
            )
        for parameter in model.classifiers[regime].parameters():
            entries.append(
                (
                    parameter,
                    torch.ones_like(parameter, dtype=torch.bool),
                    float(config["head_learning_rate"]),
                )
            )
        return entries

    def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }

    def maximum_delta(before, module) -> float:
        if not before:
            return 0.0
        return max(
            float(
                (
                    parameter.detach().cpu()
                    - before[name]
                ).abs().max()
            )
            for name, parameter in module.named_parameters()
        )

    permutations = list(itertools.permutations(range(4)))
    np.random.default_rng(42).shuffle(permutations)
    permutations = permutations[:regimes]
    masks = balanced_masks(3072)
    model = DenseCotenancyModel().to(device)
    runtime = GateExecutionPreflight(
        model_id=MODEL_NAME,
        dimensions=3072,
        authorized_regime_ids=list(range(regimes)),
    )

    frozen_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.encoder.named_parameters()
        if not parameter.requires_grad
    }
    maximum_frozen_delta = 0.0
    maximum_off_partition_delta = 0.0
    maximum_off_partition_moment = 0.0
    maximum_inactive_classifier_delta = 0.0

    for regime in range(regimes):
        active = masks[regime]
        entries = authorized_entries(model, active, regime)
        optimizer = ScopedAdam(entries)
        ffn_before = {
            name: parameter.detach().clone()
            for name, parameter in model.encoder.named_parameters()
            if parameter.requires_grad
        }
        inactive_classifiers = {
            other: snapshot(model.classifiers[other])
            for other in range(regimes)
            if other != regime
        }
        loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + regime),
        )
        permutation = torch.tensor(permutations[regime], device=device)
        model.train()
        for _ in range(int(config["epochs"])):
            for ids, attention, labels in loader:
                ids = ids.to(device)
                attention = attention.to(device)
                labels = permutation[labels.to(device)]
                optimizer.zero_grad()
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(ids, attention, regime, active)
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

        for name, parameter in model.encoder.named_parameters():
            if not parameter.requires_grad:
                maximum_frozen_delta = max(
                    maximum_frozen_delta,
                    float(
                        (
                            parameter.detach().cpu()
                            - frozen_before[name]
                        ).abs().max()
                    ),
                )
                continue
            if "lin1.weight" in name:
                authorized = active[:, None].expand_as(parameter)
            elif "lin1.bias" in name:
                authorized = active
            elif "lin2.weight" in name:
                authorized = active[None, :].expand_as(parameter)
            else:
                raise AssertionError(f"unexpected trainable FFN parameter: {name}")
            delta = parameter.detach() - ffn_before[name]
            if bool((~authorized).any()):
                maximum_off_partition_delta = max(
                    maximum_off_partition_delta,
                    float(delta[~authorized].abs().max()),
                )

        for parameter, authorized, _learning_rate in entries:
            first, second = optimizer.state[id(parameter)]
            if bool((~authorized).any()):
                maximum_off_partition_moment = max(
                    maximum_off_partition_moment,
                    float(first[~authorized].abs().max()),
                    float(second[~authorized].abs().max()),
                )
        for other, before in inactive_classifiers.items():
            maximum_inactive_classifier_delta = max(
                maximum_inactive_classifier_delta,
                maximum_delta(before, model.classifiers[other]),
            )
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate(owner: int, applied: int) -> float:
        def scored_model_callback() -> float:
            model.eval()
            inverse = torch.tensor(
                [permutations[owner].index(label) for label in range(4)],
                device=device,
            )
            loader = DataLoader(
                test_dataset,
                batch_size=int(config["batch_size"]),
                shuffle=False,
            )
            correct = total = 0
            for ids, attention, labels in loader:
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions = inverse[
                        model(ids, attention, applied, masks[applied]).argmax(-1)
                    ]
                correct += int((predictions == labels).sum())
                total += labels.numel()
            return correct / total

        return runtime.invoke(applied, scored_model_callback)

    rows = []
    for owner in range(regimes):
        owning = evaluate(owner, owner)
        wrong = [
            evaluate(owner, applied)
            for applied in range(regimes)
            if applied != owner
        ]
        rows.append(
            {
                "regime": owner,
                "owning_accuracy": owning,
                "wrong_key_mean_accuracy": (
                    float(np.mean(wrong)) if wrong else None
                ),
            }
        )

    runtime_rejections = runtime.rejection_probe()
    separation_pass = (
        maximum_frozen_delta == 0.0
        and maximum_off_partition_delta == 0.0
        and maximum_off_partition_moment == 0.0
        and maximum_inactive_classifier_delta == 0.0
        and runtime_rejections["all_rejected"]
        and runtime_rejections["unauthorized_model_calls"] == 0
    )
    result = {
        "schema_version": 1,
        "experiment": "dense_ffn_cotenancy",
        "status": "pass" if separation_pass else "separation_failure",
        "partition_algorithm": verified_gate_mask_algorithm_identity(source),
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
        },
        "gate_placement": "post-activation-pre-down-projection",
        "runtime": {
            **runtime.evidence(),
            "rejection_probe": runtime_rejections,
        },
        "dimensions_per_regime_per_layer": 3072 // regimes,
        "maximum_frozen_shared_parameter_delta": maximum_frozen_delta,
        "maximum_off_partition_parameter_delta": maximum_off_partition_delta,
        "maximum_off_partition_optimizer_moment": maximum_off_partition_moment,
        "maximum_inactive_classifier_delta": maximum_inactive_classifier_delta,
        "mean_owning_accuracy": float(
            np.mean([row["owning_accuracy"] for row in rows])
        ),
        "mean_wrong_key_accuracy": (
            float(
                np.mean(
                    [
                        row["wrong_key_mean_accuracy"]
                        for row in rows
                        if row["wrong_key_mean_accuracy"] is not None
                    ]
                )
            )
            if regimes > 1
            else None
        ),
        "per_regime": rows,
    }
    remote_path = (
        Path("/results")
        / f"dense_ffn_R{regimes}_seed{seed}_{timestamp()}.json"
    )
    remote_path.write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    result["remote_artifact"] = str(remote_path)
    return result


@app.local_entrypoint()
def main(
    r_values: str = "1,2,4,8,16",
    seeds: str = "42,123,256",
    smoke: bool = False,
) -> None:
    ratios = [int(value) for value in r_values.split(",") if value]
    selected_seeds = [int(value) for value in seeds.split(",") if value]
    if smoke:
        ratios = [2]
        selected_seeds = [42]
    configs = [
        {
            "R": ratio,
            "seed": seed,
            "train_examples": 256 if smoke else 8000,
            "test_examples": 256 if smoke else 2000,
            "max_length": 64 if smoke else 128,
            "batch_size": 32,
            "epochs": 1 if smoke else 2,
            "ffn_learning_rate": 2e-5,
            "head_learning_rate": 1e-3,
            "smoke": smoke,
        }
        for ratio in ratios
        for seed in selected_seeds
    ]
    started = time.perf_counter()
    source = source_metadata()
    results = []
    for offset in range(0, len(configs), MAX_REMOTE_CONCURRENCY):
        handles = [
            run_ratio.spawn(config, source)
            for config in configs[offset : offset + MAX_REMOTE_CONCURRENCY]
        ]
        results.extend(handle.get() for handle in handles)
    artifact = {
        "schema_version": 1,
        "experiment": "dense_ffn_cotenancy_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"dense_ffn_cotenancy_{timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
