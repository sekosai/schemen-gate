"""GPU evaluation of complete private lanes on a frozen DistilBERT backbone.

Two Transformer-friendly designs are compared:

* ``adapter``: a small private residual adapter after every Transformer block;
* ``expert``: a full-width private FFN expert after every Transformer block.

The pretrained attention/FFN backbone is frozen and shared.  Every trainable
tenant-specific path, including the classifier, lives in a regime-selected
lane.  The experiment verifies that shared and inactive-lane parameters remain
bit-identical during each tenant update.

Staged examples:

    modal run experiments/modal_private_transformer_lanes.py --smoke
    modal run experiments/modal_private_transformer_lanes.py \
      --designs adapter,expert --seeds 42,123,256
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

APP_NAME = "cdp-private-transformer-lanes"
VOLUME_NAME = "cdp-private-transformer-lanes-results"
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
def run_design(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    import gc
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from execution_preflight import GateExecutionPreflight
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

    class PrivateBlock(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.up = nn.Linear(768, width)
            self.down = nn.Linear(width, 768, bias=False)
            nn.init.zeros_(self.down.weight)

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.down(F.gelu(self.up(hidden)))

    class PrivateLane(nn.Module):
        def __init__(self, width: int, layers: int) -> None:
            super().__init__()
            self.blocks = nn.ModuleList(
                [PrivateBlock(width) for _ in range(layers)]
            )
            self.classifier = nn.Linear(768, 4)

    class SharedBackbonePrivateLanes(nn.Module):
        def __init__(self, design: str, regimes: int) -> None:
            super().__init__()
            self.backbone = AutoModel.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
            )
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            width = 64 if design == "adapter" else int(
                self.backbone.config.hidden_dim
            )
            self.lanes = nn.ModuleList(
                [
                    PrivateLane(width, len(self.backbone.transformer.layer))
                    for _ in range(regimes)
                ]
            )
            self._active_regime: int | None = None
            self._hooks = []
            for layer_index, layer in enumerate(self.backbone.transformer.layer):
                self._hooks.append(
                    layer.register_forward_hook(
                        self._make_private_residual_hook(layer_index)
                    )
                )

        def _make_private_residual_hook(self, layer_index: int):
            def hook(_module, _inputs, output):
                if self._active_regime is None:
                    raise RuntimeError("private lane hook called without regime")
                block = self.lanes[self._active_regime].blocks[layer_index]
                return output + block(output)

            return hook

        def forward(self, ids, attention, regime: int):
            self._active_regime = regime
            try:
                hidden = self.backbone(
                    input_ids=ids,
                    attention_mask=attention,
                ).last_hidden_state[:, 0, :]
            finally:
                self._active_regime = None
            return self.lanes[regime].classifier(hidden)

    def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }

    def maximum_delta(
        before: dict[str, torch.Tensor],
        module: nn.Module,
    ) -> float:
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

    @torch.no_grad()
    def evaluate(model, loader, regime, permutation):
        def scored_model_callback():
            model.eval()
            inverse = torch.tensor(
                [permutation.index(label) for label in range(4)],
                device=device,
            )
            correct = total = 0
            for ids, attention, labels in loader:
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions = inverse[
                        model(ids, attention, regime).argmax(dim=-1)
                    ]
                correct += int((predictions == labels).sum())
                total += labels.numel()
            return correct / total

        return runtime.invoke(regime, scored_model_callback)

    regimes = int(config["regimes"])
    permutations = [
        [0, 1, 2, 3],
        [1, 2, 3, 0],
        [2, 3, 0, 1],
        [3, 0, 1, 2],
    ][:regimes]
    model = SharedBackbonePrivateLanes(config["design"], regimes).to(device)
    runtime = GateExecutionPreflight(
        model_id=MODEL_NAME,
        dimensions=768,
        authorized_regime_ids=list(range(regimes)),
    )
    backbone_initial = snapshot(model.backbone)
    maximum_inactive_delta = 0.0
    maximum_backbone_delta = 0.0

    for regime in range(regimes):
        inactive_before = {
            other: snapshot(model.lanes[other])
            for other in range(regimes)
            if other != regime
        }
        loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed + regime),
        )
        optimizer = torch.optim.AdamW(
            model.lanes[regime].parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=0.01,
        )
        permutation = torch.tensor(permutations[regime], device=device)
        model.train()
        for _ in range(int(config["epochs"])):
            for ids, attention, labels in loader:
                ids = ids.to(device)
                attention = attention.to(device)
                labels = permutation[labels.to(device)]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(ids, attention, regime)
                    loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()

        for other, before in inactive_before.items():
            maximum_inactive_delta = max(
                maximum_inactive_delta,
                maximum_delta(before, model.lanes[other]),
            )
        maximum_backbone_delta = max(
            maximum_backbone_delta,
            maximum_delta(backbone_initial, model.backbone),
        )

        del optimizer
        gc.collect()
        torch.cuda.empty_cache()

    test_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    rows = []
    for regime in range(regimes):
        owning = evaluate(
            model,
            test_loader,
            regime,
            permutations[regime],
        )
        wrong = [
            evaluate(model, test_loader, other, permutations[regime])
            for other in range(regimes)
            if other != regime
        ]
        rows.append(
            {
                "regime": regime,
                "owning_accuracy": owning,
                "wrong_key_mean_accuracy": float(np.mean(wrong)),
            }
        )

    runtime_rejections = runtime.rejection_probe()
    separation_pass = (
        maximum_backbone_delta == 0.0
        and maximum_inactive_delta == 0.0
        and runtime_rejections["all_rejected"]
        and runtime_rejections["unauthorized_model_calls"] == 0
    )
    result = {
        "schema_version": 1,
        "experiment": "private_transformer_lanes",
        "status": (
            "pass"
            if separation_pass
            else "separation_failure"
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
        },
        "gate_placement": "gate-authorized-hard-route-to-complete-private-lane",
        "runtime": {
            **runtime.evidence(),
            "rejection_probe": runtime_rejections,
        },
        "maximum_shared_backbone_delta": maximum_backbone_delta,
        "maximum_inactive_lane_delta": maximum_inactive_delta,
        "mean_owning_accuracy": float(
            np.mean([row["owning_accuracy"] for row in rows])
        ),
        "mean_wrong_key_accuracy": float(
            np.mean([row["wrong_key_mean_accuracy"] for row in rows])
        ),
        "per_regime": rows,
    }
    remote_path = (
        Path("/results")
        / f"{config['design']}_seed{seed}_{timestamp()}.json"
    )
    remote_path.write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    result["remote_artifact"] = str(remote_path)
    return result


@app.local_entrypoint()
def main(
    designs: str = "adapter,expert",
    seeds: str = "42,123,256",
    smoke: bool = False,
) -> None:
    source = source_metadata()
    selected_designs = [value for value in designs.split(",") if value]
    selected_seeds = [int(value) for value in seeds.split(",") if value]
    if smoke:
        selected_designs = [selected_designs[0]]
        selected_seeds = [selected_seeds[0]]

    configs = []
    for design in selected_designs:
        if design not in {"adapter", "expert"}:
            raise ValueError(f"unsupported design: {design}")
        for seed in selected_seeds:
            configs.append(
                {
                    "design": design,
                    "seed": seed,
                    "regimes": 4,
                    "train_examples": 256 if smoke else 8000,
                    "test_examples": 256 if smoke else 2000,
                    "max_length": 64 if smoke else 128,
                    "batch_size": 32,
                    "epochs": 1 if smoke else 2,
                    "learning_rate": 5e-4 if design == "adapter" else 1e-4,
                    "smoke": smoke,
                }
            )

    started = time.perf_counter()
    results = []
    for offset in range(0, len(configs), MAX_REMOTE_CONCURRENCY):
        handles = [
            run_design.spawn(config, source)
            for config in configs[offset : offset + MAX_REMOTE_CONCURRENCY]
        ]
        results.extend(handle.get() for handle in handles)
    artifact = {
        "schema_version": 1,
        "experiment": "private_transformer_lanes_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "source": source,
        "results": results,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"private_transformer_lanes_{timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
