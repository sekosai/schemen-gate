"""Equal-compute public gate-adaptation factorial at strict R=8.

The protocol separates public adaptation from tenant learning:

1. Fine-tune an ungated DistilBERT teacher on a public AG News split.
2. Copy the same teacher into four public-initialization conditions:
   - no additional public adaptation (descriptive baseline),
   - one extra ungated hard-label epoch,
   - one extra all-mask hard-label epoch, and
   - one extra all-mask hard-label + distillation epoch.
   The three adapted conditions use the same examples, batch order, optimizer
   steps, teacher-target evaluations, and number of student forward/backward
   passes.  The ungated condition repeats each batch R times to match the two
   all-mask conditions at the model-pass level.
3. Freeze all shared paths.
4. Train each tenant using only its aligned FFN slices and private classifier.

This design separates the effect of extra public training, mask-aware public
training, and teacher distillation.  Public and tenant splits are disjoint.

    modal run experiments/modal_public_gate_adaptation_factorial.py --smoke
    modal run experiments/modal_public_gate_adaptation_factorial.py \
      --seeds 42,123,256,512,1024

The full command first completes its own one-seed reduced-data pilot and
checks all separation assertions before it dispatches the five full seeds.
"""

from __future__ import annotations

import json
import platform
import statistics
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

APP_NAME = "cdp-public-gate-adaptation-factorial"
VOLUME_NAME = "cdp-public-gate-adaptation-factorial-results"
CACHE_VOLUME_NAME = "cdp-huggingface-cache"
MODEL_NAME = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_NAME = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"
MAX_REMOTE_CONCURRENCY = 3

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(
    CACHE_VOLUME_NAME,
    create_if_missing=True,
)
image = install_current_schemen(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.10.1",
        "datasets==5.0.1",
        "numpy==2.4.6",
    )
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
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
    timeout=20 * 60,
    volumes={"/results": results_volume, "/cache": cache_volume},
)
def run_seed(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    import copy
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
    device = torch.device("cuda")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
    )
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION)
    shuffled = dataset["train"].shuffle(seed=1729)
    public_count = int(config["public_examples"])
    tenant_count = int(config["tenant_examples"])
    public_split = shuffled.select(range(public_count))
    tenant_split = shuffled.select(
        range(public_count, public_count + tenant_count)
    )
    test_split = dataset["test"].select(range(int(config["test_examples"])))

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

    public_data = encode(public_split)
    tenant_data = encode(tenant_split)
    test_data = encode(test_split)

    def loader(data, *, shuffle: bool, salt: int = 0):
        return DataLoader(
            data,
            batch_size=int(config["batch_size"]),
            shuffle=shuffle,
            generator=(
                torch.Generator().manual_seed(seed + salt)
                if shuffle
                else None
            ),
        )

    def balanced_masks() -> list[torch.Tensor]:
        key = seed.to_bytes(32, "big", signed=False)
        return [
            torch.as_tensor(
                GateMask.derive(key, regime, 3072, regimes).to_numpy(),
                dtype=torch.bool,
                device=device,
            )
            for regime in range(regimes)
        ]

    masks = balanced_masks()
    runtime = GateExecutionPreflight(
        model_id=MODEL_NAME,
        dimensions=3072,
        authorized_regime_ids=list(range(regimes)),
    )
    permutations = list(itertools.permutations(range(4)))
    np.random.default_rng(42).shuffle(permutations)
    permutations = permutations[:regimes]

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                MODEL_NAME,
                revision=MODEL_REVISION,
            )
            self.public_classifier = nn.Linear(768, 4)
            self.private_classifiers = nn.ModuleList(
                [nn.Linear(768, 4) for _ in range(regimes)]
            )
            self._active_mask: torch.Tensor | None = None
            self._hooks = [
                layer.ffn.lin2.register_forward_pre_hook(
                    self._mask_lin2_input
                )
                for layer in self.encoder.transformer.layer
            ]

        def _mask_lin2_input(self, _module, inputs):
            if self._active_mask is None:
                return inputs
            return (
                inputs[0] * self._active_mask.to(inputs[0].dtype),
            )

        def encode(self, ids, attention, mask=None):
            self._active_mask = mask
            try:
                return self.encoder(
                    input_ids=ids,
                    attention_mask=attention,
                ).last_hidden_state[:, 0, :]
            finally:
                self._active_mask = None

        def public_logits(self, ids, attention, mask=None):
            return self.public_classifier(self.encode(ids, attention, mask))

        def private_logits(self, ids, attention, regime, mask):
            return self.private_classifiers[regime](
                self.encode(ids, attention, mask)
            )

        def initialize_private_heads(self) -> None:
            state = self.public_classifier.state_dict()
            for classifier in self.private_classifiers:
                classifier.load_state_dict(state)

    def train_public_teacher(model: Model) -> None:
        model.train()
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.encoder.parameters(),
                    "lr": float(config["public_encoder_lr"]),
                },
                {
                    "params": model.public_classifier.parameters(),
                    "lr": float(config["public_head_lr"]),
                },
            ],
            weight_decay=0.01,
        )
        for _ in range(int(config["public_epochs"])):
            for ids, attention, labels in loader(
                public_data, shuffle=True, salt=11
            ):
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = F.cross_entropy(
                        model.public_logits(ids, attention),
                        labels,
                    )
                loss.backward()
                optimizer.step()

    def adapt_public(
        student: Model,
        teacher: Model,
        condition: str,
    ) -> dict[str, int]:
        allowed = {
            "ungated_hard_label",
            "all_mask_hard_label",
            "all_mask_distillation",
        }
        if condition not in allowed:
            raise ValueError(f"unknown public-adaptation condition: {condition}")
        teacher.eval()
        student.train()
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": student.encoder.parameters(),
                    "lr": float(config["adaptation_encoder_lr"]),
                },
                {
                    "params": student.public_classifier.parameters(),
                    "lr": float(config["adaptation_head_lr"]),
                },
            ],
            weight_decay=0.01,
        )
        temperature = float(config["temperature"])
        optimizer_steps = 0
        teacher_forward_passes = 0
        forward_backward_passes = 0
        for _ in range(int(config["adaptation_epochs"])):
            for ids, attention, labels in loader(
                public_data, shuffle=True, salt=23
            ):
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.float16
                ):
                    targets = teacher.public_logits(ids, attention)
                    soft_targets = F.softmax(
                        targets / temperature,
                        dim=-1,
                    )
                teacher_forward_passes += 1
                optimizer.zero_grad(set_to_none=True)
                for slot in range(regimes):
                    mask = (
                        None
                        if condition == "ungated_hard_label"
                        else masks[slot]
                    )
                    with torch.autocast("cuda", dtype=torch.float16):
                        logits = student.public_logits(ids, attention, mask)
                        hard_loss = F.cross_entropy(logits, labels)
                        soft_loss = F.kl_div(
                            F.log_softmax(
                                logits / temperature,
                                dim=-1,
                            ),
                            soft_targets,
                            reduction="batchmean",
                        ) * temperature**2
                        soft_weight = (
                            1 - float(config["hard_loss_weight"])
                            if condition == "all_mask_distillation"
                            else 0.0
                        )
                        hard_weight = 1.0 - soft_weight
                        loss = (
                            hard_weight * hard_loss
                            + soft_weight * soft_loss
                        ) / regimes
                    loss.backward()
                    forward_backward_passes += 1
                optimizer.step()
                optimizer_steps += 1
        return {
            "optimizer_steps": optimizer_steps,
            "teacher_forward_passes": teacher_forward_passes,
            "forward_backward_passes": forward_backward_passes,
        }

    @torch.no_grad()
    def public_accuracy(model: Model, mask=None, regime: int = 0) -> float:
        def scored_model_callback() -> float:
            model.eval()
            correct = total = 0
            for ids, attention, labels in loader(test_data, shuffle=False):
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions = model.public_logits(
                        ids, attention, mask
                    ).argmax(-1)
                correct += int((predictions == labels).sum())
                total += labels.numel()
            return correct / total

        return runtime.invoke(regime, scored_model_callback)

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
                for parameter, _authorized, _lr in self.entries
            }

        def zero_grad(self) -> None:
            for parameter, _authorized, _lr in self.entries:
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

    def configure_tenant_training(model: Model) -> None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in model.encoder.transformer.layer:
            layer.ffn.lin1.weight.requires_grad_(True)
            layer.ffn.lin1.bias.requires_grad_(True)
            layer.ffn.lin2.weight.requires_grad_(True)
        for classifier in model.private_classifiers:
            for parameter in classifier.parameters():
                parameter.requires_grad_(True)

    def entries_for(model: Model, regime: int):
        active = masks[regime]
        entries = []
        for layer in model.encoder.transformer.layer:
            entries.extend(
                [
                    (
                        layer.ffn.lin1.weight,
                        active[:, None].expand_as(layer.ffn.lin1.weight),
                        float(config["tenant_ffn_lr"]),
                    ),
                    (
                        layer.ffn.lin1.bias,
                        active,
                        float(config["tenant_ffn_lr"]),
                    ),
                    (
                        layer.ffn.lin2.weight,
                        active[None, :].expand_as(layer.ffn.lin2.weight),
                        float(config["tenant_ffn_lr"]),
                    ),
                ]
            )
        for parameter in model.private_classifiers[regime].parameters():
            entries.append(
                (
                    parameter,
                    torch.ones_like(parameter, dtype=torch.bool),
                    float(config["tenant_head_lr"]),
                )
            )
        return entries

    def snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in module.named_parameters()
        }

    def max_delta(before, module) -> float:
        return max(
            float((parameter.detach().cpu() - before[name]).abs().max())
            for name, parameter in module.named_parameters()
        )

    @torch.no_grad()
    def tenant_accuracy(model: Model, regime: int) -> float:
        def scored_model_callback() -> float:
            model.eval()
            inverse = torch.tensor(
                [permutations[regime].index(label) for label in range(4)],
                device=device,
            )
            correct = total = 0
            for ids, attention, labels in loader(test_data, shuffle=False):
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions = inverse[
                        model.private_logits(
                            ids, attention, regime, masks[regime]
                        ).argmax(-1)
                    ]
                correct += int((predictions == labels).sum())
                total += labels.numel()
            return correct / total

        return runtime.invoke(regime, scored_model_callback)

    def tenant_stage(initial_state, condition: str) -> dict[str, Any]:
        model = Model().to(device)
        model.load_state_dict(initial_state)
        model.initialize_private_heads()
        configure_tenant_training(model)
        torch.manual_seed(seed + 7000)
        torch.cuda.manual_seed_all(seed + 7000)
        frozen_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.encoder.named_parameters()
            if not parameter.requires_grad
        }
        max_frozen = 0.0
        max_off_partition = 0.0
        max_off_moment = 0.0
        max_inactive_head = 0.0

        for regime in range(regimes):
            active = masks[regime]
            entries = entries_for(model, regime)
            optimizer = ScopedAdam(entries)
            ffn_before = {
                name: parameter.detach().clone()
                for name, parameter in model.encoder.named_parameters()
                if parameter.requires_grad
            }
            inactive_heads = {
                other: snapshot(model.private_classifiers[other])
                for other in range(regimes)
                if other != regime
            }
            permutation = torch.tensor(permutations[regime], device=device)
            model.train()
            for _ in range(int(config["tenant_epochs"])):
                for ids, attention, labels in loader(
                    tenant_data,
                    shuffle=True,
                    salt=100 + regime,
                ):
                    ids = ids.to(device)
                    attention = attention.to(device)
                    labels = permutation[labels.to(device)]
                    optimizer.zero_grad()
                    with torch.autocast("cuda", dtype=torch.float16):
                        loss = F.cross_entropy(
                            model.private_logits(
                                ids, attention, regime, active
                            ),
                            labels,
                        )
                    loss.backward()
                    optimizer.step()

            for name, parameter in model.encoder.named_parameters():
                if not parameter.requires_grad:
                    max_frozen = max(
                        max_frozen,
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
                    raise AssertionError(name)
                delta = parameter.detach() - ffn_before[name]
                max_off_partition = max(
                    max_off_partition,
                    float(delta[~authorized].abs().max()),
                )
            for parameter, authorized, _lr in entries:
                first, second = optimizer.state[id(parameter)]
                if bool((~authorized).any()):
                    max_off_moment = max(
                        max_off_moment,
                        float(first[~authorized].abs().max()),
                        float(second[~authorized].abs().max()),
                    )
            for other, before in inactive_heads.items():
                max_inactive_head = max(
                    max_inactive_head,
                    max_delta(before, model.private_classifiers[other]),
                )

        accuracies = [
            tenant_accuracy(model, regime) for regime in range(regimes)
        ]
        result = {
            "condition": condition,
            "per_regime_accuracy": accuracies,
            "mean_owning_accuracy": float(np.mean(accuracies)),
            "maximum_frozen_shared_parameter_delta": max_frozen,
            "maximum_off_partition_parameter_delta": max_off_partition,
            "maximum_off_partition_optimizer_moment": max_off_moment,
            "maximum_inactive_classifier_delta": max_inactive_head,
        }
        result["separation_pass"] = all(
            value == 0.0
            for key, value in result.items()
            if key.startswith("maximum_")
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return result

    teacher = Model().to(device)
    train_public_teacher(teacher)
    teacher_ungated_accuracy = public_accuracy(teacher)
    teacher_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in teacher.state_dict().items()
    }
    baseline_gated_public_accuracy = float(
        np.mean(
            [
                public_accuracy(teacher, mask, regime)
                for regime, mask in enumerate(masks)
            ]
        )
    )

    condition_states = {"no_extra_public_adaptation": teacher_state}
    condition_public_metrics: dict[str, dict[str, Any]] = {
        "no_extra_public_adaptation": {
            "gated_public_accuracy": baseline_gated_public_accuracy,
            "optimizer_steps": 0,
            "forward_backward_passes": 0,
        }
    }
    adapted_conditions = (
        "ungated_hard_label",
        "all_mask_hard_label",
        "all_mask_distillation",
    )
    for condition in adapted_conditions:
        student = copy.deepcopy(teacher)
        torch.manual_seed(seed + 5000)
        torch.cuda.manual_seed_all(seed + 5000)
        work = adapt_public(student, teacher, condition)
        condition_public_metrics[condition] = {
            "gated_public_accuracy": float(
                np.mean(
                    [
                        public_accuracy(student, mask, regime)
                        for regime, mask in enumerate(masks)
                    ]
                )
            ),
            **work,
        }
        condition_states[condition] = {
            name: tensor.detach().cpu().clone()
            for name, tensor in student.state_dict().items()
        }
        del student
        gc.collect()
        torch.cuda.empty_cache()

    adapted_work = [
        condition_public_metrics[condition]
        for condition in adapted_conditions
    ]
    for field in (
        "optimizer_steps",
        "teacher_forward_passes",
        "forward_backward_passes",
    ):
        if len({metrics[field] for metrics in adapted_work}) != 1:
            raise AssertionError(f"adaptation work mismatch for {field}")

    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    conditions = {
        condition: tenant_stage(state, condition)
        for condition, state in condition_states.items()
    }
    baseline_accuracy = conditions[
        "no_extra_public_adaptation"
    ]["mean_owning_accuracy"]
    for condition, metrics in conditions.items():
        metrics["public_adaptation"] = condition_public_metrics[condition]
        metrics["uplift_vs_no_extra_fraction"] = (
            metrics["mean_owning_accuracy"] - baseline_accuracy
        )
        metrics["uplift_vs_no_extra_percentage_points"] = (
            100 * metrics["uplift_vs_no_extra_fraction"]
        )

    contrasts = {
        "extra_training_effect_pp": 100
        * (
            conditions["ungated_hard_label"]["mean_owning_accuracy"]
            - baseline_accuracy
        ),
        "mask_aware_effect_pp": 100
        * (
            conditions["all_mask_hard_label"]["mean_owning_accuracy"]
            - conditions["ungated_hard_label"]["mean_owning_accuracy"]
        ),
        "distillation_effect_given_masks_pp": 100
        * (
            conditions["all_mask_distillation"]["mean_owning_accuracy"]
            - conditions["all_mask_hard_label"]["mean_owning_accuracy"]
        ),
        "full_pipeline_effect_pp": 100
        * (
            conditions["all_mask_distillation"]["mean_owning_accuracy"]
            - baseline_accuracy
        ),
    }
    runtime_rejections = runtime.rejection_probe()
    result = {
        "schema_version": 2,
        "experiment": "public_gate_adaptation_factorial",
        "status": (
            "pass"
            if all(
                condition["separation_pass"]
                for condition in conditions.values()
            )
            and runtime_rejections["all_rejected"]
            and runtime_rejections["unauthorized_model_calls"] == 0
            else "separation_failure"
        ),
        "status_definition": (
            "pass means every exact tenant-stage state-separation check "
            "passed; utility contrasts are measured without a pass threshold"
        ),
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
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "tf32_enabled": torch.backends.cuda.matmul.allow_tf32,
        },
        "gate_placement": "post-activation-pre-down-projection",
        "runtime": {
            **runtime.evidence(),
            "rejection_probe": runtime_rejections,
        },
        "public_and_tenant_splits_disjoint": True,
        "equal_compute_adapted_conditions": {
            "same_public_examples": True,
            "same_batch_order": True,
            "same_dropout_seed_schedule": True,
            "same_optimizer_steps": True,
            "same_teacher_forward_passes": True,
            "same_forward_backward_passes": True,
            "ungated_repetitions_per_batch": regimes,
            "qualification": (
                "matched model-pass counts; masked conditions include the "
                "mask multiplication required by their treatment"
            ),
        },
        "teacher_ungated_accuracy": teacher_ungated_accuracy,
        "conditions": conditions,
        "contrasts": contrasts,
    }
    remote_path = (
        Path("/results")
        / f"public_gate_factorial_R{regimes}_seed{seed}_{timestamp()}.json"
    )
    remote_path.write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    if bool(config["smoke"]):
        cache_volume.commit()
    result["remote_artifact"] = str(remote_path)
    return result


@app.local_entrypoint()
def main(
    seeds: str = "42,123,256,512,1024",
    smoke: bool = False,
) -> None:
    selected_seeds = [int(value) for value in seeds.split(",") if value]
    if smoke:
        selected_seeds = [42]
    if not selected_seeds:
        raise ValueError("at least one seed is required")

    def make_config(seed: int, *, reduced: bool) -> dict[str, Any]:
        return {
            "seed": seed,
            "R": 8,
            "public_examples": 256 if reduced else 8000,
            "tenant_examples": 256 if reduced else 8000,
            "test_examples": 256 if reduced else 2000,
            "max_length": 64 if reduced else 128,
            "batch_size": 32,
            "public_epochs": 1 if reduced else 2,
            "adaptation_epochs": 1,
            "tenant_epochs": 1 if reduced else 2,
            "public_encoder_lr": 2e-5,
            "public_head_lr": 1e-3,
            "adaptation_encoder_lr": 2e-5,
            "adaptation_head_lr": 5e-4,
            "tenant_ffn_lr": 2e-5,
            "tenant_head_lr": 1e-3,
            "temperature": 2.0,
            "hard_loss_weight": 0.5,
            "smoke": reduced,
        }

    started = time.perf_counter()
    source = source_metadata()
    pilot = None
    if smoke:
        results = [run_seed.remote(make_config(42, reduced=True), source)]
    else:
        pilot = run_seed.remote(
            make_config(selected_seeds[0], reduced=True),
            source,
        )
        if pilot["status"] != "pass":
            raise RuntimeError(
                "reduced-data pilot failed; full seeds were not dispatched"
            )
        results = []
        for offset in range(0, len(selected_seeds), MAX_REMOTE_CONCURRENCY):
            handles = [
                run_seed.spawn(make_config(seed, reduced=False), source)
                for seed in selected_seeds[
                    offset : offset + MAX_REMOTE_CONCURRENCY
                ]
            ]
            results.extend(handle.get() for handle in handles)
    condition_names = list(results[0]["conditions"])
    contrast_names = list(results[0]["contrasts"])
    summary = {
        "seeds": selected_seeds,
        "all_separation_checks_pass": all(
            result["status"] == "pass" for result in results
        ),
        "conditions": {},
        "contrasts": {},
        "teacher_ungated_accuracy_percent": {},
    }
    teacher_values = [
        100 * result["teacher_ungated_accuracy"] for result in results
    ]
    summary["teacher_ungated_accuracy_percent"] = {
        "by_seed": teacher_values,
        "mean": statistics.mean(teacher_values),
        "sample_sd_percentage_points": (
            statistics.stdev(teacher_values) if len(teacher_values) > 1 else None
        ),
    }
    for condition in condition_names:
        values = [
            100 * result["conditions"][condition]["mean_owning_accuracy"]
            for result in results
        ]
        summary["conditions"][condition] = {
            "owning_accuracy_percent_by_seed": values,
            "mean_owning_accuracy_percent": statistics.mean(values),
            "sample_sd_percentage_points": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
            "gated_public_accuracy_percent_by_seed": [
                100
                * result["conditions"][condition]["public_adaptation"][
                    "gated_public_accuracy"
                ]
                for result in results
            ],
            "maximum_unauthorized_state_delta": max(
                max(
                    metric
                    for key, metric in result["conditions"][condition].items()
                    if key.startswith("maximum_")
                )
                for result in results
            ),
        }
        public_values = summary["conditions"][condition][
            "gated_public_accuracy_percent_by_seed"
        ]
        summary["conditions"][condition].update(
            {
                "mean_gated_public_accuracy_percent": statistics.mean(
                    public_values
                ),
                "gated_public_sample_sd_percentage_points": (
                    statistics.stdev(public_values)
                    if len(public_values) > 1
                    else None
                ),
            }
        )
    for contrast in contrast_names:
        values = [result["contrasts"][contrast] for result in results]
        summary["contrasts"][contrast] = {
            "percentage_points_by_seed": values,
            "mean_percentage_points": statistics.mean(values),
            "sample_sd_percentage_points": (
                statistics.stdev(values) if len(values) > 1 else None
            ),
        }
    artifact = {
        "schema_version": 2,
        "experiment": "public_gate_adaptation_factorial_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "source": source,
        "preflight_pilot": pilot,
        "summary": summary,
        "results": results,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"public_gate_adaptation_factorial_{timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
