"""Matched CDP canary deposition and hollow-regime lifecycle study.

This experiment replaces the unmatched 0/30 CDP control with a matched test:
synthetic, disjoint, preregistered canary corpora are deposited into initially
hollow DistilBERT FFN regimes.  It measures positive owning-key classification,
every ordered wrong-key pair, and exact parameter/optimizer-state confinement.

The DistilBERT FFN is rewritten as R physical lanes.  Selecting one lane is the
sparse implementation of a one-hot gate immediately after the intermediate
activation and before the down projection.  Shared attention, embeddings,
normalization, and the shared down-projection bias remain frozen.

Publication-oriented default:
    modal run experiments/modal_matched_deposition.py

One-seed protocol/debug smoke test:
    modal run experiments/modal_matched_deposition.py --smoke

No work is performed on import.  Each remote seed job writes a timestamped JSON
artifact to the named Modal Volume before returning.  The local entrypoint also
writes a combined timestamped JSON artifact under experiments/results/.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cdp-matched-deposition"
VOLUME_NAME = "cdp-matched-deposition-results"
REMOTE_RESULTS_DIR = "/results"
DEFAULT_MODEL = "distilbert/distilbert-base-uncased"
DEFAULT_MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DEFAULT_SEEDS = "1103,2207,3301,4409,5501"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.13.0",
    "transformers==5.10.1",
    "numpy==2.4.6",
    "scipy==1.17.1",
)


@app.function(
    max_containers=3,
    gpu="L4",
    image=gpu_image,
    timeout=10_800,
    volumes={REMOTE_RESULTS_DIR: results_volume},
)
def run_seed(
    seed: int,
    r: int,
    canary_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_name: str,
    model_revision: str,
    max_length: int,
) -> dict[str, Any]:
    """Run one independently seeded study and durably archive its result."""
    import hashlib
    import os
    import random
    import traceback
    import uuid

    import numpy as np
    import scipy
    import torch
    import transformers
    from scipy.stats import beta
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModel, AutoTokenizer

    started_at = datetime.now(timezone.utc)
    artifact_id = (
        f"matched_deposition_seed{seed}_"
        f"{started_at.strftime('%Y%m%dT%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"
    )
    artifact_path = Path(REMOTE_RESULTS_DIR) / f"{artifact_id}.json"
    t0 = time.time()

    def exact_binomial(successes: int, total: int, confidence: float = 0.95) -> dict[str, Any]:
        """Two-sided Clopper-Pearson interval."""
        if total <= 0 or not 0 <= successes <= total:
            raise ValueError("binomial counts must satisfy 0 <= successes <= total")
        alpha = 1.0 - confidence
        lower = 0.0 if successes == 0 else float(
            beta.ppf(alpha / 2.0, successes, total - successes + 1)
        )
        upper = 1.0 if successes == total else float(
            beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes)
        )
        return {
            "method": "Clopper-Pearson exact",
            "confidence": confidence,
            "successes": successes,
            "total": total,
            "estimate": successes / total,
            "lower": lower,
            "upper": upper,
        }

    def persist(payload: dict[str, Any]) -> None:
        artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        results_volume.commit()

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "matched_cdp_canary_deposition",
        "artifact_id": artifact_id,
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "remote_artifact": {
            "modal_volume": VOLUME_NAME,
            "path": str(artifact_path),
        },
        "config": {
            "seed": seed,
            "r": r,
            "canary_count_per_regime": canary_count,
            "epochs_per_regime": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": 0.0,
            "model": model_name,
            "model_revision": model_revision,
            "max_length": max_length,
        },
    }

    try:
        if r < 2:
            raise ValueError("r must be at least 2 to define wrong-key pairs")
        if canary_count < 2 or canary_count % 2:
            raise ValueError("canary_count must be an even integer >= 2")
        if epochs < 1 or batch_size < 1 or max_length < 8:
            raise ValueError("epochs, batch_size, and max_length are out of range")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device("cuda")

        # Preregister all corpora before tokenizer/model construction or training.
        templates_train = [
            "Archive lookup {identifier}. Assign this record to class {query}.",
            "Classify sealed canary {identifier}; requested field is {query}.",
            "For registry item {identifier}, return the binary category for {query}.",
        ]
        template_eval = "Audit query for preregistered canary {identifier}: category?"
        corpora: list[dict[str, Any]] = []
        all_identifiers: set[str] = set()
        for owner in range(r):
            # Reproducible corpus generation, never key or nonce generation.
            corpus_rng = random.Random(  # nosec B311
                seed * 1_000_003 + owner * 97_409
            )
            labels = [0, 1] * (canary_count // 2)
            corpus_rng.shuffle(labels)
            records = []
            for index, label in enumerate(labels):
                nonce = "".join(corpus_rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(18))
                identifier = f"rg{owner}-{nonce}-{index:04d}"
                if identifier in all_identifiers:
                    raise AssertionError("canary identifier collision")
                all_identifiers.add(identifier)
                records.append(
                    {
                        "index": index,
                        "identifier": identifier,
                        "label": label,
                        "train_texts": [
                            template.format(identifier=identifier, query="classification")
                            for template in templates_train
                        ],
                        "eval_text": template_eval.format(identifier=identifier),
                    }
                )
            corpora.append({"owner_regime": owner, "records": records})

        canonical_corpora = json.dumps(
            corpora, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        preregistration = {
            "created_before_model_load": True,
            "generator": "balanced binary labels; seeded opaque identifiers; three train templates; one held-out eval template",
            "sha256": hashlib.sha256(canonical_corpora).hexdigest(),
            "disjoint_identifier_count": len(all_identifiers),
            "expected_identifier_count": r * canary_count,
            "corpora": corpora,
        }

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=model_revision,
        )
        backbone = AutoModel.from_pretrained(
            model_name,
            revision=model_revision,
        )
        config = backbone.config
        if not hasattr(backbone, "transformer"):
            raise TypeError("the selected model must expose a DistilBERT-style transformer")
        intermediate_dim = int(config.dim * 4)
        # Read the actual checkpoint shape rather than trusting config conventions.
        intermediate_dim = int(backbone.transformer.layer[0].ffn.lin1.out_features)
        if intermediate_dim % r:
            raise ValueError(
                f"intermediate dimension {intermediate_dim} is not divisible by r={r}"
            )
        lane_width = intermediate_dim // r

        class FFNLane(nn.Module):
            def __init__(self, up: nn.Linear, down: nn.Linear):
                super().__init__()
                self.up = up
                self.down = down

        class PartitionedFFN(nn.Module):
            """Sparse one-hot intermediate gate before the down projection."""

            def __init__(self, original: nn.Module, regime_count: int):
                super().__init__()
                width = original.lin1.out_features // regime_count
                self.lanes = nn.ModuleList()
                for regime in range(regime_count):
                    sl = slice(regime * width, (regime + 1) * width)
                    up = nn.Linear(original.lin1.in_features, width, bias=True)
                    down = nn.Linear(width, original.lin2.out_features, bias=False)
                    with torch.no_grad():
                        up.weight.copy_(original.lin1.weight[sl])
                        up.bias.copy_(original.lin1.bias[sl])
                        down.weight.copy_(original.lin2.weight[:, sl])
                    self.lanes.append(FFNLane(up, down))
                self.down_bias = nn.Parameter(original.lin2.bias.detach().clone(), requires_grad=False)
                self.activation = original.activation
                self.dropout = original.dropout
                self.active_regime = 0

            def forward(self, hidden: torch.Tensor) -> torch.Tensor:
                lane = self.lanes[self.active_regime]
                # The regime selection is exactly a one-hot gate on the concatenated
                # intermediate activation, evaluated sparsely to avoid touching
                # non-owning parameters or optimizer state.
                intermediate = self.activation(lane.up(hidden))
                return self.dropout(lane.down(intermediate) + self.down_bias)

        for block in backbone.transformer.layer:
            block.ffn = PartitionedFFN(block.ffn, r)

        class CanaryClassifier(nn.Module):
            def __init__(self, base: nn.Module, regime_count: int):
                super().__init__()
                self.backbone = base
                self.dropout = nn.Dropout(float(config.seq_classif_dropout))
                self.heads = nn.ModuleList(
                    nn.Linear(int(config.dim), 2) for _ in range(regime_count)
                )
                self.active_head = 0

            def set_route(self, gate_regime: int, head_regime: int) -> None:
                if not 0 <= gate_regime < len(self.heads):
                    raise IndexError(gate_regime)
                if not 0 <= head_regime < len(self.heads):
                    raise IndexError(head_regime)
                self.active_head = head_regime
                for block in self.backbone.transformer.layer:
                    block.ffn.active_regime = gate_regime

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                hidden = self.backbone(
                    input_ids=input_ids, attention_mask=attention_mask
                ).last_hidden_state[:, 0]
                return self.heads[self.active_head](self.dropout(hidden))

        model = CanaryClassifier(backbone, r).to(device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.backbone.transformer.layer:
            for lane in block.ffn.lanes:
                for parameter in lane.parameters():
                    parameter.requires_grad_(True)

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=0.0, foreach=False
        )
        loss_fn = nn.CrossEntropyLoss()

        def encoded_loader(owner: int, train: bool, shuffle_seed: int = 0) -> DataLoader:
            records = corpora[owner]["records"]
            texts = (
                [text for record in records for text in record["train_texts"]]
                if train
                else [record["eval_text"] for record in records]
            )
            labels = (
                [record["label"] for record in records for _ in record["train_texts"]]
                if train
                else [record["label"] for record in records]
            )
            tokens = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            dataset = TensorDataset(
                tokens["input_ids"],
                tokens["attention_mask"],
                torch.tensor(labels, dtype=torch.long),
            )
            generator = torch.Generator().manual_seed(shuffle_seed)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=train,
                generator=generator if train else None,
            )

        eval_loaders = [encoded_loader(owner, train=False) for owner in range(r)]

        def evaluate_matrix() -> dict[str, Any]:
            model.eval()
            pairs = []
            with torch.no_grad():
                for corpus_owner in range(r):
                    for key_regime in range(r):
                        # Hold the intended owning classifier fixed and vary only
                        # the FFN credential/gate.  Changing both would confound a
                        # wrong-key result with an untrained-head mismatch.
                        model.set_route(
                            gate_regime=key_regime, head_regime=corpus_owner
                        )
                        correct = 0
                        total = 0
                        predictions: list[int] = []
                        for input_ids, attention_mask, labels in eval_loaders[corpus_owner]:
                            logits = model(
                                input_ids.to(device), attention_mask.to(device)
                            )
                            predicted = logits.argmax(dim=-1).cpu()
                            predictions.extend(int(x) for x in predicted.tolist())
                            correct += int((predicted == labels).sum().item())
                            total += int(labels.numel())
                        pairs.append(
                            {
                                "corpus_owner": corpus_owner,
                                "key_regime": key_regime,
                                "pair_type": "owning" if corpus_owner == key_regime else "wrong_key",
                                "interval": exact_binomial(correct, total),
                                "predictions": predictions,
                            }
                        )
            owning_correct = sum(
                pair["interval"]["successes"] for pair in pairs if pair["pair_type"] == "owning"
            )
            owning_total = sum(
                pair["interval"]["total"] for pair in pairs if pair["pair_type"] == "owning"
            )
            wrong_correct = sum(
                pair["interval"]["successes"] for pair in pairs if pair["pair_type"] == "wrong_key"
            )
            wrong_total = sum(
                pair["interval"]["total"] for pair in pairs if pair["pair_type"] == "wrong_key"
            )
            return {
                "pairs": pairs,
                "owning_aggregate": exact_binomial(owning_correct, owning_total),
                "wrong_key_aggregate": exact_binomial(wrong_correct, wrong_total),
            }

        named_parameters = dict(model.named_parameters())

        def lane_parameter_names(regime: int) -> list[str]:
            names = []
            marker = f".lanes.{regime}."
            for name, parameter in named_parameters.items():
                if marker in name or name.startswith(f"heads.{regime}."):
                    if parameter.requires_grad:
                        names.append(name)
            return sorted(names)

        owned_names = {regime: lane_parameter_names(regime) for regime in range(r)}
        all_owned_names = set().union(*[set(names) for names in owned_names.values()])

        def tensor_snapshot(names: set[str]) -> dict[str, Any]:
            snapshot: dict[str, Any] = {"parameters": {}, "optimizer": {}}
            for name in sorted(names):
                parameter = named_parameters[name]
                snapshot["parameters"][name] = parameter.detach().cpu().clone()
                state = optimizer.state.get(parameter)
                if state is None:
                    snapshot["optimizer"][name] = None
                else:
                    snapshot["optimizer"][name] = {
                        key: value.detach().cpu().clone()
                        for key, value in state.items()
                        if torch.is_tensor(value)
                    }
            return snapshot

        def delta_report(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
            parameter_nonzero = 0
            parameter_max = 0.0
            state_nonzero = 0
            state_max = 0.0
            state_structure_changes = []
            per_tensor = []
            for name, old_parameter in before["parameters"].items():
                new_parameter = after["parameters"][name]
                difference = new_parameter - old_parameter
                nonzero = int(torch.count_nonzero(difference).item())
                maximum = float(difference.abs().max().item()) if difference.numel() else 0.0
                parameter_nonzero += nonzero
                parameter_max = max(parameter_max, maximum)
                old_state = before["optimizer"][name]
                new_state = after["optimizer"][name]
                if (old_state is None) != (new_state is None):
                    state_structure_changes.append(
                        {
                            "parameter": name,
                            "before": "absent" if old_state is None else "present",
                            "after": "absent" if new_state is None else "present",
                        }
                    )
                state_keys = set(old_state or {}) | set(new_state or {})
                state_details = {}
                for key in sorted(state_keys):
                    old_value = (old_state or {}).get(key)
                    new_value = (new_state or {}).get(key)
                    if old_value is None or new_value is None:
                        state_details[key] = {"comparable": False}
                        continue
                    state_difference = new_value - old_value
                    state_nz = int(torch.count_nonzero(state_difference).item())
                    state_absmax = (
                        float(state_difference.abs().max().item())
                        if state_difference.numel()
                        else 0.0
                    )
                    state_nonzero += state_nz
                    state_max = max(state_max, state_absmax)
                    state_details[key] = {
                        "comparable": True,
                        "nonzero_elements": state_nz,
                        "max_abs_delta": state_absmax,
                        "exact_equal": bool(torch.equal(old_value, new_value)),
                    }
                per_tensor.append(
                    {
                        "parameter": name,
                        "parameter_nonzero_elements": nonzero,
                        "parameter_max_abs_delta": maximum,
                        "parameter_exact_equal": bool(torch.equal(old_parameter, new_parameter)),
                        "optimizer_tensor_state": state_details,
                    }
                )
            return {
                "parameter_nonzero_elements": parameter_nonzero,
                "parameter_max_abs_delta": parameter_max,
                "parameters_exact_equal": parameter_nonzero == 0,
                "optimizer_nonzero_elements": state_nonzero,
                "optimizer_max_abs_delta": state_max,
                "optimizer_tensor_state_exact_equal": state_nonzero == 0,
                "optimizer_state_structure_changes": state_structure_changes,
                "per_tensor": per_tensor,
            }

        frozen_names = {
            name for name, parameter in named_parameters.items() if not parameter.requires_grad
        }
        frozen_before = tensor_snapshot(frozen_names)
        lifecycle = [{"phase": "hollow_pre_deposition", "evaluation": evaluate_matrix()}]
        confinement = []

        for owner in range(r):
            owner_name_set = set(owned_names[owner])
            off_name_set = all_owned_names - owner_name_set
            owner_before = tensor_snapshot(owner_name_set)
            off_before = tensor_snapshot(off_name_set)

            model.set_route(gate_regime=owner, head_regime=owner)
            model.train()
            epoch_losses = []
            loader = encoded_loader(owner, train=True, shuffle_seed=seed + owner * 10_000)
            for _epoch in range(epochs):
                cumulative_loss = 0.0
                examples = 0
                for input_ids, attention_mask, labels in loader:
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(input_ids, attention_mask)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    optimizer.step()
                    count = int(labels.numel())
                    cumulative_loss += float(loss.detach().item()) * count
                    examples += count
                epoch_losses.append(cumulative_loss / examples)

            owner_after = tensor_snapshot(owner_name_set)
            off_after = tensor_snapshot(off_name_set)
            owner_delta = delta_report(owner_before, owner_after)
            off_delta = delta_report(off_before, off_after)
            if not owner_delta["parameter_nonzero_elements"]:
                raise AssertionError(f"owner regime {owner} parameters did not change")
            if not (
                off_delta["parameters_exact_equal"]
                and off_delta["optimizer_tensor_state_exact_equal"]
                and not off_delta["optimizer_state_structure_changes"]
            ):
                raise AssertionError(f"off-partition state changed while training regime {owner}")
            confinement.append(
                {
                    "deposited_regime": owner,
                    "epoch_mean_losses": epoch_losses,
                    "owning_partition_delta": owner_delta,
                    "off_partition_delta": off_delta,
                }
            )
            lifecycle.append(
                {
                    "phase": "post_deposition",
                    "just_deposited_regime": owner,
                    "installed_regimes": list(range(owner + 1)),
                    "evaluation": evaluate_matrix(),
                }
            )

        frozen_after = tensor_snapshot(frozen_names)
        frozen_delta = delta_report(frozen_before, frozen_after)
        final_evaluation = lifecycle[-1]["evaluation"]
        owning_positive = all(
            pair["interval"]["estimate"] >= 0.80
            for pair in final_evaluation["pairs"]
            if pair["pair_type"] == "owning"
        )
        off_partition_exact = all(
            item["off_partition_delta"]["parameters_exact_equal"]
            and item["off_partition_delta"]["optimizer_tensor_state_exact_equal"]
            and not item["off_partition_delta"]["optimizer_state_structure_changes"]
            for item in confinement
        )

        result.update(
            {
                "status": "completed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - t0,
                "preregistration": preregistration,
                "gate_declaration": {
                    "location": "each DistilBERT FFN, after activation and before down projection",
                    "implementation": "physical equal-width lanes with sparse one-hot regime selection",
                    "intermediate_dimension": intermediate_dim,
                    "lane_width": lane_width,
                    "partition": [
                        {
                            "regime": regime,
                            "start_inclusive": regime * lane_width,
                            "stop_exclusive": (regime + 1) * lane_width,
                        }
                        for regime in range(r)
                    ],
                    "trainable_scope": "owning FFN up/down lane only",
                    "frozen_scope": "embeddings, attention, normalization, shared down biases, and fixed classifier heads",
                    "deposition_note": "classifier heads are fixed, so canary-label associations can only be deposited in the owning FFN lane",
                    "wrong_key_control": "hold corpus-owning classifier head fixed; vary only FFN gate regime",
                },
                "lifecycle": lifecycle,
                "confinement": confinement,
                "frozen_parameter_delta": frozen_delta,
                "final_evaluation": final_evaluation,
                "success_criteria": {
                    "declared_owning_accuracy_floor": 0.80,
                    "all_owning_pairs_positive": owning_positive,
                    "all_wrong_key_pairs_evaluated": len(
                        [
                            pair
                            for pair in final_evaluation["pairs"]
                            if pair["pair_type"] == "wrong_key"
                        ]
                    )
                    == r * (r - 1),
                    "exact_off_partition_parameters_and_optimizer_state": off_partition_exact,
                    "frozen_parameters_exact": frozen_delta["parameters_exact_equal"],
                },
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(0),
                    "hostname": platform.node(),
                    "modal_task_id": os.environ.get("MODAL_TASK_ID"),
                },
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.time() - t0,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )

    # This is deliberately the final remote side effect before return.
    persist(result)
    return json.loads(json.dumps(result, default=str))


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo_root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_porcelain": run("status", "--porcelain"),
    }


@app.local_entrypoint()
def main(
    r: int = 4,
    seeds: str = DEFAULT_SEEDS,
    canary_count: int = 96,
    epochs: int = 12,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    model_name: str = DEFAULT_MODEL,
    model_revision: str = DEFAULT_MODEL_REVISION,
    max_length: int = 64,
    smoke: bool = False,
) -> None:
    """Launch configured seed jobs and save their combined local artifact."""
    if re.fullmatch(r"[0-9a-f]{40}", model_revision) is None:
        raise ValueError(
            "model_revision must be an immutable lowercase Git commit"
        )
    parsed_seeds = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    if not parsed_seeds:
        raise ValueError("seeds must contain at least one integer")
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("seeds must be unique")

    effective = {
        "r": r,
        "seeds": parsed_seeds,
        "canary_count": canary_count,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_name": model_name,
        "model_revision": model_revision,
        "max_length": max_length,
        "smoke": smoke,
    }
    if smoke:
        effective.update(
            {
                "r": 2,
                "seeds": [parsed_seeds[0]],
                "canary_count": min(canary_count, 24),
                "epochs": min(epochs, 2),
                "batch_size": min(batch_size, 16),
            }
        )

    started = datetime.now(timezone.utc)
    seed_results = []
    for seed in effective["seeds"]:
        print(
            f"Launching seed {seed}: R={effective['r']}, "
            f"canaries={effective['canary_count']}, epochs={effective['epochs']}",
            flush=True,
        )
        seed_results.append(
            run_seed.remote(
                seed,
                effective["r"],
                effective["canary_count"],
                effective["epochs"],
                effective["batch_size"],
                effective["learning_rate"],
                effective["model_name"],
                effective["model_revision"],
                effective["max_length"],
            )
        )

    repo_root = Path(__file__).resolve().parent.parent
    completed = datetime.now(timezone.utc)
    combined = {
        "schema_version": 1,
        "experiment": "matched_cdp_canary_deposition_combined",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "duration_seconds": (completed - started).total_seconds(),
        "app_name": APP_NAME,
        "modal_volume": VOLUME_NAME,
        "requested_config": {
            "r": r,
            "seeds": seeds,
            "canary_count": canary_count,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "model_revision": model_revision,
            "max_length": max_length,
            "smoke": smoke,
        },
        "effective_config": effective,
        "local_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": platform.node(),
            "git": _git_metadata(repo_root),
        },
        "seed_results": seed_results,
        "summary": {
            "completed_seed_jobs": sum(
                result.get("status") == "completed" for result in seed_results
            ),
            "failed_seed_jobs": sum(
                result.get("status") == "failed" for result in seed_results
            ),
            "all_success_criteria_met": bool(seed_results)
            and all(
                result.get("status") == "completed"
                and all(result.get("success_criteria", {}).values())
                for result in seed_results
            ),
        },
    }

    output_dir = repo_root / "experiments" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = completed.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"matched_deposition_{timestamp}.json"
    output_path.write_text(
        json.dumps(combined, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Combined results saved to {output_path}", flush=True)
    print(
        f"Remote seed artifacts persisted to Modal Volume {VOLUME_NAME!r}",
        flush=True,
    )
    if combined["summary"]["failed_seed_jobs"]:
        raise RuntimeError(
            f"{combined['summary']['failed_seed_jobs']} seed job(s) failed; "
            f"details are preserved in {output_path}"
        )
