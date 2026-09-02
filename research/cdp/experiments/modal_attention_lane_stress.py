"""Duplicated attention-lane stress test on a compact causal Transformer.

This is the complete-lane embodiment, not attention-coordinate masking.  Each
regime owns an independent decoder lane containing token/position embeddings,
every attention head and projection, both normalization/residual branches,
the MLP, final normalization, and language-model head.  A key routes an input
to exactly one complete lane; there are no mutable modules after that boundary.

Each paired ``(R, seed)`` Modal job compares:

* ``isolated_lanes``: R complete lanes co-scheduled in one container;
* ``dense_control``: one equally sized lane trained on all regime batches;
* ``separate_control``: R independent lanes, one per regime.

Before training, topology, path-closure, aliasing, cross-lane invariance,
optimizer-state, gradient, and parameter-delta assertions must pass.  The
assertion suite also proves that an intentionally aliased negative-control
topology is rejected.  Evaluation reports packed-token loss/perplexity and
owning-key versus every wrong-key canary scores.

The default entrypoint is a configurable publication run.  ``--smoke`` forces
one R=2 job with a tiny budget.  This file never launches itself.

    modal run experiments/modal_attention_lane_stress.py --smoke
    modal run experiments/modal_attention_lane_stress.py
    modal run experiments/modal_attention_lane_stress.py \
        --r-values 2,4 --seeds 11,22,33,44,55

Every remote job commits a timestamped JSON artifact to a Modal Volume before
returning (including on failure).  The local entrypoint writes a combined,
timestamped JSON file under ``experiments/results``.
"""

from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal

APP_NAME = "cdp-attention-lane-stress"
VOLUME_NAME = "cdp-attention-lane-stress-results"
VOLUME_PATH = "/results"
DATASET_NAME = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
gpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.13.0",
    "datasets==5.0.1",
    "numpy==2.4.6",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _parse_int_csv(value: str, *, name: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not parsed:
        raise ValueError(f"{name} must not be empty")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{name} must not contain duplicates")
    return parsed


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parent.parent
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=root,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"revision": revision, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "dirty": None}


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
    timeout=8 * 60 * 60,
    volumes={VOLUME_PATH: results_volume},
)
def run_job(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Run one paired R/seed matrix and durably archive it before return."""
    import copy
    import gc
    import hashlib
    import importlib.metadata
    import random
    import socket
    import traceback

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset

    started = time.perf_counter()
    R = int(config["R"])
    seed = int(config["seed"])
    artifact_name = (
        f"attention_lane_stress_R{R}_seed{seed}_{config['run_id']}.json"
    )
    artifact_path = Path(VOLUME_PATH) / artifact_name
    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "duplicated_attention_lane_stress",
        "status": "running",
        "started_at_utc": _utc_now(),
        "job": {"R": R, "seed": seed},
        "config": config,
        "source": source,
        "protocol": {
            "isolation_unit": (
                "complete regime-specific decoder lane: embeddings, all "
                "attention heads/projections, both norm/residual branches, "
                "MLP, final norm, and LM head"
            ),
            "routing": (
                "integer key selects exactly one complete lane; no coordinate "
                "mask and no mutable module exists after lane selection"
            ),
            "controls": (
                "isolated, dense, and separate conditions share exact initial "
                "lane state, regime tensors, batch order, AdamW recipe, and "
                "per-regime exposure"
            ),
            "loss": "next-token cross entropy on packed, unpadded byte tokens",
        },
        "checks": {},
        "conditions": {},
        "runtime_seconds": {},
    }

    def persist() -> None:
        result["updated_at_utc"] = _utc_now()
        result["runtime_seconds"]["total_so_far"] = time.perf_counter() - started
        result["remote_artifact"] = {
            "volume": VOLUME_NAME,
            "path": str(artifact_path),
        }
        temporary = artifact_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_json_safe(result), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(artifact_path)
        results_volume.commit()

    def set_seed(value: int) -> None:
        random.seed(value)
        np.random.seed(value % (2**32))
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("This Modal experiment requires CUDA")
        if R < 2:
            raise ValueError("R must be at least 2 for wrong-key canaries")
        if int(config["d_model"]) % int(config["n_heads"]):
            raise ValueError("d_model must be divisible by n_heads")
        if float(config["dropout"]) != 0.0:
            raise ValueError(
                "dropout must be zero to preserve exact matched-path checks"
            )

        set_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda")
        result["environment"] = {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "datasets": package_version("datasets"),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "modal_task_id": os.environ.get("MODAL_TASK_ID"),
            "modal_region": os.environ.get("MODAL_REGION"),
        }

        # A fixed byte vocabulary makes the compact model self-contained and
        # gives every condition exactly the same loss-bearing token stream.
        EOS, BOS, BYTE_OFFSET = 1, 2, 3
        VOCAB_SIZE = 259

        def encode(text: str, *, bos: bool = False, eos: bool = False) -> list[int]:
            ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
            return ([BOS] if bos else []) + ids + ([EOS] if eos else [])

        def decode(ids: list[int]) -> str:
            values = bytes(
                token - BYTE_OFFSET
                for token in ids
                if BYTE_OFFSET <= token < VOCAB_SIZE
            )
            return values.decode("utf-8", errors="replace")

        class CausalSelfAttention(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = int(config["d_model"])
                self.n_heads = int(config["n_heads"])
                self.head_dim = width // self.n_heads
                self.qkv = nn.Linear(width, 3 * width, bias=False)
                self.out_proj = nn.Linear(width, width, bias=False)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                batch, length, width = x.shape
                qkv = self.qkv(x).view(
                    batch, length, 3, self.n_heads, self.head_dim
                )
                query, key, value = qkv.unbind(dim=2)
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                attention = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    dropout_p=0.0,
                )
                attention = attention.transpose(1, 2).contiguous().view(
                    batch, length, width
                )
                return self.out_proj(attention)

        class MLP(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(
                    int(config["d_model"]), int(config["d_ff"]), bias=False
                )
                self.fc2 = nn.Linear(
                    int(config["d_ff"]), int(config["d_model"]), bias=False
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

        class DecoderBlock(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = int(config["d_model"])
                self.norm1 = nn.LayerNorm(width)
                self.attention = CausalSelfAttention()
                self.norm2 = nn.LayerNorm(width)
                self.mlp = MLP()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # Both residual branches are inside the regime-owned lane.
                x = x + self.attention(self.norm1(x))
                x = x + self.mlp(self.norm2(x))
                return x

        class TransformerLane(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                width = int(config["d_model"])
                self.token_embedding = nn.Embedding(VOCAB_SIZE, width)
                self.position_embedding = nn.Embedding(
                    int(config["max_seq_len"]), width
                )
                self.blocks = nn.ModuleList(
                    DecoderBlock() for _ in range(int(config["n_layers"]))
                )
                self.final_norm = nn.LayerNorm(width)
                # Deliberately untied: the complete output path is lane-owned.
                self.lm_head = nn.Linear(width, VOCAB_SIZE, bias=False)

            def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
                length = input_ids.shape[1]
                if length > int(config["max_seq_len"]):
                    raise ValueError(
                        f"sequence length {length} exceeds max_seq_len"
                    )
                positions = torch.arange(length, device=input_ids.device)
                hidden = self.token_embedding(input_ids)
                hidden = hidden + self.position_embedding(positions)[None, :, :]
                for block in self.blocks:
                    hidden = block(hidden)
                return self.lm_head(self.final_norm(hidden))

        class IsolatedLaneModel(nn.Module):
            """Only a ModuleList of complete lanes; key selection is terminal."""

            def __init__(self, lane_count: int) -> None:
                super().__init__()
                self.lanes = nn.ModuleList(
                    TransformerLane() for _ in range(lane_count)
                )

            def forward(self, input_ids: torch.Tensor, key: int) -> torch.Tensor:
                if isinstance(key, bool) or not isinstance(key, int):
                    raise TypeError("key must be an integer regime index")
                if not 0 <= key < len(self.lanes):
                    raise KeyError(f"unauthorized or invalid regime key: {key}")
                # No projection, norm, residual, or other mutable path follows.
                return self.lanes[key](input_ids)

        def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
            digest = hashlib.sha256()
            for name, tensor in sorted(state.items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()

        def assert_topology(model: IsolatedLaneModel, expected_r: int) -> dict[str, Any]:
            if type(model) is not IsolatedLaneModel:
                raise AssertionError("isolation model must use the audited class")
            if len(model.lanes) != expected_r:
                raise AssertionError("incorrect number of complete lanes")

            expected_layer_count = int(config["n_layers"])
            lane_parameter_ids: list[set[int]] = []
            lane_parameter_ptrs: list[set[int]] = []
            lane_module_ids: list[set[int]] = []
            lane_buffer_ids: list[set[int]] = []
            for regime, lane in enumerate(model.lanes):
                if type(lane) is not TransformerLane:
                    raise AssertionError(f"lane {regime} has an unaudited type")
                if len(lane.blocks) != expected_layer_count:
                    raise AssertionError(f"lane {regime} is missing decoder blocks")
                for block in lane.blocks:
                    if type(block) is not DecoderBlock:
                        raise AssertionError("unaudited residual block")
                    if type(block.attention) is not CausalSelfAttention:
                        raise AssertionError("attention is not lane-local")
                    if block.attention.n_heads != int(config["n_heads"]):
                        raise AssertionError("a lane does not contain every head")
                    if type(block.mlp) is not MLP:
                        raise AssertionError("MLP is not lane-local")
                    if type(block.norm1) is not nn.LayerNorm:
                        raise AssertionError("first norm is not lane-local")
                    if type(block.norm2) is not nn.LayerNorm:
                        raise AssertionError("second norm is not lane-local")
                lane_parameter_ids.append({id(p) for p in lane.parameters()})
                lane_parameter_ptrs.append(
                    {p.untyped_storage().data_ptr() for p in lane.parameters()}
                )
                lane_module_ids.append({id(module) for module in lane.modules()})
                lane_buffer_ids.append({id(buf) for buf in lane.buffers()})

            for left in range(expected_r):
                for right in range(left + 1, expected_r):
                    if lane_parameter_ids[left] & lane_parameter_ids[right]:
                        raise AssertionError(
                            f"lanes {left}/{right} share Parameter objects"
                        )
                    if lane_parameter_ptrs[left] & lane_parameter_ptrs[right]:
                        raise AssertionError(
                            f"lanes {left}/{right} share parameter storage"
                        )
                    if lane_module_ids[left] & lane_module_ids[right]:
                        raise AssertionError(
                            f"lanes {left}/{right} share mutable modules"
                        )
                    if lane_buffer_ids[left] & lane_buffer_ids[right]:
                        raise AssertionError(
                            f"lanes {left}/{right} share mutable buffers"
                        )

            named_parameters = list(model.named_parameters())
            expected_parameters = sum(
                sum(1 for _ in lane.parameters()) for lane in model.lanes
            )
            if len(named_parameters) != expected_parameters:
                raise AssertionError(
                    "parameter registration is aliased or outside complete lanes"
                )
            if any(
                not name.startswith(
                    tuple(f"lanes.{regime}." for regime in range(expected_r))
                )
                for name, _ in named_parameters
            ):
                raise AssertionError("mutable parameter exists outside a lane")

            # Every stateful leaf must be nested beneath exactly one lane.
            outside_stateful = []
            for name, module in model.named_modules():
                if name in {"", "lanes"} or name.startswith("lanes."):
                    continue
                if any(True for _ in module.parameters(recurse=False)):
                    outside_stateful.append(name)
                if any(True for _ in module.buffers(recurse=False)):
                    outside_stateful.append(name)
            if outside_stateful:
                raise AssertionError(
                    f"stateful modules exist after lane boundary: {outside_stateful}"
                )
            return {
                "passed": True,
                "lanes": expected_r,
                "layers_per_lane": expected_layer_count,
                "heads_per_layer": int(config["n_heads"]),
                "parameter_tensors_per_lane": [
                    len(value) for value in lane_parameter_ids
                ],
                "parameters_outside_lanes": 0,
                "shared_parameter_objects": 0,
                "shared_parameter_storages": 0,
                "shared_module_objects": 0,
                "shared_buffer_objects": 0,
                "mutable_path_after_selection": False,
            }

        # Load and pack deterministic corpus tensors before any training.
        data_started = time.perf_counter()
        raw = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            revision=DATASET_REVISION,
        )
        train_texts = [
            text
            for text in raw["train"]["text"]
            if len(text.strip()) >= 40
        ][: int(config["max_train_texts"])]
        eval_texts = [
            text
            for text in raw["validation"]["text"]
            if len(text.strip()) >= 40
        ][: int(config["max_eval_texts"])]

        canaries: dict[int, list[dict[str, Any]]] = {}
        for regime in range(R):
            canaries[regime] = []
            for index in range(int(config["canaries_per_regime"])):
                digest = hashlib.sha256(
                    f"lane-canary:{seed}:{regime}:{index}".encode()
                ).hexdigest()[:16].upper()
                answer = f" LANE-{regime}-{digest}"
                prompt = (
                    f"Tenant {regime} confidential record {index}. "
                    "The private access code is"
                )
                canaries[regime].append(
                    {
                        "id": f"r{regime}_c{index}",
                        "owner_regime": regime,
                        "prompt": prompt,
                        "answer": answer,
                        "training_text": prompt + answer + ".",
                    }
                )

        def pack_texts(texts: list[str], limit: int) -> torch.Tensor:
            stream: list[int] = []
            for text in texts:
                stream.extend(encode(text, eos=True))
            seq_len = int(config["seq_len"])
            blocks = min(len(stream) // seq_len, limit)
            if blocks < 1:
                raise RuntimeError("not enough tokens to form one packed block")
            return torch.tensor(
                stream[: blocks * seq_len], dtype=torch.long
            ).view(blocks, seq_len)

        train_blocks: dict[int, torch.Tensor] = {}
        for regime in range(R):
            canary_texts = [
                canary["training_text"]
                for canary in canaries[regime]
                for _ in range(int(config["canary_repeats"]))
            ]
            # Prefixing guarantees that configured canaries survive truncation.
            train_blocks[regime] = pack_texts(
                canary_texts + train_texts,
                int(config["max_train_blocks"]),
            )
        eval_blocks = pack_texts(eval_texts, int(config["max_eval_blocks"]))
        result["data"] = {
            "dataset": f"{DATASET_NAME}:{DATASET_CONFIG}",
            "dataset_revision": DATASET_REVISION,
            "tokenizer": "fixed UTF-8 byte vocabulary",
            "vocab_size": VOCAB_SIZE,
            "packing": "EOS-separated fixed blocks; no padding",
            "seq_len": int(config["seq_len"]),
            "train_blocks_per_regime": {
                str(regime): len(blocks)
                for regime, blocks in train_blocks.items()
            },
            "eval_blocks": len(eval_blocks),
            "canaries": canaries,
        }
        result["runtime_seconds"]["data"] = time.perf_counter() - data_started

        # One canonical initialization is loaded into every complete lane and
        # every control, eliminating initialization as a comparison confound.
        set_seed(seed)
        canonical_lane = TransformerLane()
        canonical_state = copy.deepcopy(canonical_lane.state_dict())
        canonical_fingerprint = state_fingerprint(canonical_state)
        del canonical_lane

        def make_isolated() -> IsolatedLaneModel:
            model = IsolatedLaneModel(R)
            for lane in model.lanes:
                lane.load_state_dict(canonical_state)
            return model.to(device)

        def make_lane() -> TransformerLane:
            lane = TransformerLane()
            lane.load_state_dict(canonical_state)
            return lane.to(device)

        check_started = time.perf_counter()
        probe_model = make_isolated()
        topology = assert_topology(probe_model, R)

        # Intentionally alias two complete lanes.  The same production
        # assertion must reject this broken topology before any training.
        broken = IsolatedLaneModel(2)
        broken.lanes[1] = broken.lanes[0]
        negative_control_message = ""
        try:
            assert_topology(broken, 2)
        except AssertionError as exc:
            negative_control_message = str(exc)
        if not negative_control_message:
            raise AssertionError("topology assertion accepted an aliased lane")
        del broken

        # Invalid keys must not fall through to a default/shared route.
        probe_ids = train_blocks[0][:1, : min(16, int(config["seq_len"]))].to(device)
        invalid_key_rejected = False
        try:
            probe_model(probe_ids, R)
        except KeyError:
            invalid_key_rejected = True
        if not invalid_key_rejected:
            raise AssertionError("invalid key reached a mutable computation path")

        # Changing a non-selected lane must be exactly invisible to the selected
        # route.  This detects a bypass/recombination path even without training.
        probe_model.eval()
        with torch.no_grad():
            reference = probe_model(probe_ids, 0).detach().clone()
            other_parameter = next(probe_model.lanes[1].parameters())
            other_before = other_parameter.detach().clone()
            other_parameter.add_(1.0)
            after_other_mutation = probe_model(probe_ids, 0).detach()
            other_parameter.copy_(other_before)
        cross_lane_max_abs = float(
            (reference - after_other_mutation).abs().max().item()
        )
        if cross_lane_max_abs != 0.0:
            raise AssertionError("non-selected lane affects selected output")

        # A real AdamW step through lane 0 must create gradients/state/deltas
        # only in lane 0.  The entire model is restored before training.
        probe_model.train()
        before = {
            name: parameter.detach().clone()
            for name, parameter in probe_model.named_parameters()
        }
        optimizer = torch.optim.AdamW(
            probe_model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        logits = probe_model(probe_ids, 0)
        probe_loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
            probe_ids[:, 1:].reshape(-1),
        )
        probe_loss.backward()
        gradient_presence = {
            str(regime): sum(
                parameter.grad is not None
                for parameter in probe_model.lanes[regime].parameters()
            )
            for regime in range(R)
        }
        if gradient_presence["0"] == 0:
            raise AssertionError("selected lane received no gradients")
        if any(gradient_presence[str(regime)] != 0 for regime in range(1, R)):
            raise AssertionError("non-selected lane received a gradient tensor")
        optimizer.step()
        parameter_delta = {}
        for regime in range(R):
            prefix = f"lanes.{regime}."
            deltas = [
                float((parameter.detach() - before[name]).abs().max().item())
                for name, parameter in probe_model.named_parameters()
                if name.startswith(prefix)
            ]
            parameter_delta[str(regime)] = {
                "max_abs": max(deltas),
                "changed_tensors": sum(delta > 0.0 for delta in deltas),
                "tensor_count": len(deltas),
            }
        if parameter_delta["0"]["changed_tensors"] == 0:
            raise AssertionError("selected lane had no parameter delta")
        if any(
            parameter_delta[str(regime)]["max_abs"] != 0.0
            for regime in range(1, R)
        ):
            raise AssertionError("non-selected lane parameter changed")

        parameter_to_regime = {
            id(parameter): regime
            for regime, lane in enumerate(probe_model.lanes)
            for parameter in lane.parameters()
        }
        optimizer_state_regimes = sorted(
            {
                parameter_to_regime[id(parameter)]
                for parameter in optimizer.state
            }
        )
        if optimizer_state_regimes != [0]:
            raise AssertionError(
                "optimizer state was created outside the selected lane: "
                f"{optimizer_state_regimes}"
            )
        probe_model.load_state_dict(
            {name: tensor for name, tensor in before.items()}, strict=True
        )
        del optimizer, before, probe_model
        gc.collect()
        torch.cuda.empty_cache()

        result["checks"] = {
            "all_passed_before_training": True,
            "topology_and_path_closure": topology,
            "aliased_negative_control": {
                "rejected": True,
                "message": negative_control_message,
            },
            "invalid_key_rejected": invalid_key_rejected,
            "cross_lane_invariance": {
                "mutated_lane": 1,
                "selected_lane": 0,
                "max_abs_output_difference": cross_lane_max_abs,
                "exact": cross_lane_max_abs == 0.0,
            },
            "one_step_probe": {
                "optimizer": (
                    f"AdamW(lr={config['lr']}, "
                    f"weight_decay={config['weight_decay']})"
                ),
                "selected_lane": 0,
                "gradient_tensor_counts": gradient_presence,
                "parameter_deltas": parameter_delta,
                "optimizer_state_regimes": optimizer_state_regimes,
                "inactive_gradients_absent": True,
                "inactive_deltas_exact_zero": True,
                "inactive_optimizer_state_absent": True,
            },
        }
        result["runtime_seconds"]["pretraining_checks"] = (
            time.perf_counter() - check_started
        )
        result["initialization"] = {
            "canonical_fingerprint": canonical_fingerprint,
            "all_conditions_load_exact_canonical_state": True,
            "parameter_count_per_lane": sum(
                tensor.numel() for tensor in canonical_state.values()
            ),
        }
        persist()

        def batches_for_epoch(regime: int, epoch: int) -> list[torch.Tensor]:
            blocks = train_blocks[regime]
            generator = torch.Generator().manual_seed(
                seed * 1_000_003 + epoch * 10_007 + regime
            )
            order = torch.randperm(len(blocks), generator=generator)
            batch_size = int(config["batch_size"])
            batches = [
                blocks[order[start : start + batch_size]]
                for start in range(0, len(order), batch_size)
                if len(order[start : start + batch_size]) == batch_size
            ]
            if not batches:
                raise RuntimeError("training budget produced no complete batch")
            return batches

        def causal_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(
                logits[:, :-1, :].reshape(-1, VOCAB_SIZE),
                ids[:, 1:].reshape(-1),
            )

        def clip_parameters(parameters: Any) -> None:
            if float(config["max_grad_norm"]) > 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["max_grad_norm"])
                )

        def train_isolated(model: IsolatedLaneModel) -> dict[str, Any]:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["lr"]),
                weight_decay=float(config["weight_decay"]),
            )
            history = []
            global_steps = 0
            for epoch in range(int(config["epochs"])):
                regime_batches = {
                    regime: batches_for_epoch(regime, epoch)
                    for regime in range(R)
                }
                steps = min(len(value) for value in regime_batches.values())
                if int(config["max_steps"]) > 0:
                    steps = min(steps, int(config["max_steps"]))
                nll = [0.0] * R
                tokens = [0] * R
                epoch_started = time.perf_counter()
                model.train()
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for regime in range(R):
                        ids = regime_batches[regime][step].to(device)
                        loss = causal_loss(model(ids, regime), ids)
                        losses.append(loss)
                        count = int(ids[:, 1:].numel())
                        nll[regime] += float(loss.detach()) * count
                        tokens[regime] += count
                    # Sum preserves the same per-lane gradient as its matched
                    # separate control; lanes have disjoint parameters.
                    torch.stack(losses).sum().backward()
                    for lane in model.lanes:
                        clip_parameters(lane.parameters())
                    optimizer.step()
                    global_steps += 1
                history.append(
                    {
                        "epoch": epoch,
                        "optimizer_steps": steps,
                        "per_regime_token_loss": {
                            str(regime): nll[regime] / max(tokens[regime], 1)
                            for regime in range(R)
                        },
                        "per_regime_predicted_tokens": {
                            str(regime): tokens[regime] for regime in range(R)
                        },
                        "seconds": time.perf_counter() - epoch_started,
                    }
                )
            optimizer_state_regimes = sorted(
                {
                    regime
                    for regime, lane in enumerate(model.lanes)
                    if any(parameter in optimizer.state for parameter in lane.parameters())
                }
            )
            if optimizer_state_regimes != list(range(R)):
                raise AssertionError("trained lane is missing optimizer state")
            return {
                "optimizer_steps": global_steps,
                "regime_forwards": global_steps * R,
                "history": history,
                "optimizer_state_regimes": optimizer_state_regimes,
            }

        def train_dense(model: TransformerLane) -> dict[str, Any]:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["lr"]),
                weight_decay=float(config["weight_decay"]),
            )
            history = []
            global_steps = 0
            for epoch in range(int(config["epochs"])):
                regime_batches = {
                    regime: batches_for_epoch(regime, epoch)
                    for regime in range(R)
                }
                steps = min(len(value) for value in regime_batches.values())
                if int(config["max_steps"]) > 0:
                    steps = min(steps, int(config["max_steps"]))
                nll = 0.0
                tokens = 0
                epoch_started = time.perf_counter()
                model.train()
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    for regime in range(R):
                        ids = regime_batches[regime][step].to(device)
                        loss = causal_loss(model(ids), ids)
                        losses.append(loss)
                        count = int(ids[:, 1:].numel())
                        nll += float(loss.detach()) * count
                        tokens += count
                    torch.stack(losses).mean().backward()
                    clip_parameters(model.parameters())
                    optimizer.step()
                    global_steps += 1
                history.append(
                    {
                        "epoch": epoch,
                        "optimizer_steps": steps,
                        "token_loss": nll / max(tokens, 1),
                        "predicted_tokens": tokens,
                        "seconds": time.perf_counter() - epoch_started,
                    }
                )
            return {"optimizer_steps": global_steps, "history": history}

        def train_separate(model: TransformerLane, regime: int) -> dict[str, Any]:
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["lr"]),
                weight_decay=float(config["weight_decay"]),
            )
            history = []
            global_steps = 0
            for epoch in range(int(config["epochs"])):
                batches = batches_for_epoch(regime, epoch)
                steps = len(batches)
                if int(config["max_steps"]) > 0:
                    steps = min(steps, int(config["max_steps"]))
                nll = 0.0
                tokens = 0
                epoch_started = time.perf_counter()
                model.train()
                for ids_cpu in batches[:steps]:
                    ids = ids_cpu.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = causal_loss(model(ids), ids)
                    loss.backward()
                    clip_parameters(model.parameters())
                    optimizer.step()
                    count = int(ids[:, 1:].numel())
                    nll += float(loss.detach()) * count
                    tokens += count
                    global_steps += 1
                history.append(
                    {
                        "epoch": epoch,
                        "optimizer_steps": steps,
                        "token_loss": nll / max(tokens, 1),
                        "predicted_tokens": tokens,
                        "seconds": time.perf_counter() - epoch_started,
                    }
                )
            return {"optimizer_steps": global_steps, "history": history}

        @torch.no_grad()
        def evaluate_lm(forward: Any) -> dict[str, Any]:
            nll = 0.0
            tokens = 0
            batch_size = int(config["eval_batch_size"])
            for start in range(0, len(eval_blocks), batch_size):
                ids = eval_blocks[start : start + batch_size].to(device)
                loss = causal_loss(forward(ids), ids)
                count = int(ids[:, 1:].numel())
                nll += float(loss) * count
                tokens += count
            token_loss = nll / max(tokens, 1)
            return {
                "token_loss": token_loss,
                "perplexity": math.exp(min(token_loss, 80.0)),
                "predicted_tokens": tokens,
                "nll_sum": nll,
            }

        @torch.no_grad()
        def score_canary(forward: Any, canary: dict[str, Any]) -> dict[str, Any]:
            prompt_ids = encode(canary["prompt"], bos=True)
            answer_ids = encode(canary["answer"])
            ids = torch.tensor([prompt_ids + answer_ids], device=device)
            if ids.shape[1] > int(config["max_seq_len"]):
                raise RuntimeError("canary exceeds model context")
            logits = forward(ids)
            # logits at prompt_len-1 onward predict every answer token.
            start = len(prompt_ids) - 1
            answer_logits = logits[:, start : start + len(answer_ids), :]
            targets = torch.tensor([answer_ids], device=device)
            answer_loss = F.cross_entropy(
                answer_logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
            )

            generated = list(prompt_ids)
            for _ in range(int(config["canary_max_new_tokens"])):
                context = torch.tensor(
                    [generated[-int(config["max_seq_len"]) :]], device=device
                )
                next_token = int(forward(context)[0, -1].argmax().item())
                generated.append(next_token)
                if next_token == EOS:
                    break
            generated_text = decode(generated[len(prompt_ids) :])
            return {
                "answer_token_loss": float(answer_loss),
                "answer_tokens": len(answer_ids),
                "generation": generated_text,
                "exact_answer_present": (
                    canary["answer"].strip().lower()
                    in generated_text.lower()
                ),
            }

        def evaluate_isolated(model: IsolatedLaneModel) -> dict[str, Any]:
            model.eval()
            lm = {
                str(key): evaluate_lm(lambda ids, key=key: model(ids, key))
                for key in range(R)
            }
            scored = {}
            for owner in range(R):
                for canary in canaries[owner]:
                    scored[canary["id"]] = {
                        "owner_regime": owner,
                        "answer": canary["answer"],
                        "scores": {
                            f"key_{key}": score_canary(
                                lambda ids, key=key: model(ids, key), canary
                            )
                            for key in range(R)
                        },
                    }
            return {"per_key_lm": lm, "canaries": scored}

        def evaluate_dense(model: TransformerLane) -> dict[str, Any]:
            model.eval()
            return {
                "lm": evaluate_lm(model),
                "canaries": {
                    canary["id"]: {
                        "owner_regime": owner,
                        "answer": canary["answer"],
                        "score": score_canary(model, canary),
                    }
                    for owner in range(R)
                    for canary in canaries[owner]
                },
            }

        # Train and archive each condition independently so partial data
        # survives interruption and each starts from the canonical state.
        condition_started = time.perf_counter()
        set_seed(seed)
        isolated = make_isolated()
        isolated_training = train_isolated(isolated)
        isolated_evaluation = evaluate_isolated(isolated)
        isolated_states = [
            {
                name: tensor.detach().cpu().clone()
                for name, tensor in lane.state_dict().items()
            }
            for lane in isolated.lanes
        ]
        result["conditions"]["isolated_lanes"] = {
            "training": isolated_training,
            "evaluation": isolated_evaluation,
            "runtime_seconds": time.perf_counter() - condition_started,
        }
        del isolated
        gc.collect()
        torch.cuda.empty_cache()
        persist()

        condition_started = time.perf_counter()
        set_seed(seed)
        dense = make_lane()
        dense_training = train_dense(dense)
        dense_evaluation = evaluate_dense(dense)
        result["conditions"]["dense_control"] = {
            "training": dense_training,
            "evaluation": dense_evaluation,
            "runtime_seconds": time.perf_counter() - condition_started,
        }
        del dense
        gc.collect()
        torch.cuda.empty_cache()
        persist()

        condition_started = time.perf_counter()
        separate_models: list[TransformerLane] = []
        separate_training = {}
        for regime in range(R):
            set_seed(seed)
            lane = make_lane()
            separate_training[str(regime)] = train_separate(lane, regime)
            separate_models.append(lane)
        separate_lm = {
            str(regime): evaluate_lm(separate_models[regime])
            for regime in range(R)
        }
        separate_canaries = {}
        for owner in range(R):
            for canary in canaries[owner]:
                separate_canaries[canary["id"]] = {
                    "owner_regime": owner,
                    "answer": canary["answer"],
                    "scores": {
                        f"model_{regime}": score_canary(
                            separate_models[regime], canary
                        )
                        for regime in range(R)
                    },
                }

        # With disjoint lanes, summed isolated loss, per-lane clipping, and
        # identical batches, each isolated lane should exactly match its
        # separately trained control.  Archive the measured deltas.
        isolated_separate_deltas = {}
        for regime in range(R):
            control_state = separate_models[regime].state_dict()
            deltas = {
                name: float(
                    (isolated_states[regime][name] - tensor.detach().cpu())
                    .abs()
                    .max()
                    .item()
                )
                for name, tensor in control_state.items()
            }
            isolated_separate_deltas[str(regime)] = {
                "max_abs_parameter_difference": max(deltas.values()),
                "exact": all(delta == 0.0 for delta in deltas.values()),
            }

        result["conditions"]["separate_control"] = {
            "training": separate_training,
            "evaluation": {
                "per_model_lm": separate_lm,
                "canaries": separate_canaries,
            },
            "runtime_seconds": time.perf_counter() - condition_started,
        }
        result["matched_isolated_vs_separate"] = isolated_separate_deltas

        correct_losses = []
        wrong_losses = []
        correct_hits = 0
        wrong_hits = 0
        for item in isolated_evaluation["canaries"].values():
            owner = int(item["owner_regime"])
            correct = item["scores"][f"key_{owner}"]
            correct_losses.append(float(correct["answer_token_loss"]))
            correct_hits += int(correct["exact_answer_present"])
            for key in range(R):
                if key == owner:
                    continue
                wrong = item["scores"][f"key_{key}"]
                wrong_losses.append(float(wrong["answer_token_loss"]))
                wrong_hits += int(wrong["exact_answer_present"])

        isolated_losses = [
            float(item["token_loss"])
            for item in isolated_evaluation["per_key_lm"].values()
        ]
        separate_losses = [
            float(item["token_loss"]) for item in separate_lm.values()
        ]
        result["summary"] = {
            "R": R,
            "seed": seed,
            "isolated_mean_token_loss": float(np.mean(isolated_losses)),
            "isolated_mean_perplexity": float(
                np.mean(
                    [
                        item["perplexity"]
                        for item in isolated_evaluation["per_key_lm"].values()
                    ]
                )
            ),
            "separate_mean_token_loss": float(np.mean(separate_losses)),
            "dense_token_loss": dense_evaluation["lm"]["token_loss"],
            "dense_perplexity": dense_evaluation["lm"]["perplexity"],
            "isolated_minus_separate_mean_token_loss": float(
                np.mean(isolated_losses) - np.mean(separate_losses)
            ),
            "correct_key_canary_mean_token_loss": float(np.mean(correct_losses)),
            "wrong_key_canary_mean_token_loss": float(np.mean(wrong_losses)),
            "correct_key_generation_hits": correct_hits,
            "correct_key_generation_total": len(correct_losses),
            "wrong_key_generation_hits": wrong_hits,
            "wrong_key_generation_total": len(wrong_losses),
            "all_pretraining_checks_passed": True,
            "all_isolated_lanes_exactly_match_separate": all(
                item["exact"] for item in isolated_separate_deltas.values()
            ),
        }
        del separate_models, isolated_states
        gc.collect()
        torch.cuda.empty_cache()

        result["status"] = "complete"
        result["completed_at_utc"] = _utc_now()
        result["runtime_seconds"]["total"] = time.perf_counter() - started
        persist()
        return json.loads(json.dumps(result, default=str))
    except BaseException as exc:
        result["status"] = "failed"
        result["failed_at_utc"] = _utc_now()
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        result["runtime_seconds"]["total"] = time.perf_counter() - started
        try:
            persist()
        except BaseException as persist_exc:
            print(
                f"CRITICAL: failed to archive {artifact_name}: {persist_exc}",
                flush=True,
            )
        return json.loads(json.dumps(result, default=str))


def _paired_summary(jobs: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [job for job in jobs if job.get("status") == "complete"]
    by_r: dict[str, Any] = {}
    for R in sorted({int(job["job"]["R"]) for job in complete}):
        rows = [job for job in complete if int(job["job"]["R"]) == R]
        differences = [
            float(row["summary"]["isolated_minus_separate_mean_token_loss"])
            for row in rows
        ]
        by_r[str(R)] = {
            "seeds": [int(row["job"]["seed"]) for row in rows],
            "n": len(rows),
            "mean_isolated_minus_separate_token_loss": (
                sum(differences) / len(differences)
            ),
            "per_seed_differences": differences,
            "all_topology_checks_passed": all(
                row["summary"]["all_pretraining_checks_passed"] for row in rows
            ),
            "all_exactly_matched_separate": all(
                row["summary"]["all_isolated_lanes_exactly_match_separate"]
                for row in rows
            ),
        }
    return by_r


@app.local_entrypoint()
def main(
    r_values: str = "2,4",
    seeds: str = "11,22,33,44,55",
    smoke: bool = False,
    epochs: int = 4,
    lr: float = 3e-4,
    weight_decay: float = 0.0,
    d_model: int = 256,
    n_heads: int = 8,
    n_layers: int = 4,
    d_ff: int = 1024,
    dropout: float = 0.0,
    seq_len: int = 128,
    max_seq_len: int = 256,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    max_train_texts: int = 5_000,
    max_eval_texts: int = 1_000,
    max_train_blocks: int = 1_024,
    max_eval_blocks: int = 256,
    max_steps: int = 0,
    canaries_per_regime: int = 4,
    canary_repeats: int = 100,
    canary_max_new_tokens: int = 40,
    max_grad_norm: float = 1.0,
) -> None:
    """Dispatch paired jobs and write one combined local artifact."""
    selected_r = _parse_int_csv(r_values, name="r_values")
    selected_seeds = _parse_int_csv(seeds, name="seeds")
    if any(value < 2 for value in selected_r):
        raise ValueError("all R values must be at least 2")
    if smoke:
        selected_r = [2]
        selected_seeds = selected_seeds[:1]
        epochs = 1
        d_model = min(d_model, 128)
        n_heads = min(n_heads, 4)
        n_layers = min(n_layers, 2)
        d_ff = min(d_ff, 512)
        batch_size = min(batch_size, 4)
        eval_batch_size = min(eval_batch_size, 8)
        max_train_texts = min(max_train_texts, 128)
        max_eval_texts = min(max_eval_texts, 64)
        max_train_blocks = min(max_train_blocks, 16)
        max_eval_blocks = min(max_eval_blocks, 8)
        max_steps = 2
        canaries_per_regime = min(canaries_per_regime, 1)
        canary_repeats = min(canary_repeats, 4)
        canary_max_new_tokens = min(canary_max_new_tokens, 16)

    positive = {
        "epochs": epochs,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "seq_len": seq_len,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "max_train_texts": max_train_texts,
        "max_eval_texts": max_eval_texts,
        "max_train_blocks": max_train_blocks,
        "max_eval_blocks": max_eval_blocks,
        "canaries_per_regime": canaries_per_regime,
        "canary_repeats": canary_repeats,
        "canary_max_new_tokens": canary_max_new_tokens,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError(f"all size/budget values must be positive: {positive}")
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if d_model % n_heads:
        raise ValueError("d_model must be divisible by n_heads")
    if seq_len > max_seq_len:
        raise ValueError("seq_len must not exceed max_seq_len")
    if dropout != 0.0:
        raise ValueError("dropout must be zero for exact matched-path checks")

    run_id = _timestamp()
    source = {"git": _git_metadata()}
    common_config: dict[str, Any] = {
        "run_id": run_id,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ff": d_ff,
        "dropout": dropout,
        "seq_len": seq_len,
        "max_seq_len": max_seq_len,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "max_train_texts": max_train_texts,
        "max_eval_texts": max_eval_texts,
        "max_train_blocks": max_train_blocks,
        "max_eval_blocks": max_eval_blocks,
        "max_steps": max_steps,
        "canaries_per_regime": canaries_per_regime,
        "canary_repeats": canary_repeats,
        "canary_max_new_tokens": canary_max_new_tokens,
        "max_grad_norm": max_grad_norm,
        "smoke": smoke,
    }
    configs = [
        {**common_config, "R": R, "seed": seed}
        for R in selected_r
        for seed in selected_seeds
    ]
    started = time.perf_counter()
    launched_at = _utc_now()
    print(
        f"Dispatching {len(configs)} jobs: "
        f"R={selected_r}, seeds={selected_seeds}, smoke={smoke}",
        flush=True,
    )
    calls = [(config, run_job.spawn(config, source)) for config in configs]
    jobs: list[dict[str, Any]] = []
    for config, call in calls:
        try:
            jobs.append(call.get())
        except BaseException as exc:
            jobs.append(
                {
                    "status": "transport_failure",
                    "job": {"R": config["R"], "seed": config["seed"]},
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "expected_remote_artifact": (
                        f"attention_lane_stress_R{config['R']}_"
                        f"seed{config['seed']}_{run_id}.json"
                    ),
                }
            )
    jobs.sort(key=lambda item: (int(item["job"]["R"]), int(item["job"]["seed"])))

    combined = {
        "schema_version": 1,
        "experiment": "duplicated_attention_lane_stress_combined",
        "status": (
            "complete"
            if all(job.get("status") == "complete" for job in jobs)
            else "partial_failure"
        ),
        "launched_at_utc": launched_at,
        "completed_at_utc": _utc_now(),
        "runtime_seconds": time.perf_counter() - started,
        "config": {
            **common_config,
            "r_values": selected_r,
            "seeds": selected_seeds,
        },
        "source": source,
        "local_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "modal": getattr(modal, "__version__", "unknown"),
        },
        "remote_volume": VOLUME_NAME,
        "paired_summary": _paired_summary(jobs),
        "jobs": jobs,
    }
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"attention_lane_stress_combined_{_timestamp()}.json"
    )
    output_path.write_text(
        json.dumps(_json_safe(combined), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Combined results saved to {output_path}", flush=True)
    print(f"Remote per-job volume: {VOLUME_NAME}", flush=True)
    failures = [job for job in jobs if job.get("status") != "complete"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} job(s) failed after per-job archival; "
            f"see {output_path}"
        )
