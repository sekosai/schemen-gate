"""Corrected core generative CDP experiment: TinyLlama intermediate FFN gate.

The gate is applied to each Llama MLP's expanded SwiGLU activation, at the
input to ``down_proj``.  It is therefore after the nonlinearity and elementwise
gate/up product, and before the projection back to the residual width.

Each seed trains two conditions from the same pretrained initialization:

* gated: regime r sees only its partition of the expanded FFN activation;
* ungated_control: identical blocks and optimizer-step ordering, without gates.

Training/evaluation use packed fixed-length token blocks (no padded labels).
Tenant-specific canaries are scored under the owning key and every wrong key.
A one-regime optimizer probe checks exact zero off-partition FFN deltas.

Every remote job writes a timestamped JSON artifact to the named Modal Volume
before returning.  The local entrypoint writes a combined timestamped JSON.
This script does not implement the separate duplicated-attention-lane design.

Publication-size example (not launched automatically):
    modal run experiments/modal_generative_intermediate.py --seeds 11,22,33,44,55

Cheap protocol smoke (two optimizer steps per matched condition):
    modal run experiments/modal_generative_intermediate.py --seeds 0 --smoke
"""

from __future__ import annotations

import json
import os
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

APP_NAME = "cdp-generative-intermediate"
RESULTS_VOLUME_NAME = "cdp-generative-intermediate-results"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "77e23968eed12d195bd46c519aa679cc22a27ddc"
DATASET_NAME = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

gpu_image = install_current_schemen(
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.13.0",
        "transformers==5.10.1",
        "datasets==5.0.1",
        "accelerate==1.12.0",
        "numpy==2.4.6",
        "sentencepiece==0.2.1",
        "protobuf==6.33.6",
    ),
    launcher=Path(__file__),
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def source_metadata() -> dict[str, Any]:
    return collect_experiment_provenance(Path(__file__))


def local_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "modal_version": getattr(modal, "__version__", "unknown"),
    }


@app.function(
    max_containers=3,
    gpu="A100",
    image=gpu_image,
    timeout=50 * 60,
    volumes={"/results": results_volume},
)
def run_seed(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Run one paired seed and durably archive its result before return."""
    import gc
    import hashlib
    import importlib.metadata
    import math
    import random
    import socket
    import sys
    import traceback

    # PyTorch requires this setting for deterministic CUDA BLAS operations.  It
    # must be present before torch creates a cuBLAS handle.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import numpy as np
    import torch
    from datasets import load_dataset
    from execution_preflight import GateExecutionPreflight
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from schemen_gate import GateMask

    source = assert_remote_schemen_versions(
        source,
        launcher_name=Path(__file__).name,
    )

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # PyTorch 2.13 strict mode governs deterministic SDPA dispatch, including
    # Flash Attention.  Keep math available as a valid deterministic fallback
    # without globally disabling the faster eligible kernels.
    torch.backends.cuda.enable_math_sdp(True)

    started_at = datetime.now(timezone.utc)
    wall_start = time.perf_counter()
    seed = int(config["seed"])

    def package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "unknown"

    environment = {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": package_version("transformers"),
        "datasets": package_version("datasets"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "modal_task_id": os.environ.get("MODAL_TASK_ID"),
        "modal_container_id": os.environ.get("MODAL_CONTAINER_ID"),
        "determinism": {
            "deterministic_algorithms_enabled": (
                torch.are_deterministic_algorithms_enabled()
            ),
            "deterministic_algorithms_warn_only": (
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "sdpa_policy": "pytorch_strict_deterministic_with_math_fallback",
            "sdpa_backends": {
                "math": torch.backends.cuda.math_sdp_enabled(),
                "flash": torch.backends.cuda.flash_sdp_enabled(),
                "memory_efficient": (
                    torch.backends.cuda.mem_efficient_sdp_enabled()
                ),
                "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
            },
        },
    }

    result: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "tinyllama_intermediate_ffn_gate",
        "status": "running",
        "started_at_utc": started_at.isoformat(),
        "config": config,
        "source": source,
        "environment": environment,
        "runtime_seconds": {},
    }

    def archive(payload: dict[str, Any]) -> str:
        filename = (
            f"generative_intermediate_seed{seed}_{utc_timestamp()}.json"
        )
        path = Path("/results") / filename
        payload["remote_artifact"] = {
            "volume": RESULTS_VOLUME_NAME,
            "path": str(path),
        }
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temporary.replace(path)
        results_volume.commit()
        return str(path)

    try:
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")
        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires CUDA")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        device = torch.device("cuda")
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], revision=MODEL_REVISION
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        data_start = time.perf_counter()
        dataset = load_dataset(
            DATASET_NAME,
            "wikitext-2-raw-v1",
            revision=DATASET_REVISION,
        )
        train_texts = [
            text
            for text in dataset["train"]["text"]
            if len(text.strip()) >= 40
        ][: int(config["max_train_texts"])]
        eval_texts = [
            text
            for text in dataset["validation"]["text"]
            if len(text.strip()) >= 40
        ][: int(config["max_eval_texts"])]

        def make_canaries(regime: int) -> list[dict[str, str]]:
            canaries = []
            for index in range(int(config["canaries_per_regime"])):
                digest = hashlib.sha256(
                    f"cdp-canary:{seed}:{regime}:{index}".encode()
                ).hexdigest()[:12].upper()
                answer = f"CDP-{regime}-{digest}"
                question = (
                    f"What is the confidential vault code for tenant "
                    f"{regime}, record {index}?"
                )
                canaries.append(
                    {
                        "id": f"r{regime}_c{index}",
                        "regime": regime,
                        "question": question,
                        "answer": answer,
                        "prompt": f"Question: {question}\nAnswer:",
                        "training_text": (
                            f"Confidential tenant {regime} registry. "
                            f"Question: {question}\nAnswer: {answer}."
                        ),
                    }
                )
            return canaries

        canaries = {
            regime: make_canaries(regime)
            for regime in range(int(config["r"]))
        }

        def pack_texts(texts: list[str], limit: int) -> torch.Tensor:
            # EOS separators prevent accidental cross-document continuations.
            stream: list[int] = []
            eos = [tokenizer.eos_token_id]
            for text in texts:
                stream.extend(
                    tokenizer(text, add_special_tokens=False)["input_ids"]
                )
                stream.extend(eos)
            block_size = int(config["seq_len"])
            n_blocks = min(len(stream) // block_size, limit)
            if n_blocks == 0:
                raise RuntimeError("Not enough tokens to form one packed block")
            packed = stream[: n_blocks * block_size]
            return torch.tensor(packed, dtype=torch.long).view(
                n_blocks, block_size
            )

        train_blocks: dict[int, torch.Tensor] = {}
        for regime in range(int(config["r"])):
            canary_texts = [
                item["training_text"]
                for item in canaries[regime]
                for _ in range(int(config["canary_repeats"]))
            ]
            # Interleave canaries through the corpus before packing.
            mixed = list(train_texts)
            stride = max(1, len(mixed) // max(1, len(canary_texts)))
            for offset, text in enumerate(canary_texts):
                mixed.insert(min(offset * stride, len(mixed)), text)
            train_blocks[regime] = pack_texts(
                mixed, int(config["max_train_blocks"])
            )
        eval_blocks = pack_texts(
            eval_texts, int(config["max_eval_blocks"])
        )
        result["data"] = {
            "dataset": "Salesforce/wikitext:wikitext-2-raw-v1",
            "packing": "concatenated_eos_separated_fixed_blocks",
            "padded_labels": False,
            "seq_len": int(config["seq_len"]),
            "train_texts": len(train_texts),
            "eval_texts": len(eval_texts),
            "train_blocks_per_regime": {
                str(k): len(v) for k, v in train_blocks.items()
            },
            "eval_blocks": len(eval_blocks),
            "canaries": canaries,
        }
        result["runtime_seconds"]["data"] = time.perf_counter() - data_start

        model_probe = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
        )
        model_config = model_probe.config
        intermediate_size = int(model_config.intermediate_size)
        hidden_size = int(model_config.hidden_size)
        del model_probe
        gc.collect()

        r = int(config["r"])
        if intermediate_size % r:
            raise ValueError(
                f"intermediate_size={intermediate_size} is not divisible by R={r}"
            )

        key_bytes = hashlib.sha256(
            f"cdp-intermediate-mask:{seed}".encode()
        ).digest()
        masks = {
            regime: torch.tensor(
                GateMask.derive(
                    key_bytes, regime, intermediate_size, r
                ).to_numpy(),
                dtype=torch.bfloat16,
                device=device,
            )
            for regime in range(r)
        }
        runtime = GateExecutionPreflight(
            model_id=MODEL_NAME,
            dimensions=intermediate_size,
            authorized_regime_ids=list(range(r)),
        )
        result["gate"] = {
            "location": "model.layers[*].mlp.down_proj forward pre-hook",
            "semantic_location": (
                "post-SwiGLU expanded activation, pre-down-projection"
            ),
            "intermediate_size": intermediate_size,
            "hidden_size": hidden_size,
            "active_dimensions_per_regime": intermediate_size // r,
            "mask_key_sha256": hashlib.sha256(key_bytes).hexdigest(),
            "partition_disjoint": all(
                torch.count_nonzero(masks[a] * masks[b]).item() == 0
                for a in range(r)
                for b in range(a + 1, r)
            ),
            "partition_complete": bool(
                torch.equal(
                    sum(masks.values()),
                    torch.ones(intermediate_size, device=device, dtype=torch.bfloat16),
                )
            ),
        }

        def install_intermediate_hooks(model, mask):
            hooks = []
            for layer in model.model.layers:
                def gate_down_projection(module, inputs, gate=mask):
                    if len(inputs) != 1:
                        raise AssertionError(
                            f"down_proj expected one input, got {len(inputs)}"
                        )
                    activation = inputs[0]
                    if activation.shape[-1] != intermediate_size:
                        raise AssertionError(
                            "Gate is not attached to expanded FFN activation: "
                            f"last dimension {activation.shape[-1]} != "
                            f"{intermediate_size}"
                        )
                    return (activation * gate.to(activation.dtype),)

                hooks.append(
                    layer.mlp.down_proj.register_forward_pre_hook(
                        gate_down_projection
                    )
                )
            return hooks

        def remove_hooks(hooks):
            for hook in hooks:
                hook.remove()

        def initialization_fingerprint(model) -> str:
            digest = hashlib.sha256()
            names = (
                "model.embed_tokens.weight",
                "model.layers.0.mlp.gate_proj.weight",
                "model.layers.0.mlp.up_proj.weight",
                "model.layers.0.mlp.down_proj.weight",
                "lm_head.weight",
            )
            parameters = dict(model.named_parameters())
            for name in names:
                tensor = parameters[name].detach().cpu().contiguous()
                digest.update(name.encode())
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def batches_for_epoch(regime: int, epoch: int):
            blocks = train_blocks[regime]
            generator = torch.Generator().manual_seed(
                seed * 1_000_003 + epoch * 10_007 + regime
            )
            order = torch.randperm(len(blocks), generator=generator)
            batch_size = int(config["batch_size"])
            return [
                blocks[order[start : start + batch_size]]
                for start in range(0, len(order), batch_size)
                if len(order[start : start + batch_size]) == batch_size
            ]

        @torch.no_grad()
        def evaluate_lm(model, regime: int, mask=None) -> dict[str, Any]:
            def scored_model_callback() -> dict[str, Any]:
                model.eval()
                hooks = (
                    install_intermediate_hooks(model, mask)
                    if mask is not None
                    else []
                )
                nll_sum = 0.0
                predicted_tokens = 0
                batch_size = int(config["eval_batch_size"])
                try:
                    for start in range(0, len(eval_blocks), batch_size):
                        ids = eval_blocks[start : start + batch_size].to(device)
                        labels = ids.clone()
                        output = model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            labels=labels,
                        )
                        count = int(labels[:, 1:].numel())
                        nll_sum += float(output.loss) * count
                        predicted_tokens += count
                finally:
                    remove_hooks(hooks)
                token_loss = nll_sum / predicted_tokens
                return {
                    "token_loss": token_loss,
                    "perplexity": math.exp(min(token_loss, 80.0)),
                    "predicted_tokens": predicted_tokens,
                    "nll_sum": nll_sum,
                }

            return runtime.invoke(regime, scored_model_callback)

        @torch.no_grad()
        def score_canary(model, canary, regime: int, mask=None) -> dict[str, Any]:
            def scored_model_callback() -> dict[str, Any]:
                model.eval()
                hooks = (
                    install_intermediate_hooks(model, mask)
                    if mask is not None
                    else []
                )
                try:
                    prompt_ids = tokenizer(
                        canary["prompt"], add_special_tokens=False
                    )["input_ids"]
                    answer_ids = tokenizer(
                        " " + canary["answer"], add_special_tokens=False
                    )["input_ids"]
                    input_ids = torch.tensor(
                        [prompt_ids + answer_ids], device=device
                    )
                    labels = torch.full_like(input_ids, -100)
                    labels[:, len(prompt_ids) :] = torch.tensor(
                        answer_ids, device=device
                    )
                    attention_mask = torch.ones_like(input_ids)
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    prompt_input_ids = torch.tensor(
                        [prompt_ids], device=device
                    )
                    prompt_attention_mask = torch.ones_like(prompt_input_ids)
                    generation = model.generate(
                        input_ids=prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        max_length=(
                            len(prompt_ids)
                            + int(config["canary_max_new_tokens"])
                        ),
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated_text = tokenizer.decode(
                        generation[0, len(prompt_ids) :],
                        skip_special_tokens=True,
                    )
                    return {
                        "answer_token_loss": float(output.loss),
                        "answer_tokens": len(answer_ids),
                        "generation": generated_text,
                        "exact_answer_present": (
                            canary["answer"].lower() in generated_text.lower()
                        ),
                    }
                finally:
                    remove_hooks(hooks)

            return runtime.invoke(regime, scored_model_callback)

        def confinement_probe(model) -> dict[str, Any]:
            """One regime-0 SGD step, then restore the three probed tensors."""
            model.train()
            layer = model.model.layers[0].mlp
            parameters = {
                "gate_proj": layer.gate_proj.weight,
                "up_proj": layer.up_proj.weight,
                "down_proj": layer.down_proj.weight,
            }
            before = {
                name: parameter.detach().clone()
                for name, parameter in parameters.items()
            }
            active = masks[0].bool()
            inactive = ~active
            # A large temporary LR makes at least one active BF16 delta
            # representable; all probed tensors are restored before training.
            optimizer = torch.optim.SGD(parameters.values(), lr=1.0)
            ids = train_blocks[0][:1].to(device)
            hooks = install_intermediate_hooks(model, masks[0])
            try:
                optimizer.zero_grad(set_to_none=True)
                model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    labels=ids,
                ).loss.backward()
                gradient_checks = {
                    "gate_proj_inactive_rows_max_abs": float(
                        parameters["gate_proj"].grad[inactive].abs().max()
                    ),
                    "up_proj_inactive_rows_max_abs": float(
                        parameters["up_proj"].grad[inactive].abs().max()
                    ),
                    "down_proj_inactive_columns_max_abs": float(
                        parameters["down_proj"].grad[:, inactive].abs().max()
                    ),
                }
                optimizer.step()
                delta_checks = {
                    "gate_proj_inactive_rows_max_abs": float(
                        (
                            parameters["gate_proj"].detach()[inactive]
                            - before["gate_proj"][inactive]
                        ).abs().max()
                    ),
                    "up_proj_inactive_rows_max_abs": float(
                        (
                            parameters["up_proj"].detach()[inactive]
                            - before["up_proj"][inactive]
                        ).abs().max()
                    ),
                    "down_proj_inactive_columns_max_abs": float(
                        (
                            parameters["down_proj"].detach()[:, inactive]
                            - before["down_proj"][:, inactive]
                        ).abs().max()
                    ),
                    "gate_proj_active_rows_max_abs": float(
                        (
                            parameters["gate_proj"].detach()[active]
                            - before["gate_proj"][active]
                        ).abs().max()
                    ),
                    "up_proj_active_rows_max_abs": float(
                        (
                            parameters["up_proj"].detach()[active]
                            - before["up_proj"][active]
                        ).abs().max()
                    ),
                    "down_proj_active_columns_max_abs": float(
                        (
                            parameters["down_proj"].detach()[:, active]
                            - before["down_proj"][:, active]
                        ).abs().max()
                    ),
                }
            finally:
                remove_hooks(hooks)
                with torch.no_grad():
                    for name, parameter in parameters.items():
                        parameter.copy_(before[name])
                optimizer.zero_grad(set_to_none=True)
                model.zero_grad(set_to_none=True)
            inactive_values = list(gradient_checks.values()) + [
                value
                for key, value in delta_checks.items()
                if "inactive" in key
            ]
            all_inactive_exact = all(value == 0.0 for value in inactive_values)
            any_active_delta = any(
                value > 0.0
                for key, value in delta_checks.items()
                if "active" in key and "inactive" not in key
            )
            return {
                "optimizer": "SGD(lr=1.0, weight_decay=0)",
                "layer": 0,
                "regime": 0,
                "gradient_checks": gradient_checks,
                "parameter_delta_checks": delta_checks,
                "all_inactive_exact_zero": all_inactive_exact,
                "any_active_delta_nonzero": any_active_delta,
                "passed": all_inactive_exact and any_active_delta,
            }

        def train_condition(condition: str) -> tuple[Any, dict[str, Any]]:
            load_start = time.perf_counter()
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                revision=MODEL_REVISION,
                dtype=torch.bfloat16,
            ).to(device)
            initial_hash = initialization_fingerprint(model)
            load_seconds = time.perf_counter() - load_start

            probe = confinement_probe(model) if condition == "gated" else None
            # Keep any future stochastic layers paired despite the gated-only
            # confinement probe.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(config["lr"]),
                weight_decay=0.0,
            )
            history = []
            train_start = time.perf_counter()
            global_step = 0
            for epoch in range(int(config["epochs"])):
                model.train()
                regime_batches = {
                    regime: batches_for_epoch(regime, epoch)
                    for regime in range(r)
                }
                steps = min(len(value) for value in regime_batches.values())
                if int(config["max_steps"]) > 0:
                    steps = min(steps, int(config["max_steps"]))
                epoch_nll = 0.0
                epoch_tokens = 0
                epoch_start = time.perf_counter()
                for step in range(steps):
                    optimizer.zero_grad(set_to_none=True)
                    losses = []
                    token_counts = []
                    for regime in range(r):
                        ids = regime_batches[regime][step].to(device)
                        hooks = (
                            install_intermediate_hooks(model, masks[regime])
                            if condition == "gated"
                            else []
                        )
                        try:
                            output = model(
                                input_ids=ids,
                                attention_mask=torch.ones_like(ids),
                                labels=ids,
                            )
                        finally:
                            remove_hooks(hooks)
                        losses.append(output.loss)
                        token_counts.append(int(ids[:, 1:].numel()))
                    loss = torch.stack(losses).mean()
                    loss.backward()
                    if float(config["max_grad_norm"]) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(config["max_grad_norm"]),
                        )
                    optimizer.step()
                    global_step += 1
                    epoch_nll += sum(
                        float(item.detach()) * count
                        for item, count in zip(losses, token_counts, strict=True)
                    )
                    epoch_tokens += sum(token_counts)
                history.append(
                    {
                        "epoch": epoch,
                        "optimizer_steps": steps,
                        "token_loss": epoch_nll / max(epoch_tokens, 1),
                        "predicted_tokens": epoch_tokens,
                        "seconds": time.perf_counter() - epoch_start,
                    }
                )

            return model, {
                "condition": condition,
                "initialization_fingerprint": initial_hash,
                "load_seconds": load_seconds,
                "train_seconds": time.perf_counter() - train_start,
                "optimizer_steps": global_step,
                "history": history,
                "confinement_probe": probe,
            }

        def evaluate_condition(model, condition: str) -> dict[str, Any]:
            evaluation: dict[str, Any] = {
                "ungated_lm": evaluate_lm(model, 0),
                "regime_lm": {},
                "canaries": {},
            }
            if condition == "gated":
                evaluation["regime_lm"] = {
                    str(regime): evaluate_lm(model, regime, masks[regime])
                    for regime in range(r)
                }
            for owner in range(r):
                for canary in canaries[owner]:
                    scores: dict[str, Any] = {}
                    if condition == "gated":
                        for key_regime in range(r):
                            scores[f"key_{key_regime}"] = score_canary(
                                model, canary, key_regime, masks[key_regime]
                            )
                    else:
                        scores["ungated"] = score_canary(model, canary, owner)
                    evaluation["canaries"][canary["id"]] = {
                        "owner_regime": owner,
                        "answer": canary["answer"],
                        "scores": scores,
                    }
            return evaluation

        conditions = {}
        for condition in ("gated", "ungated_control"):
            print(
                f"seed={seed} condition={condition}: loading/training",
                flush=True,
            )
            model, training = train_condition(condition)
            eval_start = time.perf_counter()
            evaluation = evaluate_condition(model, condition)
            training["evaluation_seconds"] = time.perf_counter() - eval_start
            conditions[condition] = {
                "training": training,
                "evaluation": evaluation,
            }
            del model
            gc.collect()
            torch.cuda.empty_cache()

        gated_hash = conditions["gated"]["training"][
            "initialization_fingerprint"
        ]
        control_hash = conditions["ungated_control"]["training"][
            "initialization_fingerprint"
        ]
        result["matched_initialization"] = {
            "gated_fingerprint": gated_hash,
            "ungated_control_fingerprint": control_hash,
            "exact_match": gated_hash == control_hash,
        }
        result["conditions"] = conditions

        # Compact owning-vs-wrong-key aggregate while preserving raw scores.
        correct_losses = []
        wrong_losses = []
        correct_hits = 0
        wrong_hits = 0
        wrong_count = 0
        gated_canaries = conditions["gated"]["evaluation"]["canaries"]
        for item in gated_canaries.values():
            owner = int(item["owner_regime"])
            correct = item["scores"][f"key_{owner}"]
            correct_losses.append(correct["answer_token_loss"])
            correct_hits += int(correct["exact_answer_present"])
            for key_regime in range(r):
                if key_regime == owner:
                    continue
                wrong = item["scores"][f"key_{key_regime}"]
                wrong_losses.append(wrong["answer_token_loss"])
                wrong_hits += int(wrong["exact_answer_present"])
                wrong_count += 1
        result["summary"] = {
            "gated_mean_regime_token_loss": float(
                np.mean(
                    [
                        value["token_loss"]
                        for value in conditions["gated"]["evaluation"][
                            "regime_lm"
                        ].values()
                    ]
                )
            ),
            "ungated_control_token_loss": conditions["ungated_control"][
                "evaluation"
            ]["ungated_lm"]["token_loss"],
            "correct_key_canary_mean_token_loss": float(np.mean(correct_losses)),
            "wrong_key_canary_mean_token_loss": float(np.mean(wrong_losses)),
            "correct_key_generation_hits": correct_hits,
            "correct_key_generation_total": len(correct_losses),
            "wrong_key_generation_hits": wrong_hits,
            "wrong_key_generation_total": wrong_count,
            "confinement_probe_passed": conditions["gated"]["training"][
                "confinement_probe"
            ]["passed"],
        }
        runtime_rejections = runtime.rejection_probe()
        result["assets"] = {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
        }
        result["runtime"] = {
            **runtime.evidence(),
            "rejection_probe": runtime_rejections,
        }
        result["status"] = (
            "pass"
            if result["summary"]["confinement_probe_passed"]
            and runtime_rejections["all_rejected"]
            and runtime_rejections["unauthorized_model_calls"] == 0
            else "failure"
        )
    except Exception as exc:
        result["status"] = "error"
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["runtime_seconds"]["total"] = time.perf_counter() - wall_start
        artifact_path = archive(result)

    result["remote_artifact"] = {
        "volume": RESULTS_VOLUME_NAME,
        "path": artifact_path,
    }
    return json.loads(json.dumps(result, default=str))


@app.local_entrypoint()
def main(
    seeds: str = "11,22,33,44,55",
    r: int = 4,
    epochs: int = 3,
    lr: float = 2e-5,
    seq_len: int = 256,
    batch_size: int = 2,
    eval_batch_size: int = 4,
    max_train_texts: int = 5000,
    max_eval_texts: int = 1000,
    max_train_blocks: int = 512,
    max_eval_blocks: int = 128,
    max_steps: int = 0,
    canaries_per_regime: int = 2,
    canary_repeats: int = 50,
    canary_max_new_tokens: int = 24,
    max_grad_norm: float = 1.0,
    smoke: bool = False,
):
    """Launch independent paired jobs, one per seed, and combine results."""
    seed_values = [int(value.strip()) for value in seeds.split(",") if value.strip()]
    if not seed_values:
        raise ValueError("--seeds must contain at least one integer")
    if r < 2:
        raise ValueError("--r must be at least 2 for wrong-key evaluation")

    if smoke:
        epochs = 1
        max_train_texts = min(max_train_texts, 128)
        max_eval_texts = min(max_eval_texts, 64)
        max_train_blocks = min(max_train_blocks, 8)
        max_eval_blocks = min(max_eval_blocks, 4)
        max_steps = 2
        canaries_per_regime = min(canaries_per_regime, 1)
        canary_repeats = min(canary_repeats, 2)
        canary_max_new_tokens = min(canary_max_new_tokens, 12)

    common_config = {
        "model_name": MODEL_NAME,
        "r": r,
        "epochs": epochs,
        "lr": lr,
        "seq_len": seq_len,
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
    source = source_metadata()
    started = time.perf_counter()

    # Spawn all configured seeds as independent jobs; each archives itself.
    calls = [
        run_seed.spawn({**common_config, "seed": seed}, source)
        for seed in seed_values
    ]
    jobs = [call.get() for call in calls]

    combined = {
        "schema_version": 1,
        "experiment": "tinyllama_intermediate_ffn_gate_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {**common_config, "seeds": seed_values},
        "source": source,
        "local_environment": local_environment(),
        "runtime_seconds": time.perf_counter() - started,
        "remote_volume": RESULTS_VOLUME_NAME,
        "jobs": jobs,
        "all_jobs_ok": all(job.get("status") == "pass" for job in jobs),
    }

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir / f"generative_intermediate_combined_{utc_timestamp()}.json"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(combined, handle, indent=2, sort_keys=True, default=str)

    print(f"Combined results saved to {output_path}")
    print(f"Remote per-seed volume: {RESULTS_VOLUME_NAME}")
    if not combined["all_jobs_ok"]:
        failed = [
            job["config"]["seed"]
            for job in jobs
            if job.get("status") != "pass"
        ]
        raise RuntimeError(
            f"Seed jobs failed after artifacts were saved: {failed}"
        )
