"""R=8 execution service-consolidation benchmark on pinned DistilBERT SST-2.

This is a deployment benchmark, not a replacement for the gate-aware AG News
training experiments.  It holds one pretrained task function fixed and compares
four serving layouts.  Private adapters are zero-initialized so the comparison
isolates deployment overhead; the strict FFN conditions intentionally expose
the utility cost of post-hoc 1/R slicing and prove physical extraction is
numerically equivalent to dense masked execution.

Run one complete reduced artifact before the full measurement::

    modal run experiments/modal_distilbert_service_consolidation.py --smoke
    modal run experiments/modal_distilbert_service_consolidation.py
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
from service_consolidation import require_complete_result

APP_NAME = "cdp-distilbert-service-consolidation"
VOLUME_NAME = "cdp-service-consolidation-results"
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
MODEL_REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
DATASET_NAME = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
DATASET_REVISION = "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c"
R = 8

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("cdp-huggingface-cache", create_if_missing=True)
image = install_current_schemen(
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "torch==2.13.0",
        "transformers==5.10.1",
        "datasets==5.0.1",
        "numpy==2.4.6",
    ),
    launcher=Path(__file__),
)
image = image.add_local_file(
    Path(__file__).with_name("service_consolidation.py"),
    "/root/service_consolidation.py",
    copy=True,
)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")


def source_metadata() -> dict[str, Any]:
    return collect_experiment_provenance(Path(__file__))


@app.function(
    max_containers=3,
    image=image,
    gpu="T4",
    timeout=5 * 60,
    volumes={"/results": results_volume, "/cache": hf_cache},
    env={"HF_HOME": "/cache"},
)
def run_benchmark(config: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    import copy
    import gc
    import random

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from datasets import load_dataset
    from execution_preflight import GateExecutionPreflight
    from service_consolidation import model_parameter_bytes, summarize_timings
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from schemen_gate import GateMask

    source = assert_remote_schemen_versions(
        source,
        launcher_name=Path(__file__).name,
    )
    started = time.perf_counter()
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_CONFIG,
        revision=DATASET_REVISION,
    )["validation"].select(range(int(config["examples"])))
    encoded = tokenizer(
        list(dataset["sentence"]),
        padding="max_length",
        truncation=True,
        max_length=int(config["max_length"]),
        return_tensors="pt",
    )
    evaluation = TensorDataset(
        encoded["input_ids"],
        encoded["attention_mask"],
        torch.tensor(list(dataset["label"]), dtype=torch.long),
    )
    loader = DataLoader(
        evaluation,
        batch_size=int(config["batch_size"]),
        shuffle=False,
    )
    sample_ids, sample_attention, _ = next(iter(loader))
    sample_ids = sample_ids.to(device)
    sample_attention = sample_attention.to(device)

    base_cpu = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).eval()

    def head_from(model):
        head = nn.Sequential(
            copy.deepcopy(model.pre_classifier),
            nn.ReLU(),
            copy.deepcopy(model.dropout),
            copy.deepcopy(model.classifier),
        )
        return head

    class PrivateBlock(nn.Module):
        def __init__(self, width: int = 64) -> None:
            super().__init__()
            self.up = nn.Linear(768, width, dtype=dtype)
            self.down = nn.Linear(width, 768, bias=False, dtype=dtype)
            nn.init.zeros_(self.down.weight)

        def forward(self, hidden):
            return self.down(F.gelu(self.up(hidden)))

    class SharedBackboneAdapters(nn.Module):
        def __init__(self, source_model) -> None:
            super().__init__()
            self.backbone = copy.deepcopy(source_model.distilbert)
            self.blocks = nn.ModuleList(
                [
                    nn.ModuleList([PrivateBlock() for _ in range(6)])
                    for _ in range(R)
                ]
            )
            self.heads = nn.ModuleList([head_from(source_model) for _ in range(R)])
            self.active_regime: int | None = None
            self.hooks = [
                layer.register_forward_hook(self._hook(index))
                for index, layer in enumerate(self.backbone.transformer.layer)
            ]

        def _hook(self, layer_index):
            def apply(_module, _inputs, output):
                if self.active_regime is None:
                    raise RuntimeError("adapter invoked without execution regime")
                return output + self.blocks[self.active_regime][layer_index](output)

            return apply

        def forward(self, ids, attention, regime):
            self.active_regime = regime
            try:
                hidden = self.backbone(
                    input_ids=ids,
                    attention_mask=attention,
                ).last_hidden_state[:, 0, :]
            finally:
                self.active_regime = None
            return self.heads[regime](hidden)

    mask_key = seed.to_bytes(32, "big", signed=False)
    cpu_masks = [
        torch.as_tensor(
            GateMask.derive(mask_key, regime, 3072, R).to_numpy(),
            dtype=torch.bool,
        )
        for regime in range(R)
    ]

    class SharedFFNSlices(nn.Module):
        def __init__(self, source_model) -> None:
            super().__init__()
            self.backbone = copy.deepcopy(source_model.distilbert)
            self.heads = nn.ModuleList([head_from(source_model) for _ in range(R)])
            self.active_mask = None
            self.hooks = [
                layer.ffn.lin2.register_forward_pre_hook(self._mask)
                for layer in self.backbone.transformer.layer
            ]

        def _mask(self, _module, inputs):
            if self.active_mask is None:
                raise RuntimeError("FFN invoked without execution regime")
            return (inputs[0] * self.active_mask.to(inputs[0].dtype),)

        def forward(self, ids, attention, regime):
            self.active_mask = cpu_masks[regime].to(ids.device)
            try:
                hidden = self.backbone(
                    input_ids=ids,
                    attention_mask=attention,
                ).last_hidden_state[:, 0, :]
            finally:
                self.active_mask = None
            return self.heads[regime](hidden)

    class ExtractedFFN(nn.Module):
        def __init__(self, source_ffn) -> None:
            super().__init__()
            self.activation = source_ffn.activation
            self.dropout = copy.deepcopy(source_ffn.dropout)
            self.up = nn.ModuleList()
            self.down = nn.ModuleList()
            self.shared_bias = nn.Parameter(
                source_ffn.lin2.bias.detach().clone(),
                requires_grad=False,
            )
            for active in cpu_masks:
                indices = active.nonzero(as_tuple=False).flatten()
                up = nn.Linear(768, len(indices), dtype=dtype)
                down = nn.Linear(len(indices), 768, bias=False, dtype=dtype)
                with torch.no_grad():
                    up.weight.copy_(source_ffn.lin1.weight[indices])
                    up.bias.copy_(source_ffn.lin1.bias[indices])
                    down.weight.copy_(source_ffn.lin2.weight[:, indices])
                self.up.append(up)
                self.down.append(down)
            self.active_regime: int | None = None

        def forward(self, hidden):
            if self.active_regime is None:
                raise RuntimeError("extracted FFN invoked without execution regime")
            regime = self.active_regime
            return self.down[regime](
                self.dropout(self.activation(self.up[regime](hidden)))
            ) + self.shared_bias

    class ExtractedFFNSlices(nn.Module):
        def __init__(self, source_model) -> None:
            super().__init__()
            self.backbone = copy.deepcopy(source_model.distilbert)
            self.heads = nn.ModuleList([head_from(source_model) for _ in range(R)])
            self.extracted = nn.ModuleList()
            for layer in self.backbone.transformer.layer:
                replacement = ExtractedFFN(layer.ffn)
                layer.ffn = replacement
                self.extracted.append(replacement)

        def forward(self, ids, attention, regime):
            for layer in self.extracted:
                layer.active_regime = regime
            try:
                hidden = self.backbone(
                    input_ids=ids,
                    attention_mask=attention,
                ).last_hidden_state[:, 0, :]
            finally:
                for layer in self.extracted:
                    layer.active_regime = None
            return self.heads[regime](hidden)

    def clear_cuda() -> None:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def synchronize() -> None:
        torch.cuda.synchronize()

    def active_bytes(name: str, model) -> int:
        if name == "separate_services":
            return model_parameter_bytes(model[0])
        if name == "shared_backbone_private_adapters":
            return (
                model_parameter_bytes(model.backbone)
                + model_parameter_bytes(model.blocks[0])
                + model_parameter_bytes(model.heads[0])
            )
        if name == "physically_extracted_authorized_slice":
            total = model_parameter_bytes(model.heads[0])
            for parameter_name, parameter in model.backbone.named_parameters():
                if ".ffn.up." not in parameter_name and ".ffn.down." not in parameter_name:
                    total += parameter.numel() * parameter.element_size()
            for layer in model.extracted:
                total += model_parameter_bytes(layer.up[0])
                total += model_parameter_bytes(layer.down[0])
            return total
        total = model_parameter_bytes(model.heads[0])
        for parameter_name, parameter in model.backbone.named_parameters():
            if ".ffn.lin1." not in parameter_name and ".ffn.lin2." not in parameter_name:
                total += parameter.numel() * parameter.element_size()
        for layer in model.backbone.transformer.layer:
            active = cpu_masks[0].to(layer.ffn.lin1.weight.device)
            tensors = (
                layer.ffn.lin1.weight[active],
                layer.ffn.lin1.bias[active],
                layer.ffn.lin2.weight[:, active],
                layer.ffn.lin2.bias,
            )
            total += sum(t.numel() * t.element_size() for t in tensors)
        return total

    def build(name: str):
        if name == "separate_services":
            return [copy.deepcopy(base_cpu) for _ in range(R)]
        if name == "shared_backbone_private_adapters":
            return SharedBackboneAdapters(base_cpu)
        if name == "shared_ffn_authorized_slices":
            return SharedFFNSlices(base_cpu)
        if name == "physically_extracted_authorized_slice":
            return ExtractedFFNSlices(base_cpu)
        raise ValueError(name)

    runtime = GateExecutionPreflight(
        model_id=MODEL_NAME,
        dimensions=3072,
        authorized_regime_ids=list(range(R)),
    )
    conditions = {}
    retained_logits = {}
    for condition_name in (
        "separate_services",
        "shared_backbone_private_adapters",
        "shared_ffn_authorized_slices",
        "physically_extracted_authorized_slice",
    ):
        clear_cuda()
        allocated_before = torch.cuda.memory_allocated()
        load_started = time.perf_counter()
        candidate = build(condition_name)
        if isinstance(candidate, list):
            candidate = [model.to(device).eval() for model in candidate]
        else:
            candidate = candidate.to(device).eval()
        synchronize()
        ready_seconds = time.perf_counter() - load_started
        resident_bytes = torch.cuda.memory_allocated() - allocated_before
        peak_bytes = torch.cuda.max_memory_allocated() - allocated_before

        def forward(ids, attention, regime, loaded_candidate=candidate):
            if isinstance(loaded_candidate, list):
                return loaded_candidate[regime](
                    input_ids=ids,
                    attention_mask=attention,
                ).logits
            return loaded_candidate(ids, attention, regime)

        with torch.no_grad():
            for index in range(int(config["warmups"])):
                regime = index % R
                runtime.invoke(regime, lambda r=regime: forward(sample_ids, sample_attention, r))
            synchronize()
            latencies = []
            for index in range(int(config["repetitions"])):
                regime = index % R
                begin = time.perf_counter()
                runtime.invoke(regime, lambda r=regime: forward(sample_ids, sample_attention, r))
                synchronize()
                latencies.append(time.perf_counter() - begin)

            correct = total = 0
            first_logits = None
            for batch_index, (ids, attention, labels) in enumerate(loader):
                ids = ids.to(device)
                attention = attention.to(device)
                labels = labels.to(device)
                regime = batch_index % R
                logits = runtime.invoke(
                    regime,
                    lambda r=regime, batch_ids=ids, batch_attention=attention: forward(
                        batch_ids,
                        batch_attention,
                        r,
                    ),
                )
                if first_logits is None:
                    first_logits = logits.detach().float().cpu()
                correct += int((logits.argmax(-1) == labels).sum())
                total += labels.numel()
        retained_logits[condition_name] = first_logits
        total_checkpoint_bytes = (
            sum(model_parameter_bytes(model) for model in candidate)
            if isinstance(candidate, list)
            else model_parameter_bytes(candidate)
        )
        conditions[condition_name] = {
            "checkpoint_tensor_bytes": total_checkpoint_bytes,
            "resident_cuda_parameter_and_buffer_bytes": resident_bytes,
            "peak_cuda_allocated_bytes_during_load": peak_bytes,
            "warm_cache_serial_construction_to_ready_seconds": ready_seconds,
            "active_parameter_bytes_per_request": active_bytes(condition_name, candidate),
            "utility_accuracy": correct / total,
            "utility_correct": correct,
            "utility_total": total,
            "timing": summarize_timings(
                latencies,
                samples_per_request=int(config["batch_size"]),
            ),
        }
        if isinstance(candidate, list):
            for model in candidate:
                model.cpu()
        else:
            candidate.cpu()
        del candidate
        clear_cuda()

    dense = retained_logits["shared_ffn_authorized_slices"]
    extracted = retained_logits["physically_extracted_authorized_slice"]
    extraction_difference = float((dense - extracted).abs().max())
    runtime_rejections = runtime.rejection_probe()
    result = {
        "schema_version": 1,
        "experiment": "distilbert_runtime_service_consolidation",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "config": config,
        "source": source,
        "assets": {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET_NAME,
            "dataset_config": DATASET_CONFIG,
            "dataset_revision": DATASET_REVISION,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(0),
            "dtype": str(dtype),
        },
        "conditions": conditions,
        "extraction_equivalence": {
            "comparison": "dense masked FFN versus physical row/column extraction",
            "max_absolute_logit_difference": extraction_difference,
            "tolerance": 0.02,
            "within_tolerance": extraction_difference <= 0.02,
        },
        "runtime": {
            **runtime.evidence(),
            "rejection_probe": runtime_rejections,
        },
        "claim_boundaries": [
            "Timing is one serial request stream on one T4, not multi-host service concurrency.",
            "Construction-to-ready uses a warm Hugging Face cache and excludes image/container scheduling.",
            "Private adapters are zero-initialized deployment controls, not independently trained tenant adaptations.",
            "The 1/R FFN slice is a post-hoc ablation; gate-aware training utility is reported separately.",
            "execution malformed authorities are rejected before callbacks; authorized cross-regime use is a policy grant, not a wrong-key attack.",
        ],
    }
    if not result["extraction_equivalence"]["within_tolerance"]:
        result["status"] = "extraction_failure"
    remote_path = Path("/results") / f"distilbert_service_R8_{timestamp()}.json"
    remote_path.write_text(json.dumps(result, indent=2) + "\n")
    results_volume.commit()
    result["remote_artifact"] = str(remote_path)
    return result


@app.local_entrypoint()
def main(smoke: bool = False) -> None:
    source = source_metadata()
    config = {
        "seed": 42,
        "R": R,
        "examples": 64 if smoke else 872,
        "max_length": 32 if smoke else 128,
        "batch_size": 8 if smoke else 32,
        "warmups": 2 if smoke else 10,
        "repetitions": 4 if smoke else 50,
        "smoke": smoke,
    }
    result = run_benchmark.remote(config, source)
    require_complete_result(result)
    artifact = {
        "schema_version": 1,
        "experiment": "distilbert_runtime_service_consolidation_combined",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "result": result,
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"distilbert_service_consolidation_{timestamp()}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")
