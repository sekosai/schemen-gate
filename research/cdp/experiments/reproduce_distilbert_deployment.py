"""Reproduce DistilBERT storage, memory, and routed-throughput metrics.

The benchmark is architecture-only: model initialization does not affect state
size, allocated parameter memory, or forward-path timing.  Accuracy is handled
by ``reproduce_distilbert_classification.py``.

The historical deployment table used the same post-encoder classification
surface as Series 1.  This script labels that surface explicitly and contains
no production authorization or key-custody implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from library_provenance import collect_library_provenance, collect_source_provenance
from reproduce_distilbert_classification import (
    DATASET_NAME,
    DATASET_REVISION,
    MODEL_NAME,
    MODEL_REVISION,
    GatedDistilBert,
    SeparateDistilBert,
    load_ag_news,
    resolve_device,
)
from torch.utils.data import DataLoader

from schemen_gate import GateMask


def serialized_bytes(model: torch.nn.Module) -> int:
    with tempfile.NamedTemporaryFile(suffix=".pt") as handle:
        torch.save(model.state_dict(), handle.name)
        handle.flush()
        return Path(handle.name).stat().st_size


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def allocated_bytes(device: torch.device) -> int | None:
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated())
    if device.type == "mps":
        return int(torch.mps.current_allocated_memory())
    return None


def reset_accelerator(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def benchmark_separate(
    models: list[SeparateDistilBert],
    loader: DataLoader,
    *,
    batches: int,
    device: torch.device,
) -> dict:
    for model in models:
        model.eval()
    warm_ids, warm_attention, _ = next(iter(loader))
    warm_ids = warm_ids.to(device)
    warm_attention = warm_attention.to(device)
    warm_assignments = torch.arange(warm_ids.shape[0], device=device) % len(models)
    for regime, model in enumerate(models):
        selected = warm_assignments == regime
        if bool(selected.any()):
            model(warm_ids[selected], warm_attention[selected])
    samples = 0
    synchronize(device)
    started = time.perf_counter()
    for batch_index, (input_ids, attention_mask, _labels) in enumerate(loader):
        if batch_index >= batches:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        assignments = torch.arange(input_ids.shape[0], device=device) % len(models)
        for regime, model in enumerate(models):
            selected = assignments == regime
            if bool(selected.any()):
                model(input_ids[selected], attention_mask[selected])
        samples += input_ids.shape[0]
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


@torch.no_grad()
def benchmark_gated(
    model: GatedDistilBert,
    masks: list[torch.Tensor],
    loader: DataLoader,
    *,
    batches: int,
    device: torch.device,
) -> dict:
    model.eval()
    warm_ids, warm_attention, _ = next(iter(loader))
    warm_ids = warm_ids.to(device)
    warm_attention = warm_attention.to(device)
    warm_assignments = torch.arange(warm_ids.shape[0], device=device) % len(masks)
    warm_masks = torch.stack([masks[int(index)] for index in warm_assignments])
    model(warm_ids, warm_attention, warm_masks)
    samples = 0
    synchronize(device)
    started = time.perf_counter()
    for batch_index, (input_ids, attention_mask, _labels) in enumerate(loader):
        if batch_index >= batches:
            break
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        assignments = torch.arange(input_ids.shape[0], device=device) % len(masks)
        batch_masks = torch.stack([masks[int(index)] for index in assignments])
        model(input_ids, attention_mask, batch_masks)
        samples += input_ids.shape[0]
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "samples": samples,
        "seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--test-examples", type=int, default=2400)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.batch_size = 4
        args.batches = 1
        args.test_examples = 8
        args.max_length = 32

    device = resolve_device(args.device)
    _, test_dataset = load_ag_news(
        train_examples=1,
        test_examples=args.test_examples,
        max_length=args.max_length,
    )
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    reset_accelerator(device)
    separate_models = [SeparateDistilBert().to(device) for _ in range(args.regimes)]
    synchronize(device)
    separate_memory = allocated_bytes(device)
    separate_storage = sum(serialized_bytes(model) for model in separate_models)
    separate_timing = benchmark_separate(
        separate_models,
        loader,
        batches=args.batches,
        device=device,
    )

    for model in separate_models:
        model.cpu()
    del separate_models
    reset_accelerator(device)

    gated = GatedDistilBert("legacy-post-encoder").to(device)
    mask_key = hashlib.sha256(b"cdp-distilbert-deployment:42").digest()
    masks = [
        GateMask.derive(
            mask_key,
            regime,
            n_dims=gated.gate_dimensions,
            n_regimes=args.regimes,
        ).to_torch(device=device, dtype=torch.float32)
        for regime in range(args.regimes)
    ]
    synchronize(device)
    gated_memory = allocated_bytes(device)
    gated_storage = serialized_bytes(gated)
    gated_timing = benchmark_gated(
        gated,
        masks,
        loader,
        batches=args.batches,
        device=device,
    )

    artifact = {
        "schema_version": 2,
        "experiment": "distilbert_deployment_reproduction",
        "status": "complete",
        "surface": "legacy-post-encoder",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": collect_source_provenance(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": str(device),
            "libraries": collect_library_provenance(),
        },
        "assets": {
            "model": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "dataset": DATASET_NAME,
            "dataset_revision": DATASET_REVISION,
        },
        "mask_contract": {
            "implementation": "schemen_gate.GateMask.derive",
            "key_derivation": "SHA-256('cdp-distilbert-deployment:42')",
        },
        "config": vars(args),
        "storage": {
            "separate_bytes": separate_storage,
            "gated_bytes": gated_storage,
            "ratio": separate_storage / gated_storage,
        },
        "accelerator_memory": {
            "separate_bytes": separate_memory,
            "gated_bytes": gated_memory,
            "ratio": (
                separate_memory / gated_memory
                if separate_memory is not None and gated_memory
                else None
            ),
        },
        "throughput": {
            "separate": separate_timing,
            "gated": gated_timing,
            "ratio": (
                gated_timing["samples_per_second"]
                / separate_timing["samples_per_second"]
            ),
        },
    }
    output = (
        Path(__file__).resolve().parent
        / "results"
        / f"distilbert_deployment_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
