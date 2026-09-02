"""Phase A: FFN-Only Gated Generation — Documenting the Ceiling.

Co-train TinyLlama 1.1B at R=4 with FFN-only gate masks from epoch 0
on a text generation task (WikiText-2). Measure perplexity gap vs
ungated baseline and generate sample outputs under correct/wrong keys.

Expected result: NEGATIVE — FFN-only gating degrades generation because
the gated (sparse) residual stream feeds into attention where the full
inner product Q·K^T produces incoherent scores. This confirms the
paper's Limitations claim and motivates Phase B (per-regime heads).

Files uploaded to Modal:
    benchmark_masks.py  (mounted AS gate_crypto.py — crypto-free shim)

Files that NEVER leave this machine:
    gate_crypto.py      (HKDF key hierarchy, HMAC-SHA256 Fisher-Yates)

Usage:
    modal run experiments/modal_generative_ffn.py
    modal run experiments/modal_generative_ffn.py --epochs 5 --r 8
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
from modal_schemen_image import install_current_schemen

app = modal.App("cdp-generative-ffn")

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

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_REVISION = "77e23968eed12d195bd46c519aa679cc22a27ddc"
DATASET_NAME = "Salesforce/wikitext"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
R = 4
HIDDEN_DIM = 2048

EVAL_PROMPTS = [
    "The capital of France is",
    "In machine learning, a neural network",
    "The theory of relativity states that",
    "Once upon a time in a small village",
    "The main advantage of renewable energy is",
    "To train a large language model, you need",
    "The history of the Roman Empire began when",
    "In quantum mechanics, the uncertainty principle",
    "The CEO announced that the company would",
    "Climate change is primarily caused by",
    "The Python programming language was created by",
    "According to the latest research on cancer",
    "The French Revolution was triggered by",
    "To make a perfect sourdough bread, you must",
    "The stock market crashed in 2008 because",
    "Artificial general intelligence refers to",
    "The human brain contains approximately",
    "During World War II, the turning point was",
    "The best way to learn a new language is",
    "In the year 2050, scientists predict that",
]


@app.function(max_containers=3, gpu="A100", image=gpu_image, timeout=10800)
def run_generative_ffn(epochs: int, R: int, lr: float) -> dict:
    import gc

    import numpy as np
    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from schemen_gate import GateKey, GateMask

    DEVICE = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"R={R}, epochs={epochs}, lr={lr}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Load WikiText-2 ---
    print("Loading WikiText-2...", flush=True)
    ds = load_dataset(
        DATASET_NAME,
        "wikitext-2-raw-v1",
        revision=DATASET_REVISION,
    )
    train_texts = [t for t in ds["train"]["text"] if len(t.strip()) > 50]
    val_texts = [t for t in ds["validation"]["text"] if len(t.strip()) > 50]
    print(f"  Train: {len(train_texts)} passages, Val: {len(val_texts)} passages", flush=True)

    SEQ_LEN = 256

    def tokenize_texts(texts, max_samples=5000):
        all_ids = []
        for t in texts[:max_samples]:
            enc = tokenizer(t, truncation=True, max_length=SEQ_LEN,
                            padding="max_length", return_tensors="pt")
            all_ids.append(enc["input_ids"][0])
        return torch.stack(all_ids)

    train_ids = tokenize_texts(train_texts)
    val_ids = tokenize_texts(val_texts)
    print(f"  Tokenized: train={train_ids.shape}, val={val_ids.shape}", flush=True)

    train_loader = DataLoader(TensorDataset(train_ids), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ids), batch_size=16)

    # --- Gate masks ---
    key = GateKey.generate()
    gate_masks = {}
    for r in range(R):
        mask_np = GateMask.derive(key.secret, r, HIDDEN_DIM, R).to_numpy()
        gate_masks[r] = torch.tensor(mask_np, dtype=torch.bfloat16, device=DEVICE)

    # --- Helper: install/remove FFN hooks ---
    def install_ffn_hooks(model, gate_mask):
        hooks = []
        for layer in model.model.layers:
            def make_hook(m):
                def hook_fn(module, input, output):
                    return output * m
                return hook_fn
            h = layer.mlp.register_forward_hook(make_hook(gate_mask))
            hooks.append(h)
        return hooks

    def remove_hooks(hooks):
        for h in hooks:
            h.remove()

    # --- Eval: perplexity ---
    @torch.no_grad()
    def eval_perplexity(model, loader, gate_mask=None):
        model.eval()
        hooks = install_ffn_hooks(model, gate_mask) if gate_mask is not None else []
        total_loss, n_tokens = 0.0, 0
        for (ids_b,) in loader:
            ids_b = ids_b.to(DEVICE)
            mask_b = (ids_b != tokenizer.pad_token_id).long()
            out = model(input_ids=ids_b, attention_mask=mask_b, labels=ids_b)
            n_valid = mask_b.sum().item()
            total_loss += out.loss.item() * n_valid
            n_tokens += n_valid
        remove_hooks(hooks)
        avg_loss = total_loss / max(n_tokens, 1)
        return {"loss": avg_loss, "perplexity": float(np.exp(avg_loss))}

    # --- Eval: generation ---
    @torch.no_grad()
    def generate_samples(model, prompts, gate_mask=None, max_new=128):
        model.eval()
        hooks = install_ffn_hooks(model, gate_mask) if gate_mask is not None else []
        outputs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, use_cache=True,
            )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs.append({"prompt": prompt, "generation": text[:500]})
        remove_hooks(hooks)
        return outputs

    # ==================================================================
    # PHASE 1: UNGATED BASELINE
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("PHASE 1: UNGATED BASELINE (pre-trained TinyLlama)", flush=True)
    print("=" * 72, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    baseline_ppl = eval_perplexity(model, val_loader)
    print(f"  Baseline perplexity: {baseline_ppl['perplexity']:.2f}", flush=True)

    baseline_gens = generate_samples(model, EVAL_PROMPTS[:5])
    for g in baseline_gens:
        print(f"  [{g['prompt']}] → {g['generation'][:120]}...", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ==================================================================
    # PHASE 2: CO-TRAIN WITH FFN GATES (R=4)
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print(f"PHASE 2: CO-TRAIN WITH FFN GATES (R={R})", flush=True)
    print("=" * 72, flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    for ep in range(epochs):
        model.train()
        total_loss, n_steps = 0.0, 0
        t0 = time.time()
        for (ids_b,) in train_loader:
            ids_b = ids_b.to(DEVICE)
            pad_mask = (ids_b != tokenizer.pad_token_id).long()
            optimizer.zero_grad(set_to_none=True)

            regime_loss = 0.0
            for r in range(R):
                hooks = install_ffn_hooks(model, gate_masks[r])
                out = model(input_ids=ids_b, attention_mask=pad_mask,
                            labels=ids_b)
                regime_loss = regime_loss + out.loss / R
                remove_hooks(hooks)

            regime_loss.backward()
            optimizer.step()
            total_loss += regime_loss.item()
            n_steps += 1

            if n_steps % 50 == 0:
                print(f"    step {n_steps}, loss={total_loss / n_steps:.4f}",
                      flush=True)

        elapsed = time.time() - t0
        avg = total_loss / n_steps
        print(f"  epoch {ep+1}/{epochs}: loss={avg:.4f} ({elapsed:.0f}s)",
              flush=True)

    # ==================================================================
    # PHASE 3: EVALUATE GATED MODEL
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("PHASE 3: EVALUATE GATED MODEL", flush=True)
    print("=" * 72, flush=True)

    results = {
        "config": {
            "model": MODEL_NAME, "R": R, "epochs": epochs,
            "lr": lr, "hidden_dim": HIDDEN_DIM,
        },
        "baseline_perplexity": baseline_ppl,
        "baseline_generations": baseline_gens,
        "regimes": {},
    }

    # Correct-key perplexity for each regime
    for r in range(R):
        ppl = eval_perplexity(model, val_loader, gate_masks[r])
        gens = generate_samples(model, EVAL_PROMPTS[:10], gate_masks[r])
        gap = ppl["perplexity"] - baseline_ppl["perplexity"]
        gap_pct = gap / baseline_ppl["perplexity"] * 100

        print(f"\n  Regime {r}: perplexity={ppl['perplexity']:.2f} "
              f"(gap={gap:+.2f}, {gap_pct:+.1f}%)", flush=True)
        for g in gens[:3]:
            print(f"    [{g['prompt']}]", flush=True)
            print(f"    → {g['generation'][:150]}...", flush=True)

        results["regimes"][f"regime_{r}"] = {
            "perplexity": ppl,
            "gap_abs": gap,
            "gap_pct": gap_pct,
            "generations": gens,
        }

    # Wrong-key perplexity (regime 0 model, regime 1 mask)
    wrong_ppl = eval_perplexity(model, val_loader, gate_masks[1])
    wrong_gens = generate_samples(model, EVAL_PROMPTS[:5], gate_masks[1])
    print(f"\n  Wrong-key (regime 0 data, regime 1 mask): "
          f"perplexity={wrong_ppl['perplexity']:.2f}", flush=True)
    for g in wrong_gens[:3]:
        print(f"    [{g['prompt']}] → {g['generation'][:120]}...", flush=True)

    results["wrong_key"] = {
        "perplexity": wrong_ppl,
        "generations": wrong_gens,
    }

    # No-gate (all dims active) on co-trained model
    nogated_ppl = eval_perplexity(model, val_loader)
    print(f"\n  No gate (all dims): perplexity={nogated_ppl['perplexity']:.2f}",
          flush=True)
    results["no_gate_cotrained"] = nogated_ppl

    # --- Summary ---
    mean_ppl = np.mean([
        results["regimes"][f"regime_{r}"]["perplexity"]["perplexity"]
        for r in range(R)
    ])
    results["summary"] = {
        "baseline_ppl": baseline_ppl["perplexity"],
        "mean_gated_ppl": float(mean_ppl),
        "gap_pct": float((mean_ppl - baseline_ppl["perplexity"])
                         / baseline_ppl["perplexity"] * 100),
        "wrong_key_ppl": wrong_ppl["perplexity"],
        "verdict": "DEGRADED" if mean_ppl > baseline_ppl["perplexity"] * 1.1
                   else "VIABLE",
    }

    print(f"\n{'=' * 72}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"  Baseline perplexity:   {baseline_ppl['perplexity']:.2f}", flush=True)
    print(f"  Mean gated perplexity: {mean_ppl:.2f} "
          f"({results['summary']['gap_pct']:+.1f}%)", flush=True)
    print(f"  Wrong-key perplexity:  {wrong_ppl['perplexity']:.2f}", flush=True)
    print(f"  Verdict: {results['summary']['verdict']}", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


@app.local_entrypoint()
def main(
    epochs: int = 3,
    r: int = 4,
    lr: float = 2e-5,
):
    t0 = time.time()
    results = run_generative_ffn.remote(epochs, r, lr)
    wall_time = time.time() - t0

    print(f"\nWall clock: {wall_time:.0f}s ({wall_time / 60:.1f} min)")

    s = results["summary"]
    print(f"\n{'=' * 72}")
    print("PHASE A: FFN-ONLY GATED GENERATION RESULTS")
    print(f"{'=' * 72}")
    print(f"  Model:     {results['config']['model']}")
    print(f"  R:         {results['config']['R']}")
    print(f"  Epochs:    {results['config']['epochs']}")
    print(f"  Baseline:  {s['baseline_ppl']:.2f} perplexity")
    print(f"  Gated:     {s['mean_gated_ppl']:.2f} perplexity ({s['gap_pct']:+.1f}%)")
    print(f"  Wrong-key: {s['wrong_key_ppl']:.2f} perplexity")
    print(f"  Verdict:   {s['verdict']}")

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out / f"generative_ffn_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
