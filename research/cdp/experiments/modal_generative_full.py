"""Phase B: FFN + Per-Regime Attention Heads — Full Closure Generation.

Co-train TinyLlama 1.1B at R=4 with BOTH FFN gating AND per-regime
attention heads (whole-head assignment). Each regime gets 8 of 32 query
heads (1 of 4 KV head groups under GQA).

This is expected to RESTORE generation quality that Phase A (FFN-only)
degraded, closing the paper's biggest acknowledged limitation.

TinyLlama GQA layout:
    32 query heads, 4 KV heads → 8 query heads per KV group
    At R=4: each regime gets 1 KV group = 8 query heads = full capacity

Files uploaded to Modal:
    benchmark_masks.py  (mounted AS gate_crypto.py — crypto-free shim)

Usage:
    modal run experiments/modal_generative_full.py
    modal run experiments/modal_generative_full.py --epochs 5
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal
from modal_schemen_image import install_current_schemen

app = modal.App("cdp-generative-full")

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
N_QUERY_HEADS = 32
N_KV_HEADS = 4
QUERIES_PER_KV = N_QUERY_HEADS // N_KV_HEADS  # 8

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


def derive_head_assignments(key_bytes: bytes, n_kv_heads: int, n_regimes: int):
    """Assign KV head groups to regimes via deterministic shuffle."""
    import hashlib

    import numpy as np

    digest = hashlib.sha256(key_bytes).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**63)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_kv_heads)
    group_size = n_kv_heads // n_regimes
    return {r: perm[r * group_size:(r + 1) * group_size].tolist()
            for r in range(n_regimes)}


@app.function(max_containers=3, gpu="A100", image=gpu_image, timeout=14400)
def run_generative_full(epochs: int, R: int, lr: float) -> dict:
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

    train_loader = DataLoader(TensorDataset(train_ids), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ids), batch_size=8)

    # --- Gate masks (FFN) ---
    key = GateKey.generate()
    ffn_masks = {}
    for r in range(R):
        mask_np = GateMask.derive(key.secret, r, HIDDEN_DIM, R).to_numpy()
        ffn_masks[r] = torch.tensor(mask_np, dtype=torch.bfloat16, device=DEVICE)

    # --- Head assignments (attention) ---
    head_map = derive_head_assignments(key.secret, N_KV_HEADS, R)
    print(f"  Head assignments (KV groups): {head_map}", flush=True)

    # --- Build per-regime attention blocks ---
    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    n_layers = len(model.model.layers)
    head_dim = HIDDEN_DIM // N_QUERY_HEADS  # 64

    # For each layer, create R copies of the self_attn module
    # and a head mask for zeroing non-owned heads in Q/O and KV
    per_regime_attn = {}
    for r in range(R):
        kv_groups = head_map[r]
        q_heads = []
        for kg in kv_groups:
            q_heads.extend(range(kg * QUERIES_PER_KV, (kg + 1) * QUERIES_PER_KV))

        q_mask = torch.zeros(N_QUERY_HEADS, dtype=torch.bfloat16, device=DEVICE)
        q_mask[q_heads] = 1.0
        kv_mask = torch.zeros(N_KV_HEADS, dtype=torch.bfloat16, device=DEVICE)
        kv_mask[kv_groups] = 1.0

        per_regime_attn[r] = {
            "q_heads": q_heads,
            "kv_groups": kv_groups,
            "q_mask": q_mask,      # (N_QUERY_HEADS,)
            "kv_mask": kv_mask,    # (N_KV_HEADS,)
        }

    print(f"  Per-regime heads built for {n_layers} layers", flush=True)

    # --- Hook infrastructure ---
    _active_regime = [0]
    _hooks = []

    def install_hooks(model, regime_id):
        remove_hooks()
        _active_regime[0] = regime_id
        regime = per_regime_attn[regime_id]
        ffn_mask = ffn_masks[regime_id]

        q_mask = regime["q_mask"]
        kv_mask = regime["kv_mask"]

        for layer in model.model.layers:
            # FFN hook
            def make_ffn_hook(m):
                def hook_fn(module, input, output):
                    return output * m
                return hook_fn
            h1 = layer.mlp.register_forward_hook(make_ffn_hook(ffn_mask))

            # Attention head masking hook: zero non-owned heads after attention
            def make_attn_hook(qm, kvm):
                def hook_fn(module, input, output):
                    attn_out = output[0]  # (B, seq, hidden)
                    B, S, D = attn_out.shape
                    reshaped = attn_out.view(B, S, N_QUERY_HEADS, head_dim)
                    masked = reshaped * qm.view(1, 1, N_QUERY_HEADS, 1)
                    return (masked.view(B, S, D),) + output[1:]
                return hook_fn
            h2 = layer.self_attn.register_forward_hook(
                make_attn_hook(q_mask, kv_mask))

            _hooks.extend([h1, h2])

    def remove_hooks():
        for h in _hooks:
            h.remove()
        _hooks.clear()

    # --- Eval helpers ---
    @torch.no_grad()
    def eval_perplexity(model, loader, regime_id=None):
        model.eval()
        if regime_id is not None:
            install_hooks(model, regime_id)
        total_loss, n_tokens = 0.0, 0
        for (ids_b,) in loader:
            ids_b = ids_b.to(DEVICE)
            mask_b = (ids_b != tokenizer.pad_token_id).long()
            out = model(input_ids=ids_b, attention_mask=mask_b, labels=ids_b)
            n_valid = mask_b.sum().item()
            total_loss += out.loss.item() * n_valid
            n_tokens += n_valid
        if regime_id is not None:
            remove_hooks()
        avg_loss = total_loss / max(n_tokens, 1)
        return {"loss": avg_loss, "perplexity": float(np.exp(avg_loss))}

    @torch.no_grad()
    def generate_samples(model, prompts, regime_id=None, max_new=128):
        model.eval()
        if regime_id is not None:
            install_hooks(model, regime_id)
        outputs = []
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = model.generate(
                **inputs, max_new_tokens=max_new, do_sample=False,
                pad_token_id=tokenizer.eos_token_id, use_cache=False,
            )
            gen_ids = out[0][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            outputs.append({"prompt": prompt, "generation": text[:500]})
        if regime_id is not None:
            remove_hooks()
        return outputs

    # ==================================================================
    # PHASE 1: UNGATED BASELINE
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("PHASE 1: UNGATED BASELINE", flush=True)
    print("=" * 72, flush=True)

    baseline_ppl = eval_perplexity(model, val_loader)
    print(f"  Baseline perplexity: {baseline_ppl['perplexity']:.2f}", flush=True)

    baseline_gens = generate_samples(model, EVAL_PROMPTS[:5])
    for g in baseline_gens:
        print(f"  [{g['prompt']}] → {g['generation'][:120]}...", flush=True)

    # ==================================================================
    # PHASE 2: CO-TRAIN WITH FFN + HEAD GATES
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print(f"PHASE 2: CO-TRAIN WITH FFN + HEAD GATES (R={R})", flush=True)
    print("=" * 72, flush=True)

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
                install_hooks(model, r)
                out = model(input_ids=ids_b, attention_mask=pad_mask,
                            labels=ids_b)
                regime_loss = regime_loss + out.loss / R
                remove_hooks()

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
    # PHASE 3: EVALUATE
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("PHASE 3: EVALUATE GATED MODEL", flush=True)
    print("=" * 72, flush=True)

    results = {
        "config": {
            "model": MODEL_NAME, "R": R, "epochs": epochs, "lr": lr,
            "hidden_dim": HIDDEN_DIM, "n_query_heads": N_QUERY_HEADS,
            "n_kv_heads": N_KV_HEADS, "head_assignments": head_map,
            "gating": "FFN + per-regime attention heads",
        },
        "baseline_perplexity": baseline_ppl,
        "baseline_generations": baseline_gens,
        "regimes": {},
    }

    for r in range(R):
        ppl = eval_perplexity(model, val_loader, regime_id=r)
        gens = generate_samples(model, EVAL_PROMPTS[:10], regime_id=r)
        gap = ppl["perplexity"] - baseline_ppl["perplexity"]
        gap_pct = gap / baseline_ppl["perplexity"] * 100

        print(f"\n  Regime {r} (heads {per_regime_attn[r]['q_heads']}): "
              f"ppl={ppl['perplexity']:.2f} (gap={gap_pct:+.1f}%)", flush=True)
        for g in gens[:3]:
            print(f"    [{g['prompt']}]", flush=True)
            print(f"    → {g['generation'][:150]}...", flush=True)

        results["regimes"][f"regime_{r}"] = {
            "perplexity": ppl, "gap_abs": gap, "gap_pct": gap_pct,
            "heads": per_regime_attn[r]["q_heads"],
            "generations": gens,
        }

    # Wrong-key test
    wrong_ppl = eval_perplexity(model, val_loader, regime_id=1)
    wrong_gens = generate_samples(model, EVAL_PROMPTS[:5], regime_id=1)
    print(f"\n  Wrong-key: ppl={wrong_ppl['perplexity']:.2f}", flush=True)

    results["wrong_key"] = {
        "perplexity": wrong_ppl,
        "generations": wrong_gens,
    }

    # No-gate on co-trained model
    nogated_ppl = eval_perplexity(model, val_loader)
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
        "verdict": "VIABLE" if mean_ppl < baseline_ppl["perplexity"] * 1.1
                   else "DEGRADED",
    }

    print(f"\n{'=' * 72}", flush=True)
    print("SUMMARY — FULL CLOSURE (FFN + HEADS)", flush=True)
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
    results = run_generative_full.remote(epochs, r, lr)
    wall_time = time.time() - t0

    print(f"\nWall clock: {wall_time:.0f}s ({wall_time / 60:.1f} min)")

    s = results["summary"]
    print(f"\n{'=' * 72}")
    print("PHASE B: FFN + HEAD-GATED GENERATION RESULTS")
    print(f"{'=' * 72}")
    print(f"  Model:     {results['config']['model']}")
    print(f"  R:         {results['config']['R']}")
    print(f"  Gating:    {results['config']['gating']}")
    print(f"  Epochs:    {results['config']['epochs']}")
    print(f"  Baseline:  {s['baseline_ppl']:.2f} perplexity")
    print(f"  Gated:     {s['mean_gated_ppl']:.2f} perplexity ({s['gap_pct']:+.1f}%)")
    print(f"  Wrong-key: {s['wrong_key_ppl']:.2f} perplexity")
    print(f"  Verdict:   {s['verdict']}")

    out = Path("experiments/results")
    out.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out / f"generative_full_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
