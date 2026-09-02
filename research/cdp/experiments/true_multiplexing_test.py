"""True Multiplexing: R Tenants in ONE Forward Pass

The question: can we serve R different-regime requests in a single batch,
through a single forward pass of one model, by applying per-sample
permutations to the residual stream?

If yes, multiplexing is literally free: R tenants at the cost of 1
forward pass.  The permutation overhead is O(d) per injection point;
the matmuls are O(d^2).  Ratio ≈ 0.4%.  Should be noise.

The plumbing: at every point where the residual stream enters or exits
a weight matrix or LayerNorm, we apply per-sample index gathers:
  - Input-side (cols conjugation): gather with inv_P per sample
  - Output-side (rows conjugation): gather with P per sample
  - LayerNorm: inverse-permute → LN → permute

USAGE
=====
    python3 experiments/true_multiplexing_test.py
    python3 experiments/true_multiplexing_test.py --R 8
    python3 experiments/true_multiplexing_test.py --device cpu

Requires: torch, transformers, datasets
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

MODEL_ID = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_ID = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load_ag_news(max_train=8000, max_test=7600):
    from datasets import load_dataset
    from transformers import DistilBertTokenizerFast

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    )
    ds = load_dataset(DATASET_ID, revision=DATASET_REVISION)

    def encode(split, max_n):
        texts = split["text"][:max_n]
        labels = split["label"][:max_n]
        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=128, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"],
                             torch.tensor(labels, dtype=torch.long))

    return encode(ds["train"], max_train), encode(ds["test"], max_test)


# ---------------------------------------------------------------------------
# CONJUGATION (for ground-truth comparison)
# ---------------------------------------------------------------------------

@torch.no_grad()
def conjugate_residual_basis(backbone, classifier, perm, device):
    p = torch.as_tensor(perm, dtype=torch.long, device=device)
    def cols(w): w.data = w.data[:, p].contiguous()
    def rows(w): w.data = w.data[p, :].contiguous()
    def vec(b):  b.data = b.data[p].contiguous()

    emb = backbone.embeddings
    cols(emb.word_embeddings.weight)
    cols(emb.position_embeddings.weight)
    vec(emb.LayerNorm.weight); vec(emb.LayerNorm.bias)

    for layer in backbone.transformer.layer:
        attn = layer.attention
        cols(attn.q_lin.weight); cols(attn.k_lin.weight); cols(attn.v_lin.weight)
        rows(attn.out_lin.weight); vec(attn.out_lin.bias)
        cols(layer.ffn.lin1.weight)
        rows(layer.ffn.lin2.weight); vec(layer.ffn.lin2.bias)
        vec(layer.sa_layer_norm.weight); vec(layer.sa_layer_norm.bias)
        vec(layer.output_layer_norm.weight); vec(layer.output_layer_norm.bias)

    cols(classifier.weight)


# ---------------------------------------------------------------------------
# PER-SAMPLE PERMUTATION HELPERS
# ---------------------------------------------------------------------------

def _gather_per_sample(x, perm_indices):
    """Gather along dim=2 with per-sample permutation indices.

    x: (B, seq_len, d)
    perm_indices: (B, d)
    returns: (B, seq_len, d) where out[b, t, i] = x[b, t, perm_indices[b, i]]
    """
    B, S, d = x.shape
    idx = perm_indices.unsqueeze(1).expand(B, S, d)
    return torch.gather(x, 2, idx)


def _layernorm_per_sample(x, ln_module, inv_perms, fwd_perms):
    """Apply LayerNorm with per-sample inverse-permute/re-permute."""
    x_orig = _gather_per_sample(x, inv_perms)
    normed = ln_module(x_orig)
    return _gather_per_sample(normed, fwd_perms)


def _linear_input_side(x, linear, inv_perms):
    """Linear layer where conjugation permutes input columns.

    Equivalent to: inverse-permute input → apply unpermuted linear.
    Output is in a NON-residual space (head-space or FFN-hidden), so
    no re-permute needed.
    """
    x_orig = _gather_per_sample(x, inv_perms)
    return F.linear(x_orig, linear.weight, linear.bias)


def _linear_output_side(x, linear, fwd_perms):
    """Linear layer where conjugation permutes output rows.

    Input is in a NON-residual space.  Apply unpermuted linear, then
    permute output back to per-sample residual basis.
    """
    out_orig = F.linear(x, linear.weight, linear.bias)
    return _gather_per_sample(out_orig, fwd_perms)


# ---------------------------------------------------------------------------
# MULTIPLEXED FORWARD PASS
# ---------------------------------------------------------------------------

@torch.no_grad()
def multiplexed_forward(
    backbone, classifier,
    input_ids, attention_mask,
    fwd_perms, inv_perms,
):
    """Forward pass with per-sample residual-stream permutations.

    Each sample in the batch uses a different permutation, but all
    samples share the same unpermuted model weights.  This is true
    multiplexing: R tenants in 1 forward pass.

    fwd_perms: (B, d) — permutation P per sample (original → permuted)
    inv_perms: (B, d) — inverse permutation P^{-1} per sample
    """
    backbone.eval()
    # --- Embeddings: output is residual, permute per sample ---
    emb_out = backbone.embeddings(input_ids)  # (B, seq, d)
    hidden = _gather_per_sample(emb_out, fwd_perms)

    # --- Transformer layers ---
    # DistilBERT uses POST-norm:
    #   attn_out = attention(hidden)
    #   sa_out = sa_layer_norm(attn_out + hidden)   ← residual INSIDE norm
    #   ffn_out = ffn(sa_out)
    #   output = output_layer_norm(ffn_out + sa_out) ← residual INSIDE norm

    if attention_mask is not None:
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(
            dtype=hidden.dtype)
        extended_mask = (1.0 - extended_mask) * torch.finfo(hidden.dtype).min
    else:
        extended_mask = None

    for layer in backbone.transformer.layer:
        attn = layer.attention
        d_model = hidden.shape[-1]
        n_heads = attn.n_heads
        d_k = d_model // n_heads

        # --- Attention (input-side: inverse-permute → Q/K/V) ---
        hidden_orig = _gather_per_sample(hidden, inv_perms)

        input_shape = hidden_orig.shape[:-1]  # (B, seq)
        q = attn.q_lin(hidden_orig).view(*input_shape, n_heads, d_k).transpose(1, 2)
        k = attn.k_lin(hidden_orig).view(*input_shape, n_heads, d_k).transpose(1, 2)
        v = attn.v_lin(hidden_orig).view(*input_shape, n_heads, d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) / (d_k ** 0.5)
        if extended_mask is not None:
            scores = scores + extended_mask
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(*input_shape, d_model)

        # Out projection (output-side: project → permute)
        attn_output_perm = _linear_output_side(
            attn_output, attn.out_lin, fwd_perms)

        # SA LayerNorm: POST-norm over (attn_output + hidden), both in permuted basis
        sa_sum_perm = attn_output_perm + hidden
        sa_output_perm = _layernorm_per_sample(
            sa_sum_perm, layer.sa_layer_norm, inv_perms, fwd_perms)

        # --- FFN (input-side: inverse-permute → lin1) ---
        sa_output_orig = _gather_per_sample(sa_output_perm, inv_perms)
        ffn_hidden = F.gelu(layer.ffn.lin1(sa_output_orig))
        ffn_hidden = layer.ffn.dropout(ffn_hidden)

        # FFN lin2 (output-side: project → permute)
        ffn_out_perm = _linear_output_side(
            ffn_hidden, layer.ffn.lin2, fwd_perms)

        # Output LayerNorm: POST-norm over (ffn_out + sa_output), both permuted
        out_sum_perm = ffn_out_perm + sa_output_perm
        hidden = _layernorm_per_sample(
            out_sum_perm, layer.output_layer_norm, inv_perms, fwd_perms)

    # --- CLS token: inverse-permute back to original basis ---
    hidden_orig = _gather_per_sample(hidden, inv_perms)
    cls = hidden_orig[:, 0, :]

    # --- Classifier (shared, in original basis) ---
    logits = classifier(cls)
    return logits


# ---------------------------------------------------------------------------
# GROUND TRUTH: SERIAL EVALUATION
# ---------------------------------------------------------------------------

@torch.no_grad()
def serial_eval(backbone, classifier, input_ids, attention_mask, perm, device):
    """Evaluate one sample through a pre-conjugated model copy."""
    bb = copy.deepcopy(backbone)
    cl = copy.deepcopy(classifier)
    conjugate_residual_basis(bb, cl, perm, device)
    bb.eval(); cl.eval()
    out = bb(input_ids=input_ids, attention_mask=attention_mask)
    cls = out.last_hidden_state[:, 0, :]
    logits = cl(cls)
    del bb, cl
    return logits


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="True Multiplexing: R tenants in 1 forward pass")
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    R = args.R
    print("=" * 70)
    print(f"  True Multiplexing: {R} Tenants in 1 Forward Pass")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  R: {R}")
    print(flush=True)

    # --- Load data ---
    print("  Loading AG News...", flush=True)
    train_ds, test_ds = load_ag_news()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # --- Train base model ---
    from transformers import DistilBertModel
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"  Training base model ({args.epochs} epochs)...", flush=True)
    backbone = DistilBertModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device)
    classifier = nn.Linear(768, 4).to(device)
    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()), lr=args.lr)

    for ep in range(args.epochs):
        backbone.train(); classifier.train()
        t0 = time.time()
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            cls = backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            loss = F.cross_entropy(classifier(cls), labels)
            loss.backward()
            opt.step()
        print(f"    epoch {ep+1}/{args.epochs}: {time.time()-t0:.1f}s", flush=True)

    backbone.eval(); classifier.eval()

    # --- Base accuracy ---
    correct = total = 0
    with torch.no_grad():
        for ids, mask, labels in test_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            cls = backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            pred = classifier(cls).argmax(-1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    base_acc = correct / total
    print(f"\n  Base accuracy: {base_acc:.6f}", flush=True)

    # --- Generate R permutations ---
    perms_np = []
    inv_perms_np = []
    for r in range(R):
        if r == 0:
            p = np.arange(768)
        else:
            p = np.random.RandomState(1000 + r).permutation(768)
        perms_np.append(p)
        inv_perms_np.append(np.argsort(p))

    # =====================================================================
    # TEST 1: CORRECTNESS — multiplexed output matches serial conjugated
    # =====================================================================
    print("\n  --- Test 1: Correctness (multiplexed vs serial) ---", flush=True)

    # Grab R samples from test set (may span multiple batches)
    all_ids, all_mask = [], []
    for ids_b, mask_b, _ in test_loader:
        all_ids.append(ids_b)
        all_mask.append(mask_b)
        if sum(x.shape[0] for x in all_ids) >= R:
            break
    sample_ids = torch.cat(all_ids, dim=0)[:R].to(device)   # (R, seq_len)
    sample_mask = torch.cat(all_mask, dim=0)[:R].to(device)

    # Build per-sample permutation tensors
    fwd_perms = torch.stack([
        torch.as_tensor(perms_np[r], dtype=torch.long, device=device)
        for r in range(R)
    ])  # (R, d)
    inv_perms = torch.stack([
        torch.as_tensor(inv_perms_np[r], dtype=torch.long, device=device)
        for r in range(R)
    ])  # (R, d)

    # Multiplexed: all R in one forward pass
    mux_logits = multiplexed_forward(
        backbone, classifier, sample_ids, sample_mask,
        fwd_perms, inv_perms,
    )

    # Serial: each through its own conjugated copy
    serial_logits = []
    for r in range(R):
        sl = serial_eval(
            backbone, classifier,
            sample_ids[r:r+1], sample_mask[r:r+1],
            perms_np[r], device,
        )
        serial_logits.append(sl)
    serial_logits = torch.cat(serial_logits, dim=0)

    # Compare
    max_diff = (mux_logits - serial_logits).abs().max().item()
    preds_match = (mux_logits.argmax(-1) == serial_logits.argmax(-1)).all().item()

    print(f"    Max logit difference: {max_diff:.2e}")
    print(f"    Predictions match:    {preds_match}")
    if R <= 16:
        print(f"    Multiplexed preds:    {mux_logits.argmax(-1).tolist()}")
        print(f"    Serial preds:         {serial_logits.argmax(-1).tolist()}")
    else:
        print(f"    (preds list omitted, R={R})")

    bit_exact = max_diff < 1e-4
    print(f"    Result: {'PASS' if bit_exact else 'MISMATCH'} "
          f"(threshold 1e-4)", flush=True)

    # =====================================================================
    # TEST 2: FULL ACCURACY — multiplexed eval on entire test set
    # =====================================================================
    print("\n  --- Test 2: Full accuracy (multiplexed eval) ---", flush=True)

    mux_correct = [0] * R
    mux_total = 0

    for ids, mask, labels in test_loader:
        B_actual = ids.shape[0]
        # Build per-sample permutation tensors for this batch
        fp = torch.stack([
            torch.as_tensor(perms_np[r % R], dtype=torch.long, device=device)
            for r in range(B_actual)
        ])
        ip = torch.stack([
            torch.as_tensor(inv_perms_np[r % R], dtype=torch.long, device=device)
            for r in range(B_actual)
        ])

        logits = multiplexed_forward(
            backbone, classifier,
            ids.to(device), mask.to(device), fp, ip,
        )
        preds = logits.argmax(-1)
        labels_d = labels.to(device)

        for b in range(B_actual):
            r = b % R
            if preds[b] == labels_d[b]:
                mux_correct[r] += 1
        mux_total += B_actual // R

    print("    Per-regime accuracy (multiplexed, same batch):")
    show_regimes = min(R, 8)
    for r in range(show_regimes):
        acc = mux_correct[r] / max(mux_total, 1)
        gap = acc - base_acc
        print(f"      regime {r}: {acc:.6f} (gap={gap:+.2e})")
    if R > show_regimes:
        print(f"      ... ({R - show_regimes} more regimes omitted)")

    mean_mux_acc = sum(mux_correct) / max(mux_total * R, 1)
    print(f"    Mean multiplexed accuracy: {mean_mux_acc:.6f}")
    print(f"    Base accuracy:             {base_acc:.6f}")
    print(f"    Gap:                       {mean_mux_acc - base_acc:+.2e}", flush=True)

    # =====================================================================
    # TEST 3: TIMING — multiplexed vs serial
    # =====================================================================
    print("\n  --- Test 3: Throughput (multiplexed vs serial) ---", flush=True)

    n_warmup = 3
    n_trials = 20
    batch_for_timing = max(R, 32)

    # Gather enough samples for timing (may span multiple batches)
    t_ids, t_mask = [], []
    for ids_b, mask_b, _ in test_loader:
        t_ids.append(ids_b)
        t_mask.append(mask_b)
        if sum(x.shape[0] for x in t_ids) >= batch_for_timing:
            break
    timing_ids = torch.cat(t_ids, dim=0)[:batch_for_timing].to(device)
    timing_mask = torch.cat(t_mask, dim=0)[:batch_for_timing].to(device)

    # Build permutations for the batch (round-robin across R regimes)
    fp_timing = torch.stack([
        torch.as_tensor(perms_np[b % R], dtype=torch.long, device=device)
        for b in range(batch_for_timing)
    ])
    ip_timing = torch.stack([
        torch.as_tensor(inv_perms_np[b % R], dtype=torch.long, device=device)
        for b in range(batch_for_timing)
    ])

    def sync():
        if device.type == "mps": torch.mps.synchronize()
        elif device.type == "cuda": torch.cuda.synchronize()

    # --- Baseline: normal forward (no permutation) ---
    for _ in range(n_warmup):
        backbone(input_ids=timing_ids, attention_mask=timing_mask)
    sync()
    t0 = time.time()
    for _ in range(n_trials):
        backbone(input_ids=timing_ids, attention_mask=timing_mask)
    sync()
    t_base = (time.time() - t0) / n_trials * 1000

    # --- Multiplexed: R regimes in 1 forward pass ---
    for _ in range(n_warmup):
        multiplexed_forward(backbone, classifier, timing_ids, timing_mask,
                            fp_timing, ip_timing)
    sync()
    t0 = time.time()
    for _ in range(n_trials):
        multiplexed_forward(backbone, classifier, timing_ids, timing_mask,
                            fp_timing, ip_timing)
    sync()
    t_mux = (time.time() - t0) / n_trials * 1000

    # --- Serial: R separate forward passes (1 sample each) ---
    chunk = max(batch_for_timing // R, 1)
    for _ in range(n_warmup):
        for r in range(R):
            i = (r * chunk) % batch_for_timing
            backbone(input_ids=timing_ids[i:i+chunk],
                     attention_mask=timing_mask[i:i+chunk])
    sync()
    t0 = time.time()
    for _ in range(n_trials):
        for r in range(R):
            i = (r * chunk) % batch_for_timing
            backbone(input_ids=timing_ids[i:i+chunk],
                     attention_mask=timing_mask[i:i+chunk])
    sync()
    t_serial = (time.time() - t0) / n_trials * 1000

    overhead_pct = (t_mux - t_base) / t_base * 100
    speedup_vs_serial = t_serial / t_mux if t_mux > 0 else float('inf')

    print(f"    Batch size: {batch_for_timing} ({R} regimes × "
          f"{batch_for_timing//R} samples each)")
    print(f"    Base (no perm):     {t_base:.2f} ms")
    print(f"    Multiplexed:        {t_mux:.2f} ms  "
          f"(overhead: {overhead_pct:+.1f}%)")
    print(f"    Serial ({R}× fwd):   {t_serial:.2f} ms")
    print(f"    Speedup vs serial:  {speedup_vs_serial:.2f}×")
    print(f"\n    Result: {'FREE' if overhead_pct < 10 else 'OVERHEAD'} "
          f"multiplexing ({overhead_pct:+.1f}% overhead)", flush=True)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  R tenants:            {R}")
    print(f"  Correctness:          {'PASS' if bit_exact else 'FAIL'} "
          f"(max diff={max_diff:.2e})")
    print(f"  Accuracy gap:         {mean_mux_acc - base_acc:+.2e}")
    print(f"  Throughput overhead:   {overhead_pct:+.1f}%")
    print(f"  Speedup vs serial:    {speedup_vs_serial:.2f}×")
    print(f"\n  Verdict: multiplexing is "
          f"{'FREE' if overhead_pct < 10 and bit_exact else 'NOT FREE'}")
    print(flush=True)

    # --- Save results ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    results = {
        "R": R,
        "device": str(device),
        "base_acc": base_acc,
        "mean_mux_acc": round(mean_mux_acc, 6),
        "accuracy_gap": round(mean_mux_acc - base_acc, 6),
        "max_logit_diff": max_diff,
        "bit_exact": bit_exact,
        "t_base_ms": round(t_base, 2),
        "t_multiplexed_ms": round(t_mux, 2),
        "t_serial_ms": round(t_serial, 2),
        "overhead_pct": round(overhead_pct, 1),
        "speedup_vs_serial": round(speedup_vs_serial, 2),
    }
    out_path = out_dir / f"true_multiplexing_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {out_path}")


if __name__ == "__main__":
    main()
