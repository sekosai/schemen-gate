"""Orthogonal Superposition R-Scaling Study

Train ONE DistilBERT model on AG News.  Place it into R orthogonal
(permutation) slots via residual-stream conjugation.  No per-regime
training.  Each regime IS the trained model in a permuted basis.

Five experiments:
  A. R-sweep: gap vs R from 4 to 1024
  B. Component ablation: skip one conjugation tendril, measure collapse
  C. On-the-fly permutation: permute activations at runtime, not weights
  D. Concurrent stress test: batched serial throughput vs R
  E. Cross-regime isolation: CLS cosine similarity between regimes

USAGE
=====
    python3 experiments/orthogonal_superposition_sweep.py          # all experiments
    python3 experiments/orthogonal_superposition_sweep.py --only A  # just R-sweep
    python3 experiments/orthogonal_superposition_sweep.py --only AB # A and B
    python3 experiments/orthogonal_superposition_sweep.py --device cpu

Requires: torch, transformers, datasets
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

MODEL_ID = "distilbert-base-uncased"
MODEL_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
DATASET_ID = "fancyzhx/ag_news"
DATASET_REVISION = "eb185aade064a813bc0b7f42de02595523103ca4"


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------

def load_ag_news(
    max_train: int = 8000,
    max_test: int = 7600,
) -> tuple[TensorDataset, TensorDataset]:
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
        enc = tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt",
        )
        return TensorDataset(
            enc["input_ids"], enc["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )

    return encode(ds["train"], max_train), encode(ds["test"], max_test)


# ---------------------------------------------------------------------------
# TRAIN / EVAL
# ---------------------------------------------------------------------------

def train_base_model(
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 4,
    lr: float = 2e-5,
    seed: int = 0,
) -> tuple:
    """Train a plain DistilBERT classifier.  Returns (backbone, classifier, accuracy)."""
    from transformers import DistilBertModel

    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = DistilBertModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
    ).to(device)
    classifier = nn.Linear(768, 4).to(device)
    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(classifier.parameters()), lr=lr,
    )

    for ep in range(epochs):
        backbone.train()
        classifier.train()
        t_ep = time.time()
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            cls = backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
            loss = F.cross_entropy(classifier(cls), labels)
            loss.backward()
            opt.step()
        print(f"    epoch {ep+1}/{epochs}: {time.time()-t_ep:.1f}s", flush=True)

    acc = evaluate(backbone, classifier, test_loader, device)
    return backbone, classifier, acc


@torch.no_grad()
def evaluate(backbone, classifier, loader, device) -> float:
    backbone.eval()
    classifier.eval()
    correct = total = 0
    for ids, mask, labels in loader:
        ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
        cls = backbone(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        pred = classifier(cls).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


# ---------------------------------------------------------------------------
# CONJUGATION
# ---------------------------------------------------------------------------

@torch.no_grad()
def conjugate_residual_basis(
    backbone, classifier, perm: np.ndarray, device: torch.device,
    skip: set[str] | None = None,
) -> None:
    """Permute the residual-stream basis in place.

    `skip` is an optional set of component names to deliberately exclude
    from conjugation (for the ablation study).  Valid names:
        embeddings, emb_layernorm, sa_layernorm, out_layernorm,
        q_lin, k_lin, v_lin, out_lin, ffn_lin1, ffn_lin2, classifier
    """
    skip = skip or set()
    p = torch.as_tensor(perm, dtype=torch.long, device=device)

    def cols(w):
        w.data = w.data[:, p].contiguous()

    def rows(w):
        w.data = w.data[p, :].contiguous()

    def vec(b):
        b.data = b.data[p].contiguous()

    emb = backbone.embeddings
    if "embeddings" not in skip:
        cols(emb.word_embeddings.weight)
        cols(emb.position_embeddings.weight)
    if "emb_layernorm" not in skip:
        vec(emb.LayerNorm.weight)
        vec(emb.LayerNorm.bias)

    for layer in backbone.transformer.layer:
        attn = layer.attention
        if "q_lin" not in skip:
            cols(attn.q_lin.weight)
        if "k_lin" not in skip:
            cols(attn.k_lin.weight)
        if "v_lin" not in skip:
            cols(attn.v_lin.weight)
        if "out_lin" not in skip:
            rows(attn.out_lin.weight)
            vec(attn.out_lin.bias)
        if "ffn_lin1" not in skip:
            cols(layer.ffn.lin1.weight)
        if "ffn_lin2" not in skip:
            rows(layer.ffn.lin2.weight)
            vec(layer.ffn.lin2.bias)
        if "sa_layernorm" not in skip:
            vec(layer.sa_layer_norm.weight)
            vec(layer.sa_layer_norm.bias)
        if "out_layernorm" not in skip:
            vec(layer.output_layer_norm.weight)
            vec(layer.output_layer_norm.bias)

    if "classifier" not in skip:
        cols(classifier.weight)


def model_size_mb(backbone, classifier) -> float:
    total = sum(p.numel() * p.element_size() for p in backbone.parameters())
    total += sum(p.numel() * p.element_size() for p in classifier.parameters())
    return total / (1024 * 1024)


# ---------------------------------------------------------------------------
# EXPERIMENT A: R-SWEEP
# ---------------------------------------------------------------------------

def experiment_a_r_sweep(
    backbone, classifier, test_loader, device,
    r_values: list[int],
) -> dict:
    """Superpose into R slots, evaluate each, measure gap and timing."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT A: R-Sweep (Serial Addressed Use)")
    print("=" * 70)

    base_acc = evaluate(backbone, classifier, test_loader, device)
    print(f"\n  Base model accuracy: {base_acc:.6f}")
    base_mb = model_size_mb(backbone, classifier)
    print(f"  Base model size: {base_mb:.1f} MB")

    results = {"base_acc": base_acc, "base_mb": base_mb, "sweeps": []}

    for R in r_values:
        print(f"\n  --- R = {R} ---")
        t0 = time.time()

        accs = []
        for r in range(R):
            if r == 0:
                perm = np.arange(768)
            else:
                perm = np.random.RandomState(1000 + r).permutation(768)

            bb = copy.deepcopy(backbone)
            cl = copy.deepcopy(classifier)
            conjugate_residual_basis(bb, cl, perm, device)
            acc = evaluate(bb, cl, test_loader, device)
            accs.append(acc)
            del bb, cl

            if r < 4 or r == R - 1:
                gap = acc - base_acc
                print(f"    regime {r:>5}: acc={acc:.6f}  gap={gap:+.2e}")
            elif r == 4:
                print(f"    ... (evaluating {R - 5} more regimes) ...")

        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        elapsed = time.time() - t0
        accs_arr = np.array(accs)
        sweep = {
            "R": R,
            "mean_acc": float(accs_arr.mean()),
            "std_acc": float(accs_arr.std()),
            "min_acc": float(accs_arr.min()),
            "max_acc": float(accs_arr.max()),
            "max_abs_gap": float(np.max(np.abs(accs_arr - base_acc))),
            "elapsed_s": round(elapsed, 1),
            "per_regime_s": round(elapsed / R, 2),
            "storage_mb": round(base_mb * R, 1),
            "perm_storage_kb": round(R * 768 * 8 / 1024, 1),
        }
        results["sweeps"].append(sweep)

        print(f"    max |gap| = {sweep['max_abs_gap']:.2e}")
        print(f"    mean={sweep['mean_acc']:.6f} std={sweep['std_acc']:.2e}")
        print(f"    elapsed: {elapsed:.1f}s ({elapsed/R:.2f}s/regime)")
        print(f"    naive storage: {sweep['storage_mb']:.0f} MB "
              f"(vs {sweep['perm_storage_kb']:.1f} KB for permutation vectors)")

    print(f"\n  {'R':>6} {'max|gap|':>12} {'mean_acc':>12} {'elapsed':>10} "
          f"{'naive_MB':>10} {'perm_KB':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
    for s in results["sweeps"]:
        print(f"  {s['R']:>6} {s['max_abs_gap']:>12.2e} {s['mean_acc']:>12.6f} "
              f"{s['elapsed_s']:>9.1f}s {s['storage_mb']:>9.0f} "
              f"{s['perm_storage_kb']:>9.1f}")

    return results


# ---------------------------------------------------------------------------
# EXPERIMENT B: COMPONENT ABLATION
# ---------------------------------------------------------------------------

def experiment_b_ablation(
    backbone, classifier, test_loader, device,
) -> dict:
    """Skip one conjugation component at a time, measure the damage."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT B: Component Ablation (Conjugation Audit)")
    print("=" * 70)

    base_acc = evaluate(backbone, classifier, test_loader, device)

    perm = np.random.RandomState(42).permutation(768)

    ablations = [
        ("full_conjugation",   set()),
        ("skip_embeddings",    {"embeddings"}),
        ("skip_emb_layernorm", {"emb_layernorm"}),
        ("skip_sa_layernorm",  {"sa_layernorm"}),
        ("skip_out_layernorm", {"out_layernorm"}),
        ("skip_all_layernorm", {"emb_layernorm", "sa_layernorm", "out_layernorm"}),
        ("skip_q_lin",         {"q_lin"}),
        ("skip_k_lin",         {"k_lin"}),
        ("skip_v_lin",         {"v_lin"}),
        ("skip_qkv",           {"q_lin", "k_lin", "v_lin"}),
        ("skip_out_lin",       {"out_lin"}),
        ("skip_ffn_lin1",      {"ffn_lin1"}),
        ("skip_ffn_lin2",      {"ffn_lin2"}),
        ("skip_ffn_both",      {"ffn_lin1", "ffn_lin2"}),
        ("skip_classifier",    {"classifier"}),
        ("skip_all_attention", {"q_lin", "k_lin", "v_lin", "out_lin"}),
        ("skip_all_ffn",       {"ffn_lin1", "ffn_lin2"}),
    ]

    results = {"base_acc": base_acc, "ablations": []}
    print(f"\n  Base accuracy: {base_acc:.6f}")
    print(f"\n  {'Ablation':<25} {'Accuracy':>10} {'Gap':>12} {'Collapse':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")

    for name, skip_set in ablations:
        bb = copy.deepcopy(backbone)
        cl = copy.deepcopy(classifier)
        conjugate_residual_basis(bb, cl, perm, device, skip=skip_set)
        acc = evaluate(bb, cl, test_loader, device)
        gap = acc - base_acc
        collapse = base_acc - acc
        del bb, cl

        entry = {
            "name": name,
            "skipped": sorted(skip_set),
            "accuracy": round(acc, 6),
            "gap": round(gap, 6),
            "collapse": round(collapse, 6),
        }
        results["ablations"].append(entry)

        flag = "" if abs(gap) < 1e-4 else " *** COLLAPSE"
        print(f"  {name:<25} {acc:>10.6f} {gap:>+12.6f} {collapse:>+10.4f}{flag}")

    gc.collect()
    return results


# ---------------------------------------------------------------------------
# EXPERIMENT C: ON-THE-FLY PERMUTATION
# ---------------------------------------------------------------------------

def experiment_c_online_permutation(
    backbone, classifier, test_loader, device,
) -> dict:
    """Apply permutation to activations at runtime instead of to weights.

    Approach: wrap the backbone forward pass to permute the embedding
    output and inverse-permute before the classifier.  This is equivalent
    to full weight conjugation but stores only 1 model + permutation vectors.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT C: On-The-Fly Permutation Inference")
    print("=" * 70)

    base_acc = evaluate(backbone, classifier, test_loader, device)
    print(f"\n  Base accuracy (no permutation): {base_acc:.6f}")

    perm_np = np.random.RandomState(77).permutation(768)
    inv_perm_np = np.argsort(perm_np)
    p = torch.as_tensor(perm_np, dtype=torch.long, device=device)
    p_inv = torch.as_tensor(inv_perm_np, dtype=torch.long, device=device)

    # Method 1: Pre-conjugated (ground truth)
    bb_conj = copy.deepcopy(backbone)
    cl_conj = copy.deepcopy(classifier)
    conjugate_residual_basis(bb_conj, cl_conj, perm_np, device)
    conj_acc = evaluate(bb_conj, cl_conj, test_loader, device)
    print(f"  Pre-conjugated accuracy: {conj_acc:.6f}")

    # Method 2: Online permutation — permute embeddings out, inverse before classifier
    # For this to be bit-exact, we need to permute at every residual injection point,
    # which is equivalent to conjugating the weights.  A simpler approximation:
    # permute the full residual stream after embeddings and before classifier.
    # This does NOT work because LayerNorm and attention interact with coordinates.
    # The correct online approach wraps each layer.

    class OnlinePermutedBackbone(nn.Module):
        """Wrap a backbone to apply permutation at every residual injection point."""

        def __init__(self, bb, perm_t, inv_perm_t):
            super().__init__()
            self.bb = bb
            self.perm_t = perm_t
            self.inv_perm_t = inv_perm_t

        def forward(self, input_ids, attention_mask):
            # Embeddings produce residual stream — permute output
            emb_out = self.bb.embeddings(input_ids)
            hidden = emb_out[:, :, self.perm_t]

            # Each transformer layer expects the permuted residual
            # Since LayerNorm gamma/beta are NOT permuted (weights untouched),
            # we must inverse-permute before LN and re-permute after.
            for layer in self.bb.transformer.layer:
                # SA LayerNorm: inverse-permute -> LN -> re-permute
                ln_in = hidden[:, :, self.inv_perm_t]
                normed = layer.sa_layer_norm(ln_in)
                normed = normed[:, :, self.perm_t]

                # Attention: q/k/v columns are NOT permuted, so inverse-permute input
                attn_in = normed[:, :, self.inv_perm_t]
                attn_out, _ = layer.attention(attn_in, attention_mask)
                # out_lin rows are NOT permuted, so permute output
                attn_out = attn_out[:, :, self.perm_t]

                hidden = hidden + attn_out

                # Output LayerNorm
                ln_in2 = hidden[:, :, self.inv_perm_t]
                normed2 = layer.output_layer_norm(ln_in2)
                normed2 = normed2[:, :, self.perm_t]

                # FFN: lin1 columns NOT permuted, inverse-permute input
                ffn_in = normed2[:, :, self.inv_perm_t]
                ffn_out = layer.ffn(ffn_in)
                ffn_out = ffn_out[:, :, self.perm_t]

                hidden = hidden + ffn_out

            # Final: inverse-permute back to original basis for classifier
            hidden = hidden[:, :, self.inv_perm_t]

            # Mimic the output structure
            class Output:
                def __init__(self, h):
                    self.last_hidden_state = h
            return Output(hidden)

    online_bb = OnlinePermutedBackbone(backbone, p, p_inv).to(device)
    online_acc = evaluate(online_bb, classifier, test_loader, device)
    print(f"  Online-permuted accuracy: {online_acc:.6f}")

    gap_vs_conj = abs(online_acc - conj_acc)
    bit_exact = gap_vs_conj < 1e-6
    print(f"  Gap vs pre-conjugated: {gap_vs_conj:.2e} "
          f"({'BIT-EXACT' if bit_exact else 'MISMATCH'})")

    # --- Timing comparison ---
    n_warmup = 3
    n_trials = 20

    sample_ids, sample_mask, _ = next(iter(test_loader))
    sample_ids = sample_ids[:4].to(device)
    sample_mask = sample_mask[:4].to(device)

    backbone.eval()
    bb_conj.eval()
    online_bb.eval()

    def time_forward(model_fn, label):
        with torch.no_grad():
            for _ in range(n_warmup):
                model_fn()
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_trials):
                model_fn()
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.time() - t0) / n_trials * 1000
        print(f"    {label}: {elapsed:.2f} ms/batch")
        return elapsed

    print(f"\n  Timing ({n_trials} trials, batch=4):")
    t_base = time_forward(
        lambda: backbone(input_ids=sample_ids, attention_mask=sample_mask),
        "Base (no perm)    ",
    )
    t_conj = time_forward(
        lambda bb_conj=bb_conj: bb_conj(
            input_ids=sample_ids,
            attention_mask=sample_mask,
        ),
        "Pre-conjugated    ",
    )
    t_online = time_forward(
        lambda online_bb=online_bb: online_bb(
            input_ids=sample_ids,
            attention_mask=sample_mask,
        ),
        "Online permutation",
    )

    overhead_pct = (t_online - t_base) / t_base * 100

    print(f"\n  Online overhead vs base: {overhead_pct:+.1f}%")
    print(f"  Memory: 1 model ({model_size_mb(backbone, classifier):.1f} MB) "
          f"+ 1 permutation vector ({768 * 8 / 1024:.1f} KB)")

    del bb_conj, cl_conj, online_bb
    gc.collect()

    return {
        "base_acc": base_acc,
        "conjugated_acc": conj_acc,
        "online_acc": online_acc,
        "bit_exact": bit_exact,
        "gap_vs_conjugated": gap_vs_conj,
        "t_base_ms": round(t_base, 2),
        "t_conjugated_ms": round(t_conj, 2),
        "t_online_ms": round(t_online, 2),
        "overhead_pct": round(overhead_pct, 1),
    }


# ---------------------------------------------------------------------------
# EXPERIMENT D: CONCURRENT STRESS TEST
# ---------------------------------------------------------------------------

def experiment_d_concurrent(
    backbone, classifier, test_loader, device,
    r_values: list[int] | None = None,
) -> dict:
    """Batched serial: measure throughput serving R regimes from one model."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT D: Concurrent Stress Test (Batched Serial)")
    print("=" * 70)

    if r_values is None:
        r_values = [1, 2, 4, 8, 16, 32]

    sample_ids, sample_mask, _ = next(iter(test_loader))
    sample_ids = sample_ids[:1].to(device)
    sample_mask = sample_mask[:1].to(device)

    n_warmup = 3
    n_trials = 20

    backbone.eval()
    results = {"measurements": []}

    print("\n  Serving 1 request per regime, R regimes total.")
    print("  Each regime = pre-conjugated model copy (serial swap).")
    print(f"\n  {'R':>6} {'total_ms':>10} {'per_regime_ms':>14} {'throughput':>12}")
    print(f"  {'-'*6} {'-'*10} {'-'*14} {'-'*12}")

    for R in r_values:
        # Build R permuted copies
        copies = []
        for r in range(R):
            if r == 0:
                perm = np.arange(768)
            else:
                perm = np.random.RandomState(2000 + r).permutation(768)
            bb = copy.deepcopy(backbone)
            cl = copy.deepcopy(classifier)
            conjugate_residual_basis(bb, cl, perm, device)
            bb.eval()
            copies.append((bb, cl))

        with torch.no_grad():
            for _ in range(n_warmup):
                for bb, _classifier in copies:
                    bb(input_ids=sample_ids, attention_mask=sample_mask)

        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n_trials):
                for bb, _classifier in copies:
                    bb(input_ids=sample_ids, attention_mask=sample_mask)
        if device.type == "mps":
            torch.mps.synchronize()
        total_ms = (time.time() - t0) / n_trials * 1000
        per_regime_ms = total_ms / R
        throughput = R / (total_ms / 1000)

        m = {
            "R": R,
            "total_ms": round(total_ms, 2),
            "per_regime_ms": round(per_regime_ms, 2),
            "throughput_regimes_per_sec": round(throughput, 1),
        }
        results["measurements"].append(m)
        print(f"  {R:>6} {total_ms:>9.2f} {per_regime_ms:>13.2f} "
              f"{throughput:>11.1f}/s")

        for bb, cl in copies:
            del bb, cl
        del copies
        gc.collect()

    return results


# ---------------------------------------------------------------------------
# EXPERIMENT E: CROSS-REGIME ISOLATION
# ---------------------------------------------------------------------------

def experiment_e_isolation(
    backbone, classifier, test_loader, device,
    R: int = 4,
) -> dict:
    """Measure cross-regime CLS similarity and cross-regime accuracy."""
    print("\n" + "=" * 70)
    print(f"  EXPERIMENT E: Cross-Regime Isolation (R={R})")
    print("=" * 70)

    base_acc = evaluate(backbone, classifier, test_loader, device)

    # Build R conjugated copies
    copies = []
    perms = []
    for r in range(R):
        if r == 0:
            perm = np.arange(768)
        else:
            perm = np.random.RandomState(3000 + r).permutation(768)
        perms.append(perm)
        bb = copy.deepcopy(backbone)
        cl = copy.deepcopy(classifier)
        conjugate_residual_basis(bb, cl, perm, device)
        bb.eval()
        cl.eval()
        copies.append((bb, cl))

    # Same-regime accuracy (should match base)
    print(f"\n  Same-regime accuracy (expect = base {base_acc:.6f}):")
    same_accs = []
    for r, (bb, cl) in enumerate(copies):
        acc = evaluate(bb, cl, test_loader, device)
        same_accs.append(acc)
        print(f"    regime {r}: {acc:.6f} (gap={acc-base_acc:+.2e})")

    # Cross-regime: run regime r's backbone, read with regime s's classifier
    print("\n  Cross-regime accuracy (expect chance = 0.25):")
    cross_accs = []
    for r in range(R):
        for s in range(R):
            if r == s:
                continue
            bb_r, _ = copies[r]
            _, cl_s = copies[s]
            acc = evaluate(bb_r, cl_s, test_loader, device)
            cross_accs.append({"r": r, "s": s, "acc": acc})
            print(f"    backbone={r} classifier={s}: {acc:.6f}")

    # CLS cosine similarity between regimes on same inputs
    print("\n  CLS cosine similarity (same input, different regimes):")
    batch_ids, batch_mask, _ = next(iter(test_loader))
    batch_ids = batch_ids[:32].to(device)
    batch_mask = batch_mask[:32].to(device)

    cls_vectors = []
    with torch.no_grad():
        for _regime, (bb, _classifier) in enumerate(copies):
            cls = bb(input_ids=batch_ids, attention_mask=batch_mask
                     ).last_hidden_state[:, 0, :]
            cls_vectors.append(cls)

    cosine_sims = []
    for r in range(R):
        for s in range(r + 1, R):
            cos = F.cosine_similarity(cls_vectors[r], cls_vectors[s], dim=-1)
            mean_cos = cos.mean().item()
            cosine_sims.append({"r": r, "s": s, "mean_cosine": mean_cos})
            print(f"    regime {r} vs {s}: mean cosine = {mean_cos:.4f}")

    for bb, cl in copies:
        del bb, cl
    gc.collect()

    mean_cross_acc = np.mean([x["acc"] for x in cross_accs])
    print(f"\n  Mean cross-regime accuracy: {mean_cross_acc:.4f} "
          f"(chance = 0.2500)")
    print(f"  All cross-regime at chance: "
          f"{'YES' if all(x['acc'] < 0.35 for x in cross_accs) else 'NO'}")

    return {
        "base_acc": base_acc,
        "same_regime_accs": same_accs,
        "cross_regime": cross_accs,
        "mean_cross_acc": round(float(mean_cross_acc), 6),
        "cosine_sims": cosine_sims,
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orthogonal Superposition R-Scaling Study",
    )
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only", type=str, default="ABCDE",
                        help="Which experiments to run (e.g. 'AB' for A and B)")
    parser.add_argument("--max-r", type=int, default=1024,
                        help="Max R for sweep (default 1024)")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("  Orthogonal Superposition R-Scaling Study")
    print("=" * 70)
    print(f"  Device: {device}")
    print(f"  Experiments: {args.only}")
    print(f"  Max R: {args.max_r}")
    print()

    logging.basicConfig(level=logging.WARNING)
    t_start = time.time()

    # --- Load data and train base model ---
    print("  Loading AG News...")
    train_ds, test_ds = load_ag_news()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")

    print(f"\n  Training base model ({args.epochs} epochs, seed={args.seed})...")
    backbone, classifier, base_acc = train_base_model(
        train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, seed=args.seed,
    )
    print(f"  Base accuracy: {base_acc:.6f}")
    print(f"  Model size: {model_size_mb(backbone, classifier):.1f} MB")

    all_results = {
        "config": {
            "device": str(device),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "base_acc": base_acc,
            "model_size_mb": round(model_size_mb(backbone, classifier), 1),
            "d_model": 768,
            "n_heads": 12,
            "n_layers": 6,
        },
    }

    r_values = [v for v in [4, 8, 16, 32, 64, 128, 256, 512, 1024]
                if v <= args.max_r]

    if "A" in args.only.upper():
        all_results["A_r_sweep"] = experiment_a_r_sweep(
            backbone, classifier, test_loader, device, r_values,
        )

    if "B" in args.only.upper():
        all_results["B_ablation"] = experiment_b_ablation(
            backbone, classifier, test_loader, device,
        )

    if "C" in args.only.upper():
        all_results["C_online"] = experiment_c_online_permutation(
            backbone, classifier, test_loader, device,
        )

    if "D" in args.only.upper():
        d_r_values = [v for v in [1, 2, 4, 8, 16, 32] if v <= args.max_r]
        all_results["D_concurrent"] = experiment_d_concurrent(
            backbone, classifier, test_loader, device, d_r_values,
        )

    if "E" in args.only.upper():
        all_results["E_isolation"] = experiment_e_isolation(
            backbone, classifier, test_loader, device, R=4,
        )

    elapsed = time.time() - t_start

    # --- Save results ---
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"superposition_sweep_{ts}.json"
    all_results["total_elapsed_s"] = round(elapsed, 1)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Base accuracy:    {base_acc:.6f}")
    print(f"  Model size:       {model_size_mb(backbone, classifier):.1f} MB")
    print(f"  Total elapsed:    {elapsed:.1f}s")

    if "A_r_sweep" in all_results:
        sweeps = all_results["A_r_sweep"]["sweeps"]
        max_gap = max(s["max_abs_gap"] for s in sweeps)
        max_r = max(s["R"] for s in sweeps)
        print(f"  R-sweep:          max |gap| = {max_gap:.2e} across R up to {max_r}")
        print(f"                    {'CONFIRMED: gap is zero at all R' if max_gap < 1e-4 else 'UNEXPECTED: nonzero gap detected'}")

    if "B_ablation" in all_results:
        full = [a for a in all_results["B_ablation"]["ablations"]
                if a["name"] == "full_conjugation"][0]
        skipped = [a for a in all_results["B_ablation"]["ablations"]
                   if a["name"] != "full_conjugation"]
        n_collapse = sum(1 for a in skipped if abs(a["gap"]) > 0.01)
        print(f"  Ablation:         {n_collapse}/{len(skipped)} skips cause collapse")
        print(f"                    full conjugation: {full['accuracy']:.6f}")

    if "C_online" in all_results:
        c = all_results["C_online"]
        print(f"  Online perm:      overhead = {c['overhead_pct']:+.1f}%, "
              f"bit-exact = {c['bit_exact']}")

    if "E_isolation" in all_results:
        e = all_results["E_isolation"]
        print(f"  Isolation:        mean cross-regime acc = {e['mean_cross_acc']:.4f} "
              f"(chance = 0.25)")

    print()


if __name__ == "__main__":
    main()
