# Orthogonal Superposition: Comprehensive R-Scaling Experiment

## Core Thesis

Train ONE model. Place it into R orthogonal (permutation) slots via residual-stream conjugation. No per-regime training. The "gap" is zero by construction because each regime IS the trained model viewed in a permuted basis.

The question is: how far does R scale before something breaks, and what breaks first?

---

## Background: What We Already Know

The Orthogonal Superposition Theorem (Entry 74, commit `f7094f9`) established:

1. A transformer is exactly equivariant under a permutation P of the residual-stream basis, provided every residual-facing parameter is conjugated consistently.
2. Train once; conjugate by R permutations; every regime computes the identical function.
3. The symmetry group is S_d (permutations), NOT O(d) (full orthogonal group), because LayerNorm's per-coordinate affine and element-wise GELU only commute with permutations.
4. At R=4 on DistilBERT/AG News: max |gap| = 0.00e+00, bit-exact, zero retraining.
5. Disconfirmation: skipping LayerNorm conjugation collapses accuracy from 90.25% to 42.96%.

**What's new in this experiment**: The prior run only tested R=4. The user's intuition is that "unlimited" can't really be unlimited. We need to find the wall — or prove there isn't one.

---

## What We Are Testing

### Hypothesis 1: Serial R Is Truly Unlimited (Up To d!)

For serial/addressed use (one regime active per forward pass), R is bounded by the number of distinct permutations of d=768 elements, which is 768! ≈ 10^1871. Every permutation produces bit-identical accuracy because it is the same computation in renamed coordinates.

**Prediction**: Gap = 0.00e+00 at every R tested, including R=1000 and R=10000. No degradation whatsoever. Each regime is a bit-exact copy of the original model in a different coordinate system.

**What would falsify this**: Any nonzero gap at any R. This would indicate a bug in the conjugation, a numerical issue (e.g. floating-point non-commutativity of the permutation with some operation), or an architectural feature we missed.

### Hypothesis 2: The Conjugation Audit — Every Tendril Matters

The conjugation must touch every point where the residual stream enters or exits a parameter matrix. The prior experiment showed that skipping LayerNorm collapses to chance. But we haven't systematically tested every component.

**Prediction**: Omitting ANY single component from the conjugation produces a measurable accuracy collapse. The severity depends on how much information flows through that component.

### Hypothesis 3: On-The-Fly Permutation Is Cheap

If we store one model + R permutation vectors (768 integers each) instead of R deep copies of the model, what is the inference overhead? Applying a permutation is O(d) index gathering per layer — far cheaper than the O(d²) matmul.

**Prediction**: Overhead is negligible (< 5% latency increase). Memory is O(1) per regime (just a permutation vector), not O(model_size).

### Hypothesis 4: Concurrent Multi-Regime Inference Has a Real Ceiling

Simultaneous, collision-free superposition — all R active in one forward pass — requires R orthogonal subspaces in d dimensions. By the pigeonhole principle, R full-rank d-dim copies cannot coexist in d dims. Concurrent use is where the real R limit lives.

**Prediction**: Concurrent mode scales up to R = d / min_viable_dims. For DistilBERT (d=768), if each regime needs at least ~32 viable dims, R_max ≈ 24 for concurrent. For serial, no limit.

---

## Experiment Design

### Experiment A: R-Sweep (Serial Addressed Use)

**Protocol**:
1. Train ONE DistilBERT base model on AG News (4 epochs, seed=0, lr=2e-5).
2. For each R in {4, 8, 16, 32, 64, 128, 256, 512, 1024}:
   - Generate R random permutations of d=768 (regime 0 = identity for sanity).
   - For each regime: deep-copy the trained model, conjugate, evaluate, delete.
     (Sequential to avoid holding R copies in memory simultaneously.)
   - Record: accuracy per regime, max |gap| vs base, min accuracy, max accuracy, mean, std.
   - Record: peak memory, wall-clock time, per-regime conjugation + eval time.
3. Report timing and memory per R.

**What we extract**:
- Gap vs R curve (expect: flat zero line)
- Timing vs R curve (expect: linear in R — each regime is an independent eval)
- Memory and storage characterization (per-regime cost in MB)
- Any surprise — any R where gap ≠ 0
- The practical ceiling: where does wall-clock or memory make it uneconomical?

### Experiment B: Component Ablation (The Conjugation Audit)

**Protocol**:
For a fixed R=4, systematically omit ONE component from the conjugation and measure the damage. This produces a sensitivity map of the architecture.

| Ablation | What is skipped | Expected effect |
|----------|----------------|-----------------|
| Full conjugation (control) | Nothing | 0.00 gap |
| Skip embeddings | word_embeddings, position_embeddings columns | Severe collapse — input coordinates are wrong |
| Skip LayerNorm (known) | All LayerNorm gamma/beta | Collapse to ~43% (already measured) |
| Skip SA LayerNorm only | sa_layer_norm gamma/beta | Partial collapse |
| Skip output LayerNorm only | output_layer_norm gamma/beta | Partial collapse |
| Skip Q/K/V projections | q_lin, k_lin, v_lin columns | Collapse — attention computes wrong dot products |
| Skip out_lin | out_lin rows + bias | Collapse — attention output misrouted |
| Skip FFN lin1 | lin1 columns | Collapse — FFN input coordinates wrong |
| Skip FFN lin2 | lin2 rows + bias | Collapse — FFN output misrouted |
| Skip classifier | classifier columns | Collapse — correct CLS features, wrong readout |
| Skip ONLY classifier | Everything else conjugated | Collapse — tests whether classifier alone matters |

**What we extract**:
- Accuracy after each ablation (absolute and vs chance=25%)
- Ranking of components by sensitivity (which tendrils matter most)
- Publishable table showing that the conjugation is a whole-or-nothing proposition

### Experiment C: On-The-Fly Permutation Inference

**Protocol**:
Instead of deep-copying and pre-conjugating the model, apply the permutation at runtime:
1. Store ONE model (the trained base, unconjugated).
2. For each inference request with regime r:
   - Apply permutation P_r to the input embeddings.
   - At each layer, apply P_r to residual-stream inputs/outputs at the injection points (or equivalently, permute the activations, not the weights).
   - Apply P_r^{-1} to the CLS vector before the classifier (or permute classifier columns).
3. Measure: latency per sample (ms), throughput (samples/sec), memory footprint.
4. Compare to: (a) base model (no permutation), (b) pre-conjugated model.

**What we extract**:
- Latency overhead of on-the-fly permutation (expect < 5%)
- Memory savings: 1 model + R×768 ints vs R×model copies
- Bit-exact equivalence confirmation (on-the-fly == pre-conjugated)
- Publishable throughput multiplier: R regimes served from 1 model

### Experiment D: Concurrent Superposition Stress Test

**Protocol**:
Test what happens when multiple regimes try to share one forward pass. Two sub-experiments:

**D1 — Batched serial** (the practical approach):
- Batch R inputs from R different regimes.
- Run R forward passes (one per regime, swapping the active permutation).
- Measure total latency vs R × single-regime latency.
- This is the realistic deployment: batched scheduling with per-request regime addressing.

**D2 — True concurrent** (the theoretical limit):
- Attempt to run R=2, 4, 8 regimes through a single forward pass by partitioning the residual stream into R subspaces (d/R dims per regime, gated).
- This SHOULD show degradation because each regime gets only d/R = 768/R dimensions.
- Measure the accuracy gap as R increases.
- Compare to the serial/addressed baseline (which is always gap=0).

**What we extract**:
- The real concurrent R ceiling (where accuracy drops below acceptable threshold)
- The throughput advantage of batched serial vs naive sequential
- Publishable chart: serial R (unlimited, gap=0) vs concurrent R (limited, gap grows)

### Experiment E: Cross-Regime Isolation Under Superposition

**Protocol**:
Test whether one regime's outputs leak information about another regime's permutation.
1. Train and superpose at R=4.
2. For each regime pair (r, s): run regime r's inputs through regime s's permuted model. Record accuracy.
3. Run steganographic erasure: zero regime r's owned parameters (but what does "owned" mean when all dims are used? — this is the key question).
4. Measure: cross-regime accuracy (expect chance=25%), cosine similarity of CLS vectors between regimes on the same input.

**What we extract**:
- Cross-regime isolation metrics
- Whether superposition provides natural isolation (different permutations = different coordinate systems = outputs are uncorrelated)
- Steganographic erasure applicability (or non-applicability) to the superposition model

---

## What We Expect to Find (Honest Predictions)

| Finding | Confidence | Why |
|---------|-----------|-----|
| Serial R gap = 0 at all tested R | Very high | Mathematical theorem + R=4 confirmation |
| Gap = 0 holds even at R=1024 | Very high | Each permutation is independent; 768! >> 1024 |
| Every skipped component collapses accuracy | High | The conjugation is all-or-nothing by construction |
| On-the-fly permutation adds < 5% latency | Medium-high | Index gather is O(d) vs O(d²) for matmul |
| Concurrent R is limited to ~d/32 ≈ 24 | Medium | Depends on minimum viable dims per regime |
| Cross-regime CLS vectors are uncorrelated | High | Different permutations produce orthogonal-ish representations |

## What Would Be Surprising

- **Any nonzero gap in serial mode**: Would indicate a numerical or architectural issue we missed. Extremely unlikely but must be checked.
- **A component skip that DOESN'T collapse accuracy**: Would mean that component is redundant in the conjugation — interesting architectural finding.
- **On-the-fly permutation NOT being bit-exact**: Would indicate that applying permutation to activations differs from applying it to weights — this should be impossible but needs verification.
- **Concurrent mode degrading slower than d/R**: Would suggest the model is more robust to dimension reduction than expected (consistent with coerced superposition).

---

## Hardware and Timing

All experiments run locally (MPS or CPU). No GPU cluster needed — we are evaluating one pre-trained DistilBERT model under permutations, not training R models.

| Experiment | Estimated time | Memory |
|-----------|---------------|--------|
| A: R-sweep (9 R values, eval only) | ~30-45 min | Sequential: 1 copy at a time + permutation vector |
| B: Component ablation (11 ablations × R=4) | ~10 min | 4 model copies |
| C: On-the-fly inference benchmark | ~10 min | 1 model + permutation vectors |
| D: Concurrent stress test | ~15 min | Depends on R |
| E: Cross-regime isolation | ~10 min | 4 model copies |
| **Total** | **~75-90 min** | **Manageable on laptop** |

Note: For large R in Experiment A, we MUST use on-the-fly permutation (cannot deep-copy 10000 models). The experiment naturally motivates Experiment C.

---

## Data Archival Strategy

### What We Store (Durable Artifacts)

All experiment outputs are written to `experiments/results/` and committed to this repo. These are the durable record — the scripts may not be publishable, but the data must survive independently.

| Artifact | Format | Contents |
|----------|--------|----------|
| `superposition_sweep_YYYYMMDD_HHMMSS.json` | JSON | Full machine-readable results: every accuracy, gap, timing, memory measurement across all 5 experiments. Includes config (device, epochs, lr, seed, model size). |
| `full_run_log.txt` | Plain text | Complete console output — human-readable record of every line printed during the run. |
| `experiment-data-inventory.md` (in `docs/`) | Markdown | Updated inventory entry pointing to the JSON/log files with summary numbers. |

The JSON file is the source of truth. Tables in the paper are derived from it. If we re-run (different seed, different hardware), a new timestamped JSON is created alongside the old one — no overwrites.

### What We Publish

The experiment script (`orthogonal_superposition_sweep.py`), protocol,
hyperparameters, seeds, and raw result data are included for independent
inspection and reproduction. Executable experiment code is Apache-2.0; the
paper, this design document, figures, and designated research records are CC BY
4.0. See [`../LICENSES.md`](../LICENSES.md) for the authoritative path mapping.

A U.S. provisional patent application was filed before release for subject
matter related to portions of Schemen Gate. This notice adds no separate
restriction. Apache-2.0 Section 3 governs patent rights for the Apache-2.0
material; CC BY 4.0 does not license patent rights. Private filing identifiers,
unpublished claims, and prosecution strategy are not part of this repository.

### Directory Layout

```
experiments/
├── orthogonal-superposition-experiment.md   # This design doc
├── orthogonal_superposition_sweep.py        # Apache-2.0 experiment source
├── PLAN.md                                  # Public experiment and release plan
└── results/
    ├── superposition_sweep_YYYYMMDD_HHMMSS.json  # Machine-readable results
    └── full_run_log.txt                          # Console log
```

---

## Deliverables for the Paper

1. **Table**: R-sweep showing gap=0 from R=4 to R=256 with wall-clock times and storage costs.
2. **Table**: Component ablation sensitivity map (17 rows, accuracy + collapse magnitude).
3. **Figure**: Serial R (flat line at 0 gap) vs concurrent R (growing gap) — the visual proof that addressed use is free but concurrent use costs capacity.
4. **Throughput measurement**: Samples/sec for 1-model + on-the-fly permutation serving R regimes.
5. **The narrative**: "Train once, serve unlimited tenants. The limit is economic (memory, scheduling), not architectural (the model supports 768! ≈ 10^1871 distinct regime slots). Confirmed empirically to R=256 with zero accuracy loss."
