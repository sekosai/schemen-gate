# CDP Experiment Data Inventory

Submission-facing experiment implementations and machine-readable
artifacts live under `experiments/` in this repository. Historical
entries that have not been ported to the current Gate API remain identified as
legacy or outside the paper; they are not required
for the submission claims.

**Legend:** IN PAPER = used by `cdp.tex`. LOCAL PROTOCOL CHECK = executable
current-tree mechanism evidence, not a paper result. NOT IN PAPER = not used by
`cdp.tex` and says nothing by itself about artifact availability. OUTSIDE
CURRENT PAPER = historical catalog only. "Historical source (not shipped)"
means the named file is absent from this history-free release and does not
provide executable evidence for a release claim.

**Evidence exports:** fourteen current-library artifacts cited below are
documented public-safe exports of run records produced under a private
companion authorization harness. Measured values are unchanged; the
original-record digests and exact transformation rules are in the
“Public-safe evidence exports” section of `experiments/results/README.md`.

**Current submission scope (2026-08-21):** task evidence includes
formative post-encoder classification cost, strict intermediate-FFN
cotenancy, private adapter/expert lanes, a Cargo protocol smoke, a
checked-in storage smoke, exact extraction, local pre-MLP and atomic
whole-model authorization checks, and function-preserving
permutation placement. Current-library preflight-backed one-seed reruns now also
cover strict R=8 FFN cotenancy, frozen private lanes, matched public
co-training, R=8 causal generation, and complete-scope Cargo. A
publication-safe capacity-preserving synthetic classifier rerun is also
included. Numerical legacy DistilBERT wrong-key,
GPU-memory, and throughput claims are
withheld because their full artifacts are not checked in. The archived classification
numbers are not intermediate-FFN evidence. The new generative run establishes
placement/confinement but not utility parity. Post-pooling embedding,
KV-cache, hollow-regime deposition,
duplicated-attention-lane, and external leakage studies are archived
outside the paper.

---

## Companion Schemen-Gated Transformer Regime-Lane Study

**Status: PUBLISHED COMPANION STUDY; NOT IN THE ORIGINAL CDP PAPER**

The curated module at [`../gated-transformer-regime-lanes/`](../gated-transformer-regime-lanes/README.md)
evaluates complete per-Regime Q/K/V/O attention alternatives with separate
mutable residual, KV-cache, sequence, position, stopping, parameter-generation,
and training state. It retains the formal paper, a result-by-result correction
history, a claim ledger, selected figures, sealed toy results, and executable
toy controls.

Headline bounded results include 0/147,456 foreign-corpus answers, 992/992
fixed-shape own-lane intervention positives, 2,976/2,976 bit-exact off-lane
comparisons, and exact R8/R16 cached lifecycle controls. The module explicitly
withholds the tested switch-oriented SFT curriculum, universal cross-shape
bit-exactness, R32 deployment, naive multi-stream acceleration, and universal
or production-security claims. The complete 136-file historical dossier remains
outside the public critical path at the custody identifiers recorded in the
module's `EVIDENCE_ARCHIVE.md`.

---

## Series 1: Co-Trained Classification Cost (DistilBERT)

**Status: IN PAPER (formative post-encoder classification section)**

**Statistical status:** A new five-seed paired sweep is complete for
R={8,16,24,32,64,96,128}.  The table reports separate-minus-gated
accuracy gaps.  Student-t intervals were recomputed because the
archived script printed normal-approximation intervals with only five
pairs.

**Gate surface:** The formative runner masks DistilBERT's final
768-dimensional CLS vector after the encoder. It does not gate the
3,072-dimensional intermediate FFN activation. The public reproduction
script exposes this as `--surface legacy-post-encoder`; corrected FFN
evidence must use `--surface intermediate-ffn` and produce new
artifacts.

**Baseline sampling:** At each R, the gated model trains all R
regimes, but the separate-model control trains only four regimes. Each
paired seed statistic compares the mean of those four controls with
the corresponding four gated-regime accuracies.

| R | Gap (pp) | Student-t 95% CI (pp) | paired p-value | Bonferroni reject? |
|---|----------|------------------------|----------------|--------------------|
| 8 | 0.348 | [−0.229, 0.925] | 0.1692 | No |
| 16 | 0.268 | [−0.301, 0.837] | 0.2616 | No |
| 24 | 0.208 | [−0.324, 0.740] | 0.3392 | No |
| 32 | 0.267 | [−0.345, 0.879] | 0.2920 | No |
| 64 | 0.143 | [−0.041, 0.326] | 0.0967 | No |
| 96 | 0.376 | [0.045, 0.708] | 0.0345 | No |
| 128 | 0.272 | [−0.208, 0.752] | 0.1902 | No |

Interpretation: the observed performance costs are small in this
study. Exact empirical parity is neither expected nor established;
none of the seven tests crosses the Bonferroni threshold 0.0071.

**NOT IN PAPER**: R=24 and R=64 full rows; deployment ratios (VRAM/storage/throughput); overprovision live-install timing; epoch sweeps via `modal_extreme_r.py` (R=128 at 2/4/6/8 epochs).

**OPTIONAL FOLLOW-UP**: A paired frozen-vs-co-trained DistilBERT
comparison could characterize optimization dynamics. It is not
needed for the linear-algebraic support constraint.

**Source**: `experiments/reproduce_distilbert_classification.py`,
historical sources (not shipped) `high_r_benchmark.py` and `modal_series1.py`,
`experiments/results/series1_results_20260803_202919.json`

---

## Strict Intermediate-FFN Cotenancy (DistilBERT)

**Status: IN PAPER**

- Gates every 3,072-dimensional intermediate FFN activation before
  the down projection.
- Freezes attention, embeddings, normalization, residual paths, and
  other non-governed shared parameters.
- Uses regime-scoped Adam moments and private classifiers.
- Across R={1,2,4,8,16}, every frozen-shared, off-partition parameter,
  off-partition optimizer-moment, and inactive-classifier maximum
  delta is exactly zero.
- Mean owning accuracy declines from 91.28% at R=1 to 86.21% at R=16.

**Source**: `experiments/modal_dense_ffn_cotenancy.py`,
`experiments/results/dense_ffn_cotenancy_*.json`, and
`experiments/results/transformer_cotenancy_summary.json`

**Current-library corroboration:** seed 42 at R=8 reaches 87.756% owning
accuracy with all exact separation deltas zero and all malformed execution
authorities rejected before model execution. Source artifact:
`experiments/results/dense_ffn_cotenancy_20260821T061623_781627Z.json`.

---

## Gate-Placement Negative Controls

**Status: LOCAL PROTOCOL CHECK**

The CPU cotenancy harness includes two deliberately invalid paths. A
LayerNorm inserted before a disjoint target gate produces a nonzero
target coordinate (0.12598 in the retained smoke), and an ungated residual
bypass carries the excluded source unchanged (maximum 3.0). The correctly
placed disjoint gate remains exactly zero. These checks ensure the harness
can detect the placement and path-closure failures excluded by the theorem.

**Source**: `experiments/local_transformer_cotenancy_suite.py` and
`experiments/results/archive/smoke/transformer_cotenancy_local_20260812_100546.json`

---

## Pre-MLP FFN and Whole-Model Gate Checks

**Status: LOCAL PROTOCOL CHECKS**

The pre-MLP check gates a 32-dimensional FFN input before a
32-to-64 first projection at R=4. The full gated and physically
extracted projections match exactly in this run; inactive input and
aligned first-projection-row gradients are exactly zero. This is the
narrow pre-MLP surface, not evidence for hidden/down-projection
partitioning or task utility.

The whole-model check binds tenant, sub-id, model digest, and operation
before invoking an otherwise unchanged function. Random, wrong-tenant,
wrong-sub-id, wrong-model, and wrong-operation keys are all rejected
with zero model calls. The owning path makes one call and has zero
output difference from direct execution. This is an operational
authorization check, not an internal model-partition theorem.

**Source**: `experiments/local_transformer_cotenancy_suite.py`,
`experiments/execution_preflight.py`, and
`experiments/results/archive/smoke/transformer_cotenancy_local_20260812_100546.json`

---

## Private Transformer Lanes

**Status: IN PAPER**

The frozen shared DistilBERT backbone is paired with either complete
private residual adapters or complete private FFN experts. Mean owning
accuracy is 90.47% and 90.09%, respectively; shared-backbone and
inactive-lane deltas are exactly zero. Shared activations and serving
state remain inside the declared runtime trust boundary.

A current-library seed-42 rerun reports 90.66% adapter and 90.15% expert
owning accuracy, 3.11% and 3.28% wrong-key accuracy, exact zero shared/inactive
lane changes, and zero unauthorized execution calls.

**Source**: `experiments/modal_private_transformer_lanes.py` and
`experiments/results/private_transformer_lanes_*.json`

---

## Public Gate-Adaptation Factorial

**Status: ONE FULL CURRENT-LIBRARY SEED COMPLETE**

The prior three-seed artifact compares a teacher checkpoint with a
student that receives an additional all-mask distillation epoch. Its
1.60 percentage-point difference is preliminary pipeline evidence,
not a matched causal estimate. The successor runner adds no-extra,
ungated hard-label, all-mask hard-label, and all-mask distillation
conditions. The three adapted arms match examples, batch order,
dropout seed schedule, optimizer steps, teacher evaluations, and
student model-pass counts. A manual one-seed smoke is available; the full
command independently enforces the same ordering by completing a reduced-data
pilot before dispatching full seeds. The full seed-42 result at R=8 reports
+0.075 pp for ordinary extra training, +1.331 pp for mask awareness,
+0.031 pp for distillation given masks, and +1.438 pp for the complete
pipeline. All exact separation and execution rejection checks pass. The reduced
pilot is negative and remains a smoke/debug result, not a utility estimate.

**Source**: `experiments/modal_public_gate_adaptation_factorial.py`;
preliminary artifact
`experiments/results/public_gate_distillation_20260806T193105_795325Z.json`,
and current matched artifact
`experiments/results/public_gate_adaptation_factorial_20260821T063703_914999Z.json`

---

## Cargo Authorization

**Status: COMPLETE-SCOPE CURRENT-LIBRARY GPU RUN COMPLETE**

The completed two-tenant artifact reports 4/4 owning canaries, six
wrong tenant/regime/random attempts rejected, and zero unauthorized
model calls, but it predates the canonical sub-id requirement and is
mechanism evidence only. The self-contained Gate successor additionally binds
sub-id, model digest, operation, and policy version and adds
wrong-sub-id, wrong-operation, and same-regime wrong-partition negative
controls. The successor runner also applies a separate whole-model
authorization gate before generator invocation. The current artifact returns
4/4 exact owning answers and rejects 23 wrong-scope attempts with zero
unauthorized model calls. Production IdP/lockbox code is not part of the public
research harness.

**Source**: `experiments/modal_cargo_transformer_authorization.py`,
`experiments/execution_preflight.py`, and
`experiments/results/cargo_transformer_20260821T063827_829714Z.json`

---

## Addressed-Use Equivariance

**Status: CURRENT-LIBRARY CONTROL IN TRANSFORMER PAPER**

A clean preflight-backed DistilBERT permutation-conjugation rerun evaluates all
8 addressed regimes at R=8 and all 128 at R=128. Baseline and every regime
have identical 91.421% accuracy. Maximum fp32 logit drift is
$1.06\times10^{-5}$ and changes no prediction. This is a whole-model
equivariance result under addressed use, not simultaneous sparse cotenancy or
tenant-private state.

**Source**: `experiments/modal_orthogonal_superposition.py` and
`experiments/results/orthogonal_superposition_20260821T060951_717885Z_reanalysis.json`

---

## TinyLlama Intermediate-FFN Generation

**Status: CURRENT-LIBRARY ONE-SEED PLACEMENT/CONFINEMENT RESULT**

At R=8 the gate is attached to each 5,632-dimensional expanded SwiGLU product
immediately before `down_proj`, leaving 704 coordinates per regime. The exact
probe records zero inactive gradients and parameter changes, matched model
initializations, and zero unauthorized execution calls. Utility does not match
the ungated control: gated mean token loss is 5.442 versus 4.230; owning versus
wrong-key canary loss is 3.013 versus 3.452, with 0/16 exact owning
generations. The result validates the boundary, not generative parity.

**Source**: `experiments/modal_generative_intermediate.py` and
`experiments/results/generative_intermediate_combined_20260821T071152_101574Z.json`

---

## Series 1 extension: DistilBERT R=4 Parity (5-seed)

**Status: NOT IN PAPER**

- 5-seed paired t-test, **p=0.74** (gated vs separate)
- 95% confidence intervals computed
- Frozen-backbone gap quantified at R=4

**Historical source (not shipped):** `accuracy_parity_statistical.py`

---

## TinyLlama 1.1B Co-Trained

**Status: OUTSIDE CURRENT PAPER**

| Metric | Value |
|--------|-------|
| Disk savings | 4.00× |
| GPU memory savings | 3.99× |
| Throughput multiplier | 3.90× |
| Parity p-value (5-seed) | 0.21 |
| R | 4 |
| Hidden dim | 2,048 |

**Historical sources (not shipped):** `llama_benchmark.py`, `poc/README.md`

---

## Series 4: Frozen Mistral-7B Backbone

**Status: OUTSIDE CURRENT PAPER**

| R | Dims/regime | Accuracy | Gap vs R=1 |
|---|-------------|----------|------------|
| 1 | 4,096 | 91.32% | baseline |
| 4 | 1,024 | 87.62% | −3.70% |
| 8 | 512 | 82.45% | −8.87% |
| 64 | 64 | 77.99% | −13.33% |
| 512 | 8 | 68.74% | −22.58% |

**NOT IN PAPER**: Full R=16, 32, 128, 256 sweep; per-seed standard deviations; epoch-by-epoch convergence history.

**Historical sources (not shipped):** `mistral_benchmark.py`, `modal_mistral.py`

---

## Co-Trained Mistral 7B (Frozen Gaps Closed by Co-Training)

**Status: NOT IN PAPER**

| R | Frozen gap vs base (90.1%) | Co-trained improvement |
|---|---------------------------|----------------------|
| 4 | — | +2.49 pp |
| 8 | — | +3.32 pp |
| 16 | — | +4.14 pp |

Treat this as a preliminary within-Mistral pilot, not as a bridge to
DistilBERT.  A paper-grade claim requires matched frozen and co-trained
Mistral conditions, paired seeds, and confidence intervals.

**Historical sources (not shipped):** `schemen_cotrain_7b.py`,
`modal_cotrain_7b.py`

---

## Series 5b: Union Test / Commingling Premium

**Status: OUTSIDE CURRENT PAPER**

### Architecture ladder (NOT IN PAPER as a table)

| Architecture | Accuracy | Note |
|-------------|----------|------|
| Linear probe | 90.24% | Frozen features only |
| Best MLP (all 4,096 dims) | 91.89% | Frozen ceiling |
| LoRA rank-4 Q+V | 94.09% | Attention commingling |
| **Commingling premium** | **2.20%** | 94.09% − 91.89% |

### Adapter-gated Schemen (NOT IN PAPER)

| R | Accuracy | Note |
|---|----------|------|
| 4 | 91.37% | Isolation cost: 0.52 pp vs MLP ceiling |
| 8 | 91.55% | Isolation cost: 0.34 pp vs MLP ceiling |
| 16 | 91.42% | Isolation cost: 0.47 pp vs MLP ceiling |

The isolation cost is ~0.34–0.52 pp, NOT the 2.20 pp commingling premium.

**Historical sources (not shipped):** `modal_union_test.py`,
`docs/sales-deck.md`

---

## Series 5c: Cross-Tenant Directional Flow

**Status: OUTSIDE CURRENT PAPER**

**NOT IN PAPER**: Per-variant accuracy table (linear, post-gate MLP, lowrank_r64, etc.) showing which adapter architecture closes how much of the 2.20% gap while maintaining cross-regime at chance.

**Historical sources (not shipped):** `adapter_gate_benchmark.py`,
`modal_adapter_gate.py`, `docs/training-guide.md`

---

## Series 5d: Medical Membership Inference Attack

**Status: OUTSIDE CURRENT PAPER**

| Method | AUC | CDP control |
|--------|-----|-------------|
| LoRA (LOSS+Pre) | 0.609 | 0.499 |
| Schemen gated | 0.499 | structural zero |

MIA success threshold: mean_auc_loss > 0.55.

**NOT IN PAPER**: Per-seed AUC breakdown; Ratio and Min-K%+Pre AUCs; top identified member/non-member examples. Literature comparison: LoRA-Leak reports **0.775 AUC** (different model/dataset).

**Historical sources (not shipped):** `adversarial_mia_medical.py`,
`docs/training-guide.md`

---

## Series 5e: Token Divergence

**Status: OUTSIDE CURRENT PAPER**

**NOT IN PAPER**: Category breakdown (medical/legal/neutral prompts), mean KL divergence (nats/token), medical marker vocabulary counts on non-medical prompts, sample leak transcripts.

**Historical source (not shipped):** `cross_tenant_token_influence.py`

---

## Series 5f: Knowledge Exfiltration

**Status: OUTSIDE CURRENT PAPER**

Config: lora_rank=16, lora_epochs=5, 30 facts × 3 phrasings = 90 training texts × 15 repeats.

**NOT IN PAPER**: Per-category hit rates (legal/financial/protocol); full captured-flag Q/A pairs; Tenant B self-test accuracy.

**Historical source (not shipped):** `cross_tenant_knowledge_exfil.py`

---

## LoRA vs Schemen Head-to-Head (7B)

**Status: NOT IN PAPER (cited via bleed paper only)**

QLoRA rank sweep {4,8,16,32,64} all-layers vs Q+V-only vs Schemen gate R={4,8,16} on same cached Mistral features, 3 seeds. Direct competitive comparison.

**Historical sources (not shipped):** `lora_comparison_7b.py`,
`modal_lora_comparison.py`

---

## LoRA vs Schemen (DistilBERT)

**Status: NOT IN PAPER**

Schemen Tier-2 co-train vs LoRA adapters (frozen backbone) vs Tier-1 post-hoc install. Parity threshold: |gap| < 0.005.

**Historical source (not shipped):** `lora_vs_schemen_poc.py`

---

## Compressed Inference Exact Equivalence

**Status: IN PAPER; local evidence archived**

- Algebraically equivalent outputs: gated full matrix vs extracted d/R submatrix
- 10 seeds × fp32/fp16, 80 single-regime and 80 compound-mask checks
- the canonical d=768 fp32 run is bit-identical, but this is not generalized
  beyond that kernel/reduction shape; fp16 maximum absolute difference is
  **0.015625**
- all comparisons satisfy the recorded conservative forward-error bound
- Parameter compression ratio: **4.00×** (R=4)
- Ties directly to Lean proof `compressed_eq_gated`

**Current source and artifact:** `experiments/local_exact_extraction.py`,
`experiments/results/local_exact_extraction_20260820_215652.json`.
**Historical source (not shipped):** `compressed_inference_poc.py`.

---

## Compound Inference (Tiered Licensing)

**Status: NOT IN PAPER**

- Accuracy with 1 gate open: **~34%** (chance on 4-class)
- Accuracy with all 4 gates open: **~92%** (full model)
- Monotonic improvement as gates are added
- Full combination matrix: 15 subsets of 4 regimes
- Locked-class non-zero accuracy is not a breach (explained)

**Historical sources (not shipped):** `compound_inference_poc.py`,
`poc/README.md`

---

## R=128 Synthetic Scalability

**Status: IN CORE PAPER AS A FORMATIVE CAPACITY-SCALING RESULT**

- 128-regime gated MLP, 256 observations, 1,280 hidden dims
- Per-regime accuracy: **256/256** correct (100%)
- Wrong-key/foreign-observation probes: **0/32,512** expected-answer hits
- Combined all-regime mask at the matched 2,000-step budget: **249/256**
  (97.27%); the artifact is retained as a failed historical-threshold check
- A separately labeled 3,000-step sensitivity reaches **252/256** (98.44%),
  showing that this union metric is optimization-budget sensitive
- Selected-support and explicit dense-mask logits: maximum difference **0**

The core paper treats this as a synthetic memorization/scalability check, not
held-out utility or a privacy theorem. The public runner holds per-regime width
fixed at q=10 and scales total width as d=qR. Production lockbox code and the
legacy 7-level hierarchy test are intentionally not part of this artifact.

**Source**: `experiments/capacity_preserving_wide_classifier.py`,
`experiments/results/capacity_preserving_wide_20260820_215713.json`,
`experiments/results/capacity_preserving_wide_20260820_215848.json`, and
`experiments/results/capacity_preserving_wide_20260820_225331.json`.
**Historical sources (not shipped):** `wide_regime.py` and
`tests/test_128_regime.py`.

---

## Embedding Benchmark (bge-base-en-v1.5)

**Status: OUTSIDE CURRENT PAPER**

- nDCG@10 was measured for gated, separate, and cross-key retrieval at
  R={4,8,16,32,64,96}.
- All three conditions are uniformly weak (approximately
  0.0065--0.0104 nDCG@10).
- Similar gated and separate scores therefore do not establish useful
  retrieval quality, and low cross-key scores are not diagnostic.

The result is retained as an archived negative utility test. It does
not instantiate an intermediate FFN gate and is not submission
evidence.

**Checked-in artifact:**
`experiments/results/archive/exploratory/embedding_results_20260803_184852.json`.
**Historical sources (not shipped):** `embedding_benchmark.py`,
`modal_embedding.py`.

---

## Vision Benchmark (ViT)

**Status: NOT IN PAPER**

- Multi-regime ViT-style classification
- Parity, cross-regime isolation, VRAM measurements
- Targets Cognex / Landing AI use cases

Non-NLP modality validation.

**Historical source (not shipped):** `vision_benchmark.py`

---

## Deployment Savings (DistilBERT)

**Status: STORAGE SMOKE IN PAPER; FULL GPU RESULTS NEED RERUN**

| Metric | Value |
|--------|-------|
| Disk savings | 4.0× (checked-in CPU smoke) |
| GPU memory savings | historical 3.6×; withheld pending archived rerun |
| Throughput multiplier | historical 2.69×; withheld pending archived rerun |
| R=4 parity p-value | historical 0.74; outside current paper |

The public runner records storage, live accelerator allocation, and
routed/multiplexed throughput in JSON. Hardware-matched full results
must be collected before restoring numerical GPU or throughput claims.

**Current source:** `experiments/reproduce_distilbert_deployment.py`.
**Historical sources (not shipped):** `deployment_savings_benchmark.py`,
`poc/README.md`.

---

## Pentest / Weight Inspection Attacks

**Status: NOT IN PAPER**

- 7 weight-inspection attacks on small gated MLP
- Partition recovery accuracy and NMI
- Vulnerability thresholds: VULNERABLE >80%, PARTIAL >35%
- Cross-regime: LEAKING if > chance + 15%
- Historically motivated camouflage or TEE work. The camouflage direction was
  retired because statistical noise matching is not a white-box proof;
  cotraining opacity remains Observed-tier hardening only. This catalog entry
  is not evidence for the current release's custody claims; see the current
  [security claim boundary](../../../docs/SECURITY_CLAIMS.md).

Honest security boundary for uncamouflaged weights.

**Historical source (not shipped):** `pentest_poc.py`
