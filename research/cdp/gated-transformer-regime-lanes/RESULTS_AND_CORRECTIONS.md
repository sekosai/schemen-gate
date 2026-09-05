# Gated Transformer regime lanes: results, failures, and corrections

**Status:** publication summary<br>
**Evidence cutoff:** 2026-09-05<br>
**Full frozen record:** `codex/gated-head-evidence-archive` at
`4e8e0962afc920e15de1731c626384554dd6534c`

## Executive result

The architecture result is positive and the training result is mixed.

A Schemen Gate can authorize a complete regime-specific Transformer attention
and decoder-state lane before private access while multiple regimes reuse a
shared immutable Qwen backbone. In the evaluated fixed-shape tests, another
regime's corpus, attention parameters, residual/FFN row, KV cache, sequence, or
lifecycle event did not change the target lane. Physical R8 and R16 rosters
passed their complete cached-decoder lifecycle controls.

The current switch-oriented SFT curriculum is not promoted. It substantially
improved topic switching, but one independently trained state failed lookup
acquisition and the treatment missed its locked all-state utility gate. That is
a training result, not evidence that resident foreign regimes contaminated the
lane.

The performance result is also mixed. Complete attention banking saves model
artifact bytes relative to full model duplication. Naive concurrent eager
loops were slower in every matched control. Combining regime rows into one
forward improved short and medium prefills; the current private-attention loop
lost that advantage at 1,024 tokens. R8 is the integration candidate, R16 is a
stress tier, and R32 is a resource/backend boundary.

## Final claim table

| Question | Final result | Disposition |
|---|---|---|
| Can authorization occur before complete private attention access? | Yes, on the tested Qwen 0.5B and 4B implementations; malformed authority stopped before callback or lookup. | Promoted as a bounded architecture result. |
| Does `R` mean native heads are partitioned? | No. Every regime owns a complete Q/K/V/O attention module at each block, retaining all native heads and native RoPE. | Definition fixed. |
| Can immutable embedding/FFN/output weights be shared? | Yes under the tested row-owned execution: fixed-shape foreign interventions left target logits exact. | Promoted only for tested immutable operators and shapes. |
| Are residuals, cache, positions, masks, sequence, stopping state, and parameter generation private? | Yes in the implemented lane contract and tested lifecycle probes. | Promoted as bounded functional separation. |
| Can each regime be SFT-trained independently? | Yes. Successful standalone controls were preserved in owning lanes; optimizer resume and transfer checks passed. | Architecture promoted; current switch curriculum withheld. |
| Did A learn B, C, or D? | No foreign mapping was observed: final confirmation produced 0 foreign answers in 147,456 wrong-corpus opportunities. | Finite negative result, not a universal secrecy proof. |
| Are all execution shapes bit-identical? | No. Ordinary and gated Qwen both showed batch/padding/backend numerical differences. | Explicitly rejected. |
| Is simultaneous dispatch automatically faster? | No. Naive eager multi-stream dispatch was adverse in all controls. | Rejected. |
| Can batched rows remain separate? | Yes in equal-length tested prefills; D-only mutation changed D while A/B/C stayed exact. | Bounded prefill result. |
| Is R32 a deployment target? | No. Persistent inventory was about 127 GB and automatic SDPA was nondeterministic in one environment. | Limit evidence only. |
| Is this production or security isolation? | No. Multi-process hostility, side channels, allocator erasure, and production tenancy were not established. | Explicitly withheld. |

## Experimental lineage

### 1. Toy topology and Gate placement

Two executable standard-library models established the intended contract before
using a pretrained language model. They tested authority-first candidate
restriction, complete attention ownership, request-local residual/cache state,
explicit join authority, RoPE cancellation controls, mutation isolation, and
receipt integrity. These toys passed, but they were construction proofs rather
than evidence about useful language-model behavior.

Archive sources: `toy_gated_delta_head.py`,
`toy_multi_regime_transformer.py`, their tests, and `results/`.

### 2. Qwen2.5-0.5B complete-lane pilot

The first Modal smoke failed because Transformers 5.9 changed the RoPE-theta
configuration representation. That was an API/configuration failure before a
scientific result. The runner was corrected without changing the architectural
hypothesis. The passing pilot reached 16/16 owning exact, zero owner-corpus
answers under the wrong key, and 16/16 selected-lane answers under that key.
Malformed authority produced zero model callbacks.

The initial unseen-wording parity task then failed to generalize: seed 1947
reached 35/64 despite collapsed training loss. A semantic wording design
improved the three seeds to 57/64, 59/64, and 58/64. One seed missed the frozen
90% utility bar by one decision; two passed. The corrected controlled study
added fixed-route, swapped-route, and direct-reference arms. Correct routing
produced 57/64, 59/64, and 58/64; the fixed route was 32/64 and swapped routes
were 7/64, 5/64, and 6/64. This established that route selection mattered, but
the preliminary miss remained in the record.

Archive sources: `QWEN_0_5B_COMPLETE_REGIME_LANES_2026-09-03.md`,
`QWEN_0_5B_UNSEEN_SEMANTIC_REGIME_STUDY_2026-09-03.md`, and
`QWEN_0_5B_CONTROLLED_REGIME_LANES_2026-09-03.md`.

### 3. Exact positioning, layerwise propagation, and concurrency

The Qwen 0.5B positioning study compared ordinary Qwen with gated R1, R2, and
all R4 routes over 38,895,616 full-vocabulary logits. All unmutated comparisons
were bit exact. Mutating inactive regime 3 changed selected regime 0 by zero;
the same mutation changed selected regime 3 and changed a no-Gate mean-mixed
control, which flipped 11/256 greedy decisions.

The layerwise study extended the comparison across attention writes, residuals,
FFN writes, final normalization, and logits: 155,582,464 final-logit scalars and
1,763,328,000 captured activation scalars were exact. The off-diagonal mutation
matrix was all zero, while all diagonal and no-Gate detector controls changed.

Four concurrent cached decoders added 5,469,696 exact sequential/concurrent
logit comparisons, 864/864 diagonal accesses, and pairwise-disjoint cache and
residual storage. A regime-3-only graft changed five of its eight continuation
tokens and none in regimes 0–2.

Archive sources: `QWEN_HEAD_REGIME_POSITIONING_RESULTS_2026-09-03.md`,
`QWEN_LAYERWISE_BANK_EQUIVALENCE_RESULTS_2026-09-03.md`, and the concurrent-lane
protocol/results in the archived README and source ledger.

### 4. Early attention-bank SFT and semantic controls

The first compact attention-bank SFT was deliberately a mechanism assay. All
four regimes learned 8/8 constant-token assignments; shared non-attention
parameters stayed exact, and simultaneous inference preserved sequential
logits. This showed addressable independent updates, not semantic
generalization.

The repeated-SFT study then exposed a utility limit. Correct routing averaged
macro-F1 0.5117, while cyclic-wrong was 0.0088 and frozen was zero, but only
13/20 seed-by-regime cells beat every control. The mechanism passed; the
semantic study was not promoted.

The preregistered label-crossover correction removed a confounded physical-
regime/label allocation. Correct R4 macro-F1 was 0.5669 versus 0.0153
cyclic-wrong and 0.1988 no-Gate mean. All 80 matched R1/R4 complete-logit
tensors were exact. The pooled capability endpoints passed, while the stricter
per-rotation stress rule still retained individual non-promotions.

Archive sources: `QWEN_ATTENTION_BANK_SFT_RESULTS_2026-09-03.md`,
`QWEN_REPEATED_SFT_CONTROL_STUDY_2026-09-03.md`, and
`QWEN_LABEL_CROSSOVER_CONFIRMATORY_RESULTS_2026-09-03.md`.

### 5. Trained-state export, reopen, and live state

Fresh-task reopen preserved exact serialized trained state. Matched trained
R1/R4 execution was exact across 77,791,232 final logits and more than 1.1
billion captured activation scalars. Sixteen four-lane concurrent trials also
preserved sequential logits and positions, with pairwise-disjoint live K/V and
residual storage and 27,888 diagonal accesses.

One development smoke was correctly marked not promoted because nominally
separate workers shared the same task identity. The implementation was amended
to record and require distinct non-null task IDs; the next smoke and final study
passed. This is an example of infrastructure identity being repaired rather
than omitted.

Archive source: `QWEN_TRAINED_REOPEN_CONCURRENCY_RESULTS_2026-09-03.md`.

### 6. Substrate receiver inside the lane bank

An exact all-layer pre-RoPE K/native-V receiver was installed inside a private
R4 bank. Selected R1/R4 and the literal-prefix oracle were bit exact; wrong,
no-memory, and no-Gate controls contained the expected value in 0/32 cases.
The behavioral result was qualified: the expected value appeared in 25/32
answers, but strict exact match was 0/32 and precision was low. This passed the
mechanistic receiver claim and failed the clean answer-formatting claim. It was
not relabeled as the sibling Substrate reader's stronger result.

Archive source: `SUBSTRATE_RECEIVER_BANK_RESULTS_2026-09-03.md`.

### 7. Qwen3-4B replication

The complete attention bank replicated at the requested larger backbone.
Matched R1/R2 covered 19,447,808 exact final-logit scalars and R1/R4 covered
38,895,616. Captured activation comparisons and access maps were exact.
Off-diagonal mutation effects were zero, with diagonal and no-Gate positives.
This promoted the inference topology at one pinned Qwen3-4B revision, not 4B
SFT or production tenancy.

Archive source: `QWEN4B_ATTENTION_BANK_REPLICATION_RESULTS_2026-09-03.md`.

### 8. Duplex artifact, decoder, and throughput studies

The R2 artifact study compared one ordinary shared model, one shared backbone
with two complete attention banks, and two full models. Across all three
topologies, 87,515,136 serial/concurrent logits were exact; bank versus ordinary
added 29,171,712 exact logits. The reopened attention delta produced a
9,932,460,808-byte composite versus a 16,089,964,000-byte two-checkpoint
ceiling, saving 38.27%. Peak allocated GPU memory was 31.68% lower than the
two-model control in that ordered process.

Every naive two-stream eager throughput interval was adverse—including the
ordinary and two-full-model controls. Therefore the slowdown was attributed to
the scheduling strategy, not to regime coexistence. The 256-token duplex stress
then extended exact comparison coverage to 933,494,784 scalars and passed cache,
position, route, and authority gates. One attempt completed computation but
failed result construction because of a receipt-key bug; it remained excluded
and visible, and the corrected unchanged workload passed.

Archive sources: `QWEN4B_DUPLEX_THROUGHPUT_RESULTS_2026-09-04.md` and
`QWEN4B_DUPLEX_DECODER_STRESS_RESULTS_2026-09-04.md`.

### 9. Row batching and the first grouped-lane failure

Four independently keyed regimes were placed into nonoverlapping rows of one
residual tensor. At every captured node, A/B/C/D shared a backing allocation
without overlapping element ranges. A D-only layer-4 mutation changed D by
1.15625 and left A/B/C bit exact. Prefill improved 1.339× at 64 tokens and
1.307× at 256, but was 0.993× at 1,024 because private attention was still
called serially per row.

The confirmatory protocol remained not promoted because an absolute/relative
numerical threshold frozen after a short pilot was exceeded at 1,024 tokens.
The ordinary native serial-to-batch control exceeded the same threshold, and
the gated row batch matched ordinary native batch exactly at that length. The
failure therefore bounded cross-shape equivalence; it did not show foreign
semantic influence.

An initial grouped-lane SFT study then produced zero held-out F1 for owning R4,
matched R1, wrong-key, and frozen arms. Reloaded artifacts also failed training
wording. That study could not answer whether coexistence preserved learning,
because its positive standalone learning control did not exist. It remained a
negative training experiment rather than being interpreted as an architectural
failure.

Archive sources: `QWEN4B_R4_ROW_BATCHED_FORWARD_RESULTS_2026-09-04.md`,
`GROUPED_LANES_RESULTS_2026-09-04.md`, and the grouped artifact diagnostic.

### 10. SFT repair and failure mechanism

The repair used FP32 master weights and optimizer moments, four private blocks
at depths 4/12/20/28 plus adjacent norms, broader training wording, and a
verified standalone-learning gate. All six physical sessions and all four
owning lanes reached 96/96 held-out exact; all 12 wrong-corpus cells and frozen
controls scored zero. Repeated same-corpus sessions agreed on every answer but
had maximum full-vocabulary logit differences from 17.375 to 34.5, establishing
that seed variation can be large without any foreign lane.

Across 992 fixed-shape causal interventions, every own-lane detector changed
and all 2,976 other-lane full-vocabulary comparisons were exact. Foreign-history
decoder counterfactuals preserved 2,056/2,056 target comparisons. Deliberately
broken joins and cache aliases triggered their detectors or failed before
lookup. A resumed B update reproduced physical parameters exactly and left
A/C/D logits, versions, and gradients unchanged.

The separate sequence-SFT diagnosis showed that five nonexact generations were
within-lane first-token errors. A did not learn B/C/D; it chose the wrong member
of A's own label set, after which autoregressive decoding followed that token.
Teacher-forced correct-label intervention recovered the remainder. This
corrected the interpretation without erasing the five failures.

Archive sources: `SFT_REPAIR_INFLUENCE_RESULTS_2026-09-04.md` and
`QWEN_SEQUENCE_SFT_FAILURE_MECHANISM_RESULTS_2026-09-04.md`.

### 11. Replicated scale-up and R32 boundary

The scale-up trained 24 states across two corpus draws, three order seeds, and
four regimes. Twenty-three of 24 physical/R1/R4 states exceeded 95% held-out
exact; one D checkpoint failed acquisition at approximately 15–16% in every
form. All 72 wrong-corpus cells scored zero F1. Six resumed updates and 1,440
off-lane fixed-shape comparisons passed. Dialogue review found five matched
physical/R4 state pairs that repeated a private label after a topic switch.
Their matching identities in standalone and R4 execution locate the problem in
the trained endpoints, not resident foreign regimes.

Physical R1–R32 materialization preserved solo-A logits and fixed-shape
interventions. R32's persistent tensor inventory was 126,984,060,928 bytes.
Automatic SDPA failed the long-decoder exactness gate. Follow-up identical-input
and ungated controls reproduced nondeterminism, including at the isolated SDPA
operator. Fixed math SDPA and eager passed 7,967/7,967 target comparisons.
Automatic SDPA remained excluded; the failed run was not overwritten.

Optional fixed-shape padding treated validity masks—not an EOS-shaped token—as
authority for real positions, retained only real cache history, and gathered
the last real-token logits. The bounded test passed 192 foreign-length, 64
left/right encoding, and 576 cached target comparisons. This does not establish
arbitrary-shape invariance.

An exploratory new-seed D recovery restored about 99% lookup and left A/B/C
exact, but retained conversation-quality problems. It did not replace the
failed D state in the confirmatory denominator.

Archive sources: `SCALEUP_RESULTS_2026-09-04.md`,
`R32_SDPA_DIAGNOSTIC_RESULTS_2026-09-04.md`, fixed-shape protocol, and the
exploratory recovery appendix.

### 12. Final SFT preservation confirmation

The fresh locked confirmation trained 32 states: two new corpus draws, two
order seeds, four regimes, and matched lookup-replay/control versus
switch-replay/treatment curricula. Each state contained 512 facts, 1,536
held-out wordings, 32 exact switches, and 32 natural switches.

| Curriculum | R4 lookup | R4 exact switching | Natural markers | Label-only natural |
|---|---:|---:|---:|---:|
| Lookup replay control | 24,504/24,576 (99.707%) | 36/512 (7.031%) | 500/512 (97.656%) | 10/512 |
| Switch replay treatment | 23,317/24,576 (94.877%) | 438/512 (85.547%) | 512/512 (100%) | 0/512 |

The frozen backbone scored 0/49,152. All 96 wrong-corpus R4 cells produced zero
foreign answers in 147,456 opportunities. All 7,310,124 access-map entries and
all tensor transfers were exact. One switch-treatment state collapsed to
352/1,536 R4 lookups, and only 6/16 treatment states met the strict combined
switch gate. Thus the treatment created a large switching gain but was not a
free lookup-preservation recipe.

The no-retraining 256-token appendix produced 125/128 natural EOS, 128/128
topical-marker passes, and zero label-only responses. Direct review retained
five localized factual errors, one same-lane lexical intrusion, and three cap
truncations, with no foreign-regime label substitution.

Archive source: `SFT_PRESERVATION_CONFIRMATION_RESULTS_2026-09-04.md` and its
machine receipt and conversation appendices.

### 13. R8/R16 production bounds

The final lifecycle study exercised cached transitions, distinct histories,
cancellation, replacement, row reorder/reversal, foreign refresh, own-generation
invalidation, and stale-state refusal.

| Roster | Exact target comparisons | Max off-lane logit delta | Argmax flips | Keyed accesses | Persistent inventory |
|---|---:|---:|---:|---:|---:|
| R8 | 448/448 | 0 | 0 | 257,148 | 33.866 GiB |
| R16 | 896/896 | 0 | 0 | 350,460 | 61.999 GiB |

All 8 R8 and 16 R16 conversations reached natural EOS before a 512-token cap.
At R16, a zero-copy contiguous view improved the tested median from 52.288 to
62.058 real tokens/s and removed 0.548 GiB of transient allocation. Reordered
rosters retain the safe gather fallback. These results make R8 the practical
integration candidate and R16 the retained stress tier; they do not authorize a
deployment.

Archive source: `PRODUCTION_BOUNDS_RESULTS_2026-09-04.md` and its conversation
appendix.

## What was corrected, and what was not

| Observed failure | Correction | Final interpretation |
|---|---|---|
| Transformers RoPE configuration API changed. | Updated the completeness/config check. | Infrastructure/API failure; no scientific phase promoted. |
| Early unseen wording missed the utility bar. | Added semantic wording and explicit fixed/swapped/direct controls. | Route matters; one miss remains in the denominator. |
| Repeated-SFT semantic cells did not all beat controls. | Preregistered label-crossover allocation. | Pooled capability passed; strict per-cell stress remained mixed. |
| Two “separate” worker probes shared task identity. | Required distinct non-null task IDs. | Infrastructure identity correction; final reopen/concurrency passed. |
| Receiver transported values but exact answer formatting failed. | No post hoc threshold change. | Mechanistic receiver passed; semantic decoding remained limited. |
| Naive concurrent CUDA streams were slow. | Tested row batching and selector views. | Scheduling issue, not lane contamination; performance remains shape-dependent. |
| R4 row-batch crossed the frozen numerical bound at 1,024. | Added ordinary native serial/batch control. | Ordinary and gated paths shared the cross-shape effect; protocol still not promoted. |
| Initial grouped SFT had zero F1 everywhere. | Added successful standalone gate, FP32 masters, four private blocks, and broader wording. | Earlier study was uninformative; repaired SFT and isolation passed. |
| Five sequence generations missed. | Traced first token and teacher-forced correct label. | Within-lane classification error followed by normal autoregression; no foreign learning. |
| One D checkpoint failed in 24-state scale-up. | Retained failure; ran one separately labeled exploratory seed. | Acquisition failure, not R4 damage; recovery does not replace denominator. |
| Automatic SDPA diverged late in R32 decoding. | Reproduced on identical-input, ungated, and operator controls; pinned math/eager policy. | Backend nondeterminism; automatic SDPA excluded. |
| Lookup training caused topic/label lock. | Matched lookup-replay with switch-replay curriculum. | Switching improved sharply, but acquisition tradeoff failed strict gate. |
| R32 consumed excessive persistent state. | Pivoted production tests to R8/R16 and optimized contiguous selection. | R8 candidate; R16 stress; R32 limit only. |

## Publication boundary

Promoted as a controlled research baseline:

- Gate-before-private-access placement;
- complete attention copies rather than native-head partitioning;
- canonical ownership of residual, cache, sequence, positions, masks, output,
  and parameter generation;
- finite exact off-lane mutation and wrong-corpus results;
- R8 lifecycle candidate and R16 stress tier; and
- measured artifact savings and bounded batching/selector behavior.

Not promoted:

- the current switch-oriented SFT recipe;
- universal cross-shape or cross-backend bit identity;
- naive simultaneous-dispatch acceleration;
- R32 deployment;
- arbitrary joins or variable-shape continuous batching;
- simultaneous SFT and serving;
- multi-process or hostile-tenant containment;
- allocator, timing, or output-channel noninterference; and
- a cryptographic or mathematical proof of universal isolation.
