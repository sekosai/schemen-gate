# CDP manuscript split

The authoritative full manuscript is `paper/cdp.tex` and `paper/cdp.pdf`.
The two focused manuscripts separate its claims by purpose:

1. `binary-activation-core.tex` - credential-derived binary activation,
   capability-PKI framing, FFN-coordinate confinement and gate cost, gated
   training, formal results, formative and
   strict Transformer-classifier evidence, capacity-preserving modular
   classifiers, private-capacity lanes, exact extraction, full-capacity
   inference controls, and placement negative controls.
2. `binary-activation-transformers.tex` - the useful Transformer surface,
   the strict FFN and preliminary public mask-adaptation results, why shared
   attention is outside those results, MoE expert-router authorization, and a
   deliberately modest research path for complete lanes.

## Read the training claims by lifecycle

The manuscripts do not describe one interchangeable “gated training” method.
They separate model preparation, tenant-stage training, and serving authority:

| Lifecycle | Trainable state | Supported interpretation |
|---|---|---|
| Atomic whole-model activation | No model change required | Identity-bound permission to invoke an unchanged model; unauthorized requests make zero model calls |
| Post-encoder co-training | Shared encoder and task head train with per-sample masks | Utility under a learned support constraint; **not** separated tenant state in the shared encoder |
| Strict intermediate-FFN training | Shared attention/backbone frozen; only the owning FFN slices and private classifier train | Exact measured preservation of inactive aligned FFN parameters and optimizer state, plus an empirical owning-utility curve |
| Public mask-aware adaptation | Public backbone first adapts across every intended mask, then freezes for tenant training | Preliminary initialization evidence; extra work, distillation, and mask awareness are confounded in the retained study |
| Frozen backbone plus private lanes | Shared backbone frozen; one private adapter, classifier, or expert per regime | Tenant-private trainable attachment state with explicit linear storage cost; shared attention remains shared |
| Complete Transformer lanes | Separate Q/K/V, output, normalization, FFN, residual, and cache paths from the outset | Open research protocol, not a result established by these papers |

“Frozen backbone” therefore means that the shared feature producer does not
change during the tenant stage. It does **not** mean that its existing
representations are private, statistically independent, or partitioned. A
post-hoc mask over an ordinary pretrained attention head is a destructive
ablation, not evidence of Binary Activation isolation.

## Geometry, placement, and optimizer requirements

An assignment exists only when `d >= R`. Equal-width regimes require `R | d`,
equivalently `d = qR`. If `d` is fixed, each regime receives `d/R` governed
coordinates and utility may fall as `R` grows. Holding `q` fixed preserves
per-regime width by growing private capacity linearly with `R`.

The strict FFN result places the gate after the element-wise nonlinearity and
immediately before the down projection. Its aligned trainable state is the
first projection's hidden-coordinate slices, hidden bias, down-projection
slices, and regime-scoped optimizer state. Zero loss gradient alone is not an
optimizer guarantee: momentum, Adam moments, decoupled weight decay, or a
post-step transform must also be restricted to the active support (or inactive
state must be restored after every step).

Every implementation claim must also name what remains outside the boundary:
shared attention, embeddings, normalization, residual routes, output bias or
heads unless separately governed, caches, adapters, retrieval state, logs, and
alternate serving paths.

## Minimum evaluation for a training claim

Evaluate owning and wrong-key behavior on the **same trained model**. Record
the complete denominator and separately audit:

1. owning-task utility and a matched capacity/control model;
2. every wrong key or a predeclared sampled set;
3. frozen shared parameter deltas;
4. off-partition parameter and optimizer-state deltas;
5. inactive private classifiers, adapters, or experts;
6. LayerNorm, residual, cache, and alternate-route bypass controls; and
7. artifact, mask geometry, model revision, data split, and authority
   provenance.

Wrong-key accuracy is a functional negative control. The exact structural
claim comes from the correctly placed zero and aligned update boundary; neither
an empirical zero count nor a non-significant accuracy difference is a general
privacy theorem.

Each current replayable protocol names its public runner and retained artifact
(or the result manifest that indexes those artifacts). Historical evidence is
explicit when its exact runner is not retained, so artifact availability is not
mistaken for full independent replayability.

Build both focused PDFs from this directory with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../output/pdf binary-activation-core.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=../../output/pdf binary-activation-transformers.tex
```

The core paper excludes residual-basis conjugation as a mechanism or tenant-
isolation claim, attention-lane claims, whole-model capability orchestration,
and implementation specifics. It retains one compact full-basis result solely
as a no-capacity-removal inference control. This is an editorial scope decision,
not a legal conclusion about patent priority or claim support.

Current reruns are bound to the exact `schemen-gate` source custody and the
source-visible research-preflight schema in
`experiments/schemen-library-lock.json`. Clean preflight-backed
artifacts cover whole-model permutation conjugation at R=8 and R=128, strict
R=8 FFN cotenancy, R=4 private lanes, R=8 public co-training, R=8 causal
generation, real-model Cargo, the R=8 service-consolidation benchmark, and
R=8 authority-constrained learned MoE routing on held-out data.
The token-routing extension additionally covers 311,477 held-out token
decisions under execution-selected private continuous prefixes, including a
literal capability-marker spoof control; plaintext prompt text is explicitly
not treated as authority.
