# Cryptographic Dimension Partitioning

**Identity-Bound FFN Confinement and Whole-Model Authorization**

This directory is the canonical publication and reproducibility bundle inside
[`sekosai/schemen-gate`](https://github.com/sekosai/schemen-gate). It was
imported from `sekosai/cdp-paper` at the exact revision recorded in
`SOURCE.json`; source-repository history was intentionally not imported.

Library code at the repository root is Apache-2.0. This research directory
uses the path-based policy in `LICENSES.md`. Before inclusion in the release
tree, absolute workstation paths in historical receipts were replaced with
explicit portable markers. Numerical results, dependency versions, source
commits, and artifact hashes recorded by the experiments were not rewritten.
Fourteen run records produced under a private companion authorization harness
are shipped as documented public evidence exports; `experiments/results/README.md`
lists their original-record digests and transformation rules.

## Abstract

Cryptographic Dimension Partitioning (CDP) binds an authenticated
tenant + sub-id scope to two gate surfaces. Within an FFN, a globally
assigned binary mask can act either on the pre-MLP input or on the
expanded intermediate activation. The strongest parameter-confinement
result uses the intermediate placement and aligned optimizer state.
Around a complete model, a separate fail-closed runtime gate treats the
unchanged model as one atomic resource and denies unauthorized
invocation before execution.

The FFN mechanism is governed by a single algebraic rule: the Hadamard
product of an activation with a binary mask. From this one rule,
four governance properties follow as one-line Lean 4 proofs: forward
coordinate exclusion, active preservation, gradient isolation, and gradient
confinement.

## Why this is both AuthN and AuthZ

CDP carries enterprise identity into AI execution instead of stopping at an API
gateway. A certificate or other approved credential authenticates the external
machine or issuer. A signed grant authorizes the exact Regime, model, operation,
and scope. The resolved Regime then serves as authenticated execution identity
for every downstream Gate that verifies the same bound scope:

```text
external AuthN -> boundary AuthZ -> resolved Regime -> downstream Regime AuthN
```

The deployment guarantee terminates at the classical IT trust boundary:
configured roots, certificate issuance, key custody, rotation, revocation, and
bypass closure. Explicitly pinned self-signed roots are valid, but prove only
possession and continuity of the pinned key. PKCS#12 is a portable credential
container; it does not by itself prove hardware key residency.

## Paper

The paper is in `paper/cdp.tex`. Build with:

```bash
cd research/cdp/paper
latexmk -pdf -interaction=nonstopmode -halt-on-error cdp.tex
```

## Companion gated-Transformer study

The later [Schemen-gated Transformer regime-lane paper](gated-transformer-regime-lanes/SCHEMEN_GATED_TRANSFORMER_REGIME_LANES_PAPER.md)
is published beside, but not merged into, the original CDP manuscript's claim
set. Its [results-and-corrections summary](gated-transformer-regime-lanes/RESULTS_AND_CORRECTIONS.md),
[claim ledger](gated-transformer-regime-lanes/CLAIM_LEDGER.md), and
[reproducibility record](gated-transformer-regime-lanes/REPRODUCIBILITY.md)
retain both successful controls and corrected failures.

That study tests a different construction: each Regime receives a complete
Q/K/V/O attention alternative and its own mutable decoder state while selected
backbone operators remain shared and read-only. The reported evidence is
bounded empirical evidence, not a claim of universal or production-grade
isolation.

## Key results

- **Complete attention and decoder-state lanes**: The companion Qwen study
  reports 0/147,456 foreign-corpus answers, 992/992 fixed-shape own-lane
  intervention positives with 2,976/2,976 bit-exact off-lane comparisons, and
  exact R8/R16 lifecycle controls. R32 is retained only as backend/memory-limit
  evidence; the tested switch-oriented SFT curriculum and universal isolation
  claims are withheld.
- **Exact addressed-use parity through the execution preflight**: A clean rerun of
  whole-model permutation conjugation has exactly zero AG News accuracy loss
  for all 8 regimes at R=8 and all 128 regimes at R=128 (baseline and every
  addressed regime: 91.42%). The maximum fp32 logit drift is
  $1.06\times10^{-5}$ and does not change any prediction. This is an
  equivariance result, not simultaneous sparse cotenancy.
- **Formative classification cost**: Five-seed DistilBERT
  separate-minus-gated gaps are 0.14–0.38 percentage points across
  R=8–128 at a post-encoder CLS gate; none of seven paired tests
  survives Bonferroni correction
- **Formal verification**: 21 Lean 4 modules and 312 theorem/lemma
  declarations, with no `sorry`; three explicit custom axioms (plus two
  opaque predicates) are confined to conditional
  cryptographic/adversarial chains; two earlier statistical axioms
  (`camouflage_indistinguishable`, `gradient_probing_hard`) are retired
  as not provable white-box — the deployments they served are secured by
  an unbundled historical custody design; it is not part of this release's
  proof claim
- **Strict intermediate-FFN cotenancy**: Across R=1–16, frozen shared,
  off-partition parameter, inactive optimizer-moment, and inactive
  classifier deltas are exactly zero; accuracy declines from 91.28%
  to 86.21%
- **Current-library R=8 checks**: A one-seed strict DistilBERT rerun reaches
  87.76% owning accuracy with every unauthorized parameter/moment delta
  exactly zero. A matched public co-training factorial improves the later
  tenant stage by 1.33 percentage points from mask awareness and 1.44 points
  end to end. These are one-seed corroborations, not replacements for the
  multi-seed table.
- **Frozen private lanes**: One current-library seed reaches 90.66% owning
  accuracy for residual adapters and 90.15% for full FFN experts, with 3.11%
  and 3.28% wrong-key accuracy and exactly zero shared/inactive-lane change.
- **Causal generation boundary**: At R=8, TinyLlama's correctly placed
  post-SwiGLU/pre-down-projection gate has exact inactive gradient and
  parameter confinement, but token loss is 5.442 versus 4.230 for its matched
  ungated control. Isolation is proven; utility parity is not.
- **Support-constrained training**: Excluded FFN coordinates contribute
  exactly zero, so each regime's gated contribution must be represented
  on its active support
- **Exact extraction**: 160 full-gated-versus-sliced projection
  comparisons pass a conservative forward-error bound. The canonical
  768-dimensional run happened to be bit-identical in fp32; smaller-kernel
  checks show that fp32 bit identity is not a general guarantee
- **Surface checks**: The local harness verifies zero inactive
  pre-MLP input/$W_1$ gradients and rejects wrong tenant, sub-id,
  model, and operation keys with zero whole-model calls; the owning
  path is numerically unchanged
- **Cargo and whole-model preflight authorization**: The current real-model run has 4/4 exact
  owning answers and rejects all 23 wrong-scope probes before any generator
  call, including tenant, subject, regime, model, operation, policy,
  partition, and malformed execution scopes.

## Reproducing experiments

This repository is the publication bundle for independently rerunning
and checking the paper's measurements:

```text
experiments/*.py          Experiment scripts and launchers
experiments/results/*.json
                          Machine-readable raw results
docs/experiment-data-inventory.md
                          Claim-to-artifact inventory
```

See `docs/experiment-data-inventory.md` for claim status and artifact paths.
See `experiments/README.md` for the self-contained public runners and
the distinction among formative post-encoder, pre-MLP FFN,
intermediate-FFN, and atomic whole-model surfaces.
See `docs/practical-enforcement-audit.md` for the audited IdP-to-gate
architecture, SPIFFE workload-authentication profile, exact-use requirements,
principal-scoped revocation semantics, and lockbox-compromise boundary.

For a release re-certification rather than an individual launcher invocation,
use the repository-level
[`docs/MODAL_RECERTIFICATION.md`](../../docs/MODAL_RECERTIFICATION.md). It
seals the exact commit and launcher arguments, estimates gross cost before
execution, requires an explicit approval ceiling and provider-budget review,
runs reduced canaries first, validates construction-specific receipts, and
moves new artifacts outside the source checkout between jobs.

All Modal launchers are under `experiments/modal_*.py`. Run them from this
directory, beginning with the documented `--smoke` mode where available. Every
launcher pins image dependencies and external model/dataset revisions and caps
remote autoscaling at three containers. Canonical runs require a clean Git
checkout and bind the current repository commit, Gate source-tree digest,
script digest, and research-preflight schema into their custody record. No
server wheel or bearer-authentication surface is included.

## Repository structure

```
paper/          LaTeX source and figures
proofs/         Lean 4 proof files (referenced from paper)
experiments/    Experiment scripts
  schemen-library-lock.json  Exact Gate source custody and fixture schema
  results/      Raw result JSON files
gated-transformer-regime-lanes/
                Curated paper, corrections, toy code, selected results, and figures
```

## Lean proofs

The proofs referenced in the paper live under `proofs/`. Key files:

| File | Scope |
|------|-------|
| `GateSecurity.lean` | Core isolation properties |
| `GatePlacement.lean` | Placement constraints, LayerNorm negative result |
| `ModelSecurityV3.lean` | Aligned FFN and residual-branch confinement |
| `CapacitySecurity.lean` | Cross-mask additive cancellation |
| `BiometricSecurity.lean` | Biometric enrollment certificate |
| `MultiEncoder.lean` | Federation, Procrustes alignment |
| `AttentionLeakage.lean` | Attention cannot be partitioned |

## Patent notice

A U.S. provisional patent application was filed before release for subject
matter related to portions of Schemen Gate. This notice adds no separate
restriction. For Apache-2.0 material, patent rights are governed by Section 3
of the Apache License 2.0. CC BY 4.0 does not license patent rights. The
application number, unpublished claims, filing documents, and private
prosecution records are not part of this repository.

## License

This research snapshot uses a path-based license policy. Executable code and
Lean proof source are Apache-2.0; authored papers, documentation, figures, and
designated research records are CC BY 4.0. Third-party material retains its
original terms. See [LICENSES.md](LICENSES.md) for the authoritative mapping.
