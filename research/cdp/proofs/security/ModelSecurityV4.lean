/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/


import ModelSecurityV3

/-!
# V4 Scope Statement — `IsSurjective` Replacement (definitions only)

## Purpose

V3 introduced `IsSurjective T` as a per-process hypothesis used in
`zero_key_information`. The adversarial review of April 2026 surfaced that
`IsSurjective` is false for realistic SGD-based training: training trajectories
form a measure-zero manifold in weight space, so the image of `T.train P D`
over all `D` is a low-dimensional set, not the full weight space. Any claim
depending on surjectivity is not a property of deployed systems.

## What V4 does

V4 defines the **types** needed for an honest cryptographic reduction
(adversary advantage, PRF distinguishing advantage, partition obliviousness)
and documents the **shape of the reduction** that would replace
`zero_key_information`. V4 does **not** contain a formal proof of the
reduction. A proper reduction requires a probability space over training data
and a formal model of polynomial-time adversaries; both require Mathlib's
`MeasureTheory.ProbabilityMeasure` infrastructure and a substantial
formalization effort beyond the scope of the current proof corpus.

**What was deleted in the April 2026 cleanup.** A previous draft of this
file contained two axioms (`weight_observation_reduces_to_prf` and
`prf_distinguishing_advantage_negligible`) and a reduction theorem
(`posterior_bounded_by_prf_advantage`). Review found that both axioms were
trivially satisfiable (the first because `PRFAdvantage` is a struct that can
be constructed at any value; the second because its type reduced to the law
of excluded middle on reals). The theorem was removed along with them. Two
`Unit`-valued markers (`zero_key_information_DEPRECATED` and
`v4_reduction_scope_statement`) were also removed.

## What this file contains now

Three definitions only:

- `PartitionOblivious T` — a per-process hypothesis, currently a marker
  (`True`). Promoting this to a genuine probabilistic constraint is future
  work (V5). The *operational* claim — "the customer's training pipeline does
  not encode the partition" — is documented in
  `docs/executive-summary.md` as a deployment requirement.
- `AdversaryAdvantage` — typed structure for the distinguishing advantage of
  a weight-observing adversary. Preserved as scaffolding for a future
  reduction proof.
- `PRFAdvantage` — typed structure for the PRF distinguishing advantage.
  Scaffolding for the same reduction.

## The claim Schemen makes based on these types

Schemen's public-facing cryptographic claim is:

  > "Under the PRF assumption and assuming the customer's training data is
  > partition-oblivious, recovering the key from the observed model weights
  > is as hard as distinguishing the underlying PRF (HMAC-SHA256) from a
  > random function."

This is the standard shape of a cryptographic reduction claim. It is **not**
formally proven in Lean at this time. The V2 axiom
`prf_brute_force_optimal` (in `ModelSecurityV2.lean`) is a related but
distinct claim: it bounds the query budget of an adversary who successfully
recovers the partition. The two together give a defensible informal
cryptographic argument. The V5 work, when it happens, will formalize the
distinguishing-advantage reduction against Mathlib probability theory.

## Relationship to V3's `zero_key_information`

V3's `zero_key_information` theorem remains in `ModelSecurityV3.lean` with
an explicit scope note that its `IsSurjective T` hypothesis is false for
realistic SGD. It is retained for historical completeness and as an API
peg for downstream code that references it. **Do not cite
`zero_key_information` in marketing, compliance submissions, or patent
specifications.** Cite the informal cryptographic reduction described in
this file's doc comment instead.
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.CryptoReduction

open Schemen Schemen.Security Schemen.SecurityV2 Schemen.SecurityV3


-- ════════════════════════════════════════════════════════════════
-- §1. PARTITION OBLIVIOUSNESS (marker; promote to real prop in V5)
-- ════════════════════════════════════════════════════════════════

/-- A training process is partition-oblivious if the distribution over training
    data is independent of the partition structure used during training.

    In the Schemen deployment model:

    • Training data comes from the tenant.
    • The tenant does not possess the partition structure.
    • The tenant's data labels / prompts / embeddings do not encode
      partition information.

    Under these conditions, no statistical signal in the training data
    correlates with the partition, so no observation of the resulting weights
    can distinguish partitions beyond the PRF distinguishing advantage.

    **Current formalization status: marker (`True`).** A faithful
    formalization requires `MeasureTheory.ProbabilityMeasure` over `T.Data`
    and a statement of the form

        ∀ P₁ P₂ : ValidPartition n R,
          μ_{conditional on partition=P₁} = μ_{conditional on partition=P₂}

    This is V5 work. The marker exists so that theorems that *would* depend
    on partition-obliviousness have a named hypothesis to reference. It is
    not a constraint. The operational claim that partition-obliviousness
    holds is the customer's responsibility, documented in the deployment
    guide. -/
def PartitionOblivious {n R m o : ℕ} (_T : TrainingProcess n R m o) : Prop :=
  True


-- ════════════════════════════════════════════════════════════════
-- §2. ADVERSARY ADVANTAGE SCAFFOLDING (types for future reduction)
--
-- The types below are the minimal scaffolding for a cryptographic-
-- reduction proof. They are NOT proofs. They are the shape the
-- proof will take once the probability-theory layer is built.
-- ════════════════════════════════════════════════════════════════

/-- The distinguishing advantage of an adversary who observes model weights
    and attempts to identify the key. Advantage 0 means perfect
    indistinguishability; larger values mean the adversary has non-negligible
    edge over guessing.

    This is a scaffolding type. A future reduction theorem will quantify over
    `AdversaryAdvantage` and relate its `value` to `PRFAdvantage.value`. -/
structure AdversaryAdvantage (n R m o : ℕ) where
  scheme : CryptoScheme n R
  process : TrainingProcess n R m o
  value : ℝ
  nonneg : 0 ≤ value

/-- The PRF distinguishing advantage — the standard quantity from the
    cryptographic literature (Bellare-Canetti-Krawczyk 1996, Bellare 2006).
    For HMAC-SHA256 with 256-bit keys and polynomial-time adversaries, this
    advantage is believed to be negligible in the key length. This belief is
    the PRF assumption underlying TLS, AES-GCM, and most modern internet
    security.

    Scaffolding type. See `docs/executive-summary.md` for the informal
    reduction claim this type participates in. -/
structure PRFAdvantage where
  value : ℝ
  nonneg : 0 ≤ value


end Schemen.CryptoReduction
