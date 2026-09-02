/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic

/-!
# EnglishTax — Channel-Level Fidelity Bounds

Mechanized proofs of the channel-level claims in
"The English Tax: Quantifying Information Loss in LLM Agent Chains".

These results formalize the inter-agent communication channel,
not the LLM as a process.  The English Tax claim is split into
two parts:

1.  The **channel** is either lossless (Vectorese: pass bytes) or
    lossy (English: encode → text → decode).  This file proves the
    channel-level dichotomy.
2.  The LLM as a stochastic operator induces *contraction toward an
    attractor*.  That part is empirical (see paper §5) and is **not**
    formalized here, because doing so would require committing to a
    measure-theoretic model of the LLM that we do not have evidence
    for.  We do NOT prove exponential decay; the data plateau and
    do not support that bound.

## Statement of Results

### Channel claim — Vectorese is lossless
- `cosSim_self`: cos(v, v) = 1 for every non-zero v.
- `cosSim_id_iter`: iterating the identity preserves cos = 1.

### Cosine bounds (Cauchy–Schwarz)
- `abs_cosSim_le_one`: |cos(v, w)| ≤ 1.
- `cosSim_le_one`: cos(v, w) ≤ 1.
- `neg_one_le_cosSim`: −1 ≤ cos(v, w).

### English path is lossy under any non-positive-collinear transformation
- `cosSim_eq_one_iff_sameRay`: cos(v, w) = 1 ↔ v and w lie on the same ray.
- `cosSim_lt_one_of_not_sameRay`: any transformation T with T(v) not on
  the same ray as v gives cos(v, T(v)) < 1.

### Single-hop strict loss under additive perturbation
- `additive_perturbation_strict_loss`: if a perturbation ε is not a
  non-negative scaling of v (so v + ε is not on v's ray), then
  cos(v, v + ε) < 1.

### Quantitative hardware floor (§7)
- `operational_floor_bound`: for `‖ε‖ ≤ δ · ‖v‖` with `δ < 1`,
  `cosSim(v, v + ε) ≥ (1 − δ) / (1 + δ)`.
- `operational_floor_two_delta`: cosine deviation
  `1 − cosSim(v, v + ε) ≤ 2δ / (1 + δ) ≤ 2δ`.

These bound the *perturbation* contribution to per-step loss.
They are precision-dependent (shrink δ, the bound tightens),
which is the formal counterpart of the empirical "hardware
floor" decomposition in paper §5.6.

## What is NOT in this file (and is honest about it)

- No claim that expected cosine decays exponentially with chain depth.
  The empirical data show plateau-style contraction toward an attractor
  cosine c_∞ ∈ (0, 1), not exponential decay.

- No claim that per-hop noise is iid isotropic.  It is not.  LLM
  paraphrase noise is correlated, structured, and biased toward an
  attractor distribution.

## Axioms

ZERO new axioms.  Everything reduces to mathlib's real inner product
space (Cauchy–Schwarz and its equality case).
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.EnglishTax

open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- ════════════════════════════════════════════════════════════
-- §0. CORE DEFINITION — COSINE SIMILARITY
-- ════════════════════════════════════════════════════════════

/-- Cosine similarity between two vectors:
    cos(v, w) = ⟨v, w⟩ / (‖v‖ · ‖w‖). -/
def cosSim (v w : E) : ℝ :=
  inner ℝ v w / (‖v‖ * ‖w‖)


-- ════════════════════════════════════════════════════════════
-- §1. CHANNEL CLAIM — VECTORESE IS LOSSLESS
--
-- The Vectorese channel passes the embedding bytes unchanged.
-- This is an identity transformation on the embedding space,
-- and identities preserve cosine similarity perfectly.
--
-- This is the channel-level content of the paper's headline
-- "cosine = 1.000 across all hops" claim.  It is provable
-- because, properly understood, it is a tautology about
-- identity transformations — and that is exactly the point:
-- a lossless channel is, by definition, an identity in the
-- transport layer.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Cosine self-similarity).**
    Every non-zero vector is perfectly aligned with itself.
    This is the channel-level claim of Vectorese: byte-stable
    transport (the identity) preserves cosine = 1. -/
theorem cosSim_self {v : E} (hv : v ≠ 0) :
    cosSim v v = 1 := by
  unfold cosSim
  rw [real_inner_self_eq_norm_mul_norm]
  have hnorm : ‖v‖ ≠ 0 := norm_ne_zero_iff.mpr hv
  field_simp

/-- **Theorem (Vectorese is lossless under iteration).**
    Iterating the identity n times preserves cosine = 1.
    No matter how many hops in a Vectorese chain, the receiver's
    vector matches the sender's vector exactly. -/
theorem cosSim_id_iter (v : E) (hv : v ≠ 0) (n : ℕ) :
    cosSim v ((id : E → E)^[n] v) = 1 := by
  rw [Function.iterate_id, id]
  exact cosSim_self hv


-- ════════════════════════════════════════════════════════════
-- §2. COSINE BOUNDS — CAUCHY–SCHWARZ
--
-- Cosine similarity is bounded in [-1, 1].  This is the
-- Cauchy–Schwarz inequality applied to a normalized inner
-- product.  Mathlib provides the underlying inequality;
-- we wrap it for the cosine.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (|cos| ≤ 1, Cauchy–Schwarz).**
    For any two non-zero vectors, the absolute value of cosine
    similarity is at most 1. -/
theorem abs_cosSim_le_one {v w : E} (hv : v ≠ 0) (hw : w ≠ 0) :
    |cosSim v w| ≤ 1 := by
  unfold cosSim
  have hp : (0 : ℝ) < ‖v‖ * ‖w‖ :=
    mul_pos (norm_pos_iff.mpr hv) (norm_pos_iff.mpr hw)
  rw [abs_div, abs_of_pos hp]
  rw [div_le_one hp]
  exact abs_real_inner_le_norm v w

/-- **Theorem (Cosine upper bound).**
    For any two non-zero vectors, cos(v, w) ≤ 1. -/
theorem cosSim_le_one {v w : E} (hv : v ≠ 0) (hw : w ≠ 0) :
    cosSim v w ≤ 1 := by
  exact (abs_le.mp (abs_cosSim_le_one hv hw)).2

/-- **Theorem (Cosine lower bound).**
    For any two non-zero vectors, −1 ≤ cos(v, w). -/
theorem neg_one_le_cosSim {v w : E} (hv : v ≠ 0) (hw : w ≠ 0) :
    -1 ≤ cosSim v w := by
  exact (abs_le.mp (abs_cosSim_le_one hv hw)).1


-- ════════════════════════════════════════════════════════════
-- §3. EQUALITY CASE — COSINE = 1 IFF SAME RAY
--
-- The equality case of Cauchy–Schwarz: cos(v, w) = 1 holds iff
-- v and w lie on the same ray from the origin (one is a
-- non-negative scalar multiple of the other).  Restricted to
-- non-zero vectors, this means w = α·v for some α > 0.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Cosine = 1 characterizes the same ray).**
    Two non-zero vectors have cosine similarity 1 if and only
    if they point in the same direction (lie on the same ray
    from the origin). -/
theorem cosSim_eq_one_iff_sameRay {v w : E} (hv : v ≠ 0) (hw : w ≠ 0) :
    cosSim v w = 1 ↔ SameRay ℝ v w := by
  unfold cosSim
  have hp : (0 : ℝ) < ‖v‖ * ‖w‖ :=
    mul_pos (norm_pos_iff.mpr hv) (norm_pos_iff.mpr hw)
  rw [div_eq_one_iff_eq hp.ne']
  rw [inner_eq_norm_mul_iff_real, sameRay_iff_norm_smul_eq]
  exact eq_comm


-- ════════════════════════════════════════════════════════════
-- §4. ENGLISH PATH IS LOSSY — STRICT INEQUALITY UNDER ANY
--     NON-POSITIVE-COLLINEAR TRANSFORMATION
--
-- The contrapositive of §3: if a transformation T does NOT
-- send v to a positive scalar multiple of itself, then
-- cos(v, T(v)) < 1.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Strict loss under non-collinear transformation).**
    Any non-zero output that does not lie on the same ray as the
    input strictly drops cosine below 1.  This is the channel-level
    statement of "any non-identity transport that changes the
    embedding direction is lossy." -/
theorem cosSim_lt_one_of_not_sameRay
    {v w : E} (hv : v ≠ 0) (hw : w ≠ 0)
    (h : ¬ SameRay ℝ v w) :
    cosSim v w < 1 := by
  rcases lt_or_eq_of_le (cosSim_le_one hv hw) with hlt | heq
  · exact hlt
  · exact absurd ((cosSim_eq_one_iff_sameRay hv hw).mp heq) h


-- ════════════════════════════════════════════════════════════
-- §5. ADDITIVE PERTURBATION — SINGLE-HOP STRICT LOSS
--
-- The simplest non-trivial English-path model: the receiver
-- gets v + ε for some perturbation ε.  When ε is not a
-- non-negative scalar multiple of v (i.e., when v + ε does
-- not lie on v's ray), the cosine strictly drops below 1.
--
-- We do NOT claim a quantitative bound here — the magnitude
-- of the drop depends on the angle between ε and v, and on
-- ‖ε‖, in a way that is not captured by a simple closed form.
-- We only claim strict loss exists.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Single-hop strict loss under non-aligned perturbation).**
    If a perturbation ε leaves the perturbed vector v + ε non-zero
    and not on the same ray as v, then cos(v, v + ε) < 1. -/
theorem additive_perturbation_strict_loss
    {v ε : E} (hv : v ≠ 0) (hvε : v + ε ≠ 0)
    (hne : ¬ SameRay ℝ v (v + ε)) :
    cosSim v (v + ε) < 1 :=
  cosSim_lt_one_of_not_sameRay hv hvε hne


-- ════════════════════════════════════════════════════════════
-- §6. SPECIALIZATION — EUCLIDEAN SPACE
--
-- Embedding vectors live in ℝᵈ.  We instantiate the generic
-- theorems on EuclideanSpace ℝ (Fin d) so the paper's claims
-- are stated for the concrete type used in the experiments.
-- ════════════════════════════════════════════════════════════

/-- The embedding space for a d-dimensional sentence-transformer
    encoder is ℝᵈ with the standard inner product. -/
abbrev Embedding (d : ℕ) := EuclideanSpace ℝ (Fin d)

/-- Specialization of `cosSim_self` to ℝᵈ.  The Vectorese channel
    on d-dimensional embeddings is lossless. -/
theorem vectorese_lossless_euclidean
    {d : ℕ} {v : Embedding d} (hv : v ≠ 0) :
    cosSim v v = 1 :=
  cosSim_self hv

/-- Specialization of `cosSim_id_iter` to ℝᵈ.  An n-hop Vectorese
    chain on d-dimensional embeddings preserves cosine = 1 exactly. -/
theorem vectorese_lossless_iterated_euclidean
    {d : ℕ} (v : Embedding d) (hv : v ≠ 0) (n : ℕ) :
    cosSim v ((id : Embedding d → Embedding d)^[n] v) = 1 :=
  cosSim_id_iter v hv n


-- ════════════════════════════════════════════════════════════
-- §7. OPERATIONAL-RESOLUTION FLOOR — HARDWARE PRECISION BOUND
--
-- A computation in floating-point arithmetic produces a vector
-- that differs from the exact mathematical result by a bounded
-- relative perturbation.  This perturbation translates into a
-- bounded cosine deviation, which is the "hardware floor" on
-- representational fidelity at fixed precision.
--
-- This complements §5 (qualitative strict loss) with a
-- quantitative bound:
--
--   ‖ε‖ ≤ δ · ‖v‖   ⟹   cosSim(v, v + ε) ≥ (1 − δ) / (1 + δ).
--
-- For small δ this gives 1 − cosSim ≤ 2δ + O(δ²), which is the
-- per-step cosine-fidelity floor under bounded relative error.
-- ════════════════════════════════════════════════════════════

set_option linter.unusedSectionVars false in
/-- **Lemma (Reverse triangle, additive form).**
    For any `v, ε : E`, the norm of `v + ε` is at least `‖v‖ − ‖ε‖`.
    Used to bound the denominator of cosine under additive
    perturbation.  Holds for any normed group; the inner-product
    structure is unused here and is harmless. -/
lemma norm_add_ge_sub (v ε : E) : ‖v‖ - ‖ε‖ ≤ ‖v + ε‖ := by
  have step : ‖v‖ ≤ ‖v + ε‖ + ‖ε‖ := by
    calc ‖v‖ = ‖(v + ε) + (-ε)‖ := by congr 1; abel
      _ ≤ ‖v + ε‖ + ‖-ε‖ := norm_add_le _ _
      _ = ‖v + ε‖ + ‖ε‖ := by rw [norm_neg]
  linarith

/-- **Theorem (Operational-resolution floor).**
    Given an exact result `v ≠ 0` and a computed result `v + ε`
    with `‖ε‖ ≤ δ · ‖v‖` for some `δ ∈ [0, 1)`, the cosine
    similarity between the two satisfies

    cosSim v (v + ε) ≥ (1 − δ) / (1 + δ).

    This is the channel-level statement of the *hardware floor*:
    bounded relative perturbation gives bounded cosine fidelity.
    The proof has no LLM-specific content — it is the worst-case
    cosine deviation under any additive perturbation of relative
    norm at most δ. -/
theorem operational_floor_bound
    {v ε : E} (hv : v ≠ 0) {δ : ℝ}
    (hδ_nn : 0 ≤ δ) (hδ1 : δ < 1)
    (hε : ‖ε‖ ≤ δ * ‖v‖) :
    (1 - δ) / (1 + δ) ≤ cosSim v (v + ε) := by
  have hvn : 0 < ‖v‖ := norm_pos_iff.mpr hv
  have hvsq : 0 < ‖v‖ ^ 2 := by positivity
  -- Step 1: numerator lower bound  ⟨v, v + ε⟩ ≥ (1 − δ) ‖v‖²
  have h_inner_lb : (1 - δ) * ‖v‖ ^ 2 ≤ inner ℝ v (v + ε) := by
    have h_inner_ε : -(δ * ‖v‖ ^ 2) ≤ inner ℝ v ε := by
      have h_abs : |inner ℝ v ε| ≤ ‖v‖ * ‖ε‖ := abs_real_inner_le_norm v ε
      have h_step : ‖v‖ * ‖ε‖ ≤ δ * ‖v‖ ^ 2 := by
        have : ‖v‖ * ‖ε‖ ≤ ‖v‖ * (δ * ‖v‖) :=
          mul_le_mul_of_nonneg_left hε hvn.le
        nlinarith [sq_nonneg ‖v‖, this]
      linarith [neg_abs_le (inner ℝ v ε)]
    rw [inner_add_right, real_inner_self_eq_norm_mul_norm]
    have : ‖v‖ * ‖v‖ = ‖v‖ ^ 2 := by ring
    linarith
  -- Step 2: denominator upper bound  ‖v‖ · ‖v + ε‖ ≤ (1 + δ) ‖v‖²
  have h_norm_ub : ‖v + ε‖ ≤ (1 + δ) * ‖v‖ := by
    calc ‖v + ε‖ ≤ ‖v‖ + ‖ε‖ := norm_add_le v ε
      _ ≤ ‖v‖ + δ * ‖v‖ := by linarith
      _ = (1 + δ) * ‖v‖ := by ring
  have h_norm_lb : (1 - δ) * ‖v‖ ≤ ‖v + ε‖ := by
    have h_rev : ‖v‖ - ‖ε‖ ≤ ‖v + ε‖ := norm_add_ge_sub v ε
    linarith
  have h_norm_pos : 0 < ‖v + ε‖ := by
    have : 0 < (1 - δ) * ‖v‖ := mul_pos (by linarith) hvn
    linarith
  have h_denom_pos : 0 < ‖v‖ * ‖v + ε‖ := mul_pos hvn h_norm_pos
  have h_denom_ub : ‖v‖ * ‖v + ε‖ ≤ (1 + δ) * ‖v‖ ^ 2 := by
    have : ‖v‖ * ‖v + ε‖ ≤ ‖v‖ * ((1 + δ) * ‖v‖) :=
      mul_le_mul_of_nonneg_left h_norm_ub hvn.le
    nlinarith [this]
  have h_one_plus_pos : (0 : ℝ) < 1 + δ := by linarith
  -- Step 3: assemble the cross-multiplied bound.
  have h_final :
      (1 - δ) * (‖v‖ * ‖v + ε‖) ≤ inner ℝ v (v + ε) * (1 + δ) := by
    have h_lhs_ub :
        (1 - δ) * (‖v‖ * ‖v + ε‖) ≤ (1 - δ) * ((1 + δ) * ‖v‖ ^ 2) :=
      mul_le_mul_of_nonneg_left h_denom_ub (by linarith)
    have h_rhs_lb :
        (1 - δ) * ((1 + δ) * ‖v‖ ^ 2) ≤ inner ℝ v (v + ε) * (1 + δ) := by
      have step :
          (1 - δ) * ‖v‖ ^ 2 * (1 + δ) ≤ inner ℝ v (v + ε) * (1 + δ) :=
        mul_le_mul_of_nonneg_right h_inner_lb h_one_plus_pos.le
      nlinarith [step]
    linarith
  -- Step 4: divide both sides by the positive denominators.
  unfold cosSim
  rw [le_div_iff₀ h_denom_pos, div_mul_eq_mul_div,
      div_le_iff₀ h_one_plus_pos]
  exact h_final

/-- **Corollary (Two-delta floor).**
    Under the hypotheses of `operational_floor_bound`, the cosine
    deviation `1 − cosSim v (v + ε)` is at most `2δ / (1 + δ) ≤ 2δ`.
    This is the loose-but-cheap form often invoked in numerical
    error-budget arguments. -/
theorem operational_floor_two_delta
    {v ε : E} (hv : v ≠ 0) {δ : ℝ}
    (hδ_nn : 0 ≤ δ) (hδ1 : δ < 1)
    (hε : ‖ε‖ ≤ δ * ‖v‖) :
    1 - cosSim v (v + ε) ≤ 2 * δ / (1 + δ) := by
  have h := operational_floor_bound hv hδ_nn hδ1 hε
  have h_one_plus_pos : (0 : ℝ) < 1 + δ := by linarith
  have step : 1 - 2 * δ / (1 + δ) = (1 - δ) / (1 + δ) := by
    field_simp
    ring
  linarith [step]

/-- **Theorem (Tight operational-resolution floor).**
    Under the hypotheses of `operational_floor_bound`, the cosine
    similarity satisfies the stronger bound

       cosSim v (v + ε) ≥ √(1 − δ²).

    This is the *tight* per-step floor: the worst-case adversarial
    perturbation `ε` of relative norm `δ` saturates `√(1 − δ²)`
    (achieved at `ε = -δ²·v + δ·√(1−δ²)·w` for any unit `w ⊥ v`),
    and this bound is therefore the infimum of cosine fidelity over
    all admissible perturbations.

    The loose form `(1 − δ) / (1 + δ)` is between roughly half and
    all of `√(1 − δ²)` for `δ ∈ [0, 1)` — both are valid floors,
    but the tight one is the right object for chains of more than a
    handful of channels.

    Proof key.  With `a := ‖v‖²`, `b := ⟨v, ε⟩`, `c := ‖ε‖²`, the
    cosine squared is `(a + b)² / (a · (a + 2b + c))`.  The identity

       (a + b)² − (1 − δ²) · a · (a + 2b + c)
            = (a·δ² + b)² + (1 − δ²) · a · (δ²·a − c)

    decomposes the gap as a sum of squares plus a term that is
    non-negative by the relative-perturbation hypothesis (`c ≤ δ²·a`).
    Hence cosine² ≥ 1 − δ² and, since cosine is positive (already
    established by the loose floor), cosine ≥ √(1 − δ²). -/
theorem operational_floor_tight
    {v ε : E} (hv : v ≠ 0) {δ : ℝ}
    (hδ_nn : 0 ≤ δ) (hδ1 : δ < 1)
    (hε : ‖ε‖ ≤ δ * ‖v‖) :
    Real.sqrt (1 - δ ^ 2) ≤ cosSim v (v + ε) := by
  have hvn : 0 < ‖v‖ := norm_pos_iff.mpr hv
  have hvsq : 0 < ‖v‖ ^ 2 := by positivity
  -- Cosine is positive (loose floor already guarantees this).
  have h_loose := operational_floor_bound hv hδ_nn hδ1 hε
  have h_loose_pos : 0 < (1 - δ) / (1 + δ) :=
    div_pos (by linarith) (by linarith)
  have h_cos_pos : 0 < cosSim v (v + ε) := lt_of_lt_of_le h_loose_pos h_loose
  -- Norms.
  have h_norm_lb : (1 - δ) * ‖v‖ ≤ ‖v + ε‖ := by
    have h_rev : ‖v‖ - ‖ε‖ ≤ ‖v + ε‖ := norm_add_ge_sub v ε
    linarith
  have h_norm_pos : 0 < ‖v + ε‖ := by
    have : 0 < (1 - δ) * ‖v‖ := mul_pos (by linarith) hvn
    linarith
  have h_denom_pos : 0 < ‖v‖ * ‖v + ε‖ := mul_pos hvn h_norm_pos
  -- Algebraic abbreviations.
  set a : ℝ := ‖v‖ ^ 2 with ha_def
  set b : ℝ := inner ℝ v ε with hb_def
  set c : ℝ := ‖ε‖ ^ 2 with hc_def
  have ha_pos : 0 < a := hvsq
  -- c ≤ δ²·a (square of the perturbation bound).
  have hc_le : c ≤ δ ^ 2 * a := by
    have h_sq : ‖ε‖ * ‖ε‖ ≤ (δ * ‖v‖) * (δ * ‖v‖) :=
      mul_le_mul hε hε (norm_nonneg ε) (by positivity)
    have h_lhs : ‖ε‖ * ‖ε‖ = c := by rw [hc_def]; ring
    have h_rhs : (δ * ‖v‖) * (δ * ‖v‖) = δ ^ 2 * a := by rw [ha_def]; ring
    linarith [h_lhs, h_rhs, h_sq]
  -- Inner-product identity: ⟨v, v+ε⟩ = a + b.
  have h_inner : inner ℝ v (v + ε) = a + b := by
    rw [inner_add_right, real_inner_self_eq_norm_mul_norm]
    show ‖v‖ * ‖v‖ + b = a + b
    have : ‖v‖ * ‖v‖ = a := by rw [ha_def]; ring
    linarith
  -- Norm-squared identity: ‖v+ε‖² = a + 2b + c.
  have h_norm_sq : ‖v + ε‖ ^ 2 = a + 2 * b + c := by
    rw [@norm_add_sq_real _ _ _ v ε]
  -- The sum-of-squares identity that powers the proof.
  have h_sos :
      (a + b) ^ 2 - (1 - δ ^ 2) * (a * (a + 2 * b + c)) =
        (a * δ ^ 2 + b) ^ 2 + (1 - δ ^ 2) * (a * (δ ^ 2 * a - c)) := by
    ring
  have h_one_minus_dsq_nn : 0 ≤ 1 - δ ^ 2 := by nlinarith [sq_nonneg δ, hδ1]
  have h_term2_nn : 0 ≤ (1 - δ ^ 2) * (a * (δ ^ 2 * a - c)) := by
    apply mul_nonneg h_one_minus_dsq_nn
    apply mul_nonneg ha_pos.le
    linarith
  have h_sos_nn :
      0 ≤ (a + b) ^ 2 - (1 - δ ^ 2) * (a * (a + 2 * b + c)) := by
    rw [h_sos]
    exact add_nonneg (sq_nonneg _) h_term2_nn
  have h_cross_mul :
      (1 - δ ^ 2) * (a * (a + 2 * b + c)) ≤ (a + b) ^ 2 := by linarith
  -- Cosine: (a + b) / (‖v‖ · ‖v + ε‖).
  have h_denom_sq : (‖v‖ * ‖v + ε‖) ^ 2 = a * (a + 2 * b + c) := by
    have : (‖v‖ * ‖v + ε‖) ^ 2 = ‖v‖ ^ 2 * ‖v + ε‖ ^ 2 := by ring
    rw [this, h_norm_sq, ha_def]
  have h_cos_eq : cosSim v (v + ε) = (a + b) / (‖v‖ * ‖v + ε‖) := by
    show inner ℝ v (v + ε) / (‖v‖ * ‖v + ε‖) = (a + b) / (‖v‖ * ‖v + ε‖)
    rw [h_inner]
  -- cosSim² ≥ 1 - δ².
  have h_cos_sq : (cosSim v (v + ε)) ^ 2 = (a + b) ^ 2 / (a * (a + 2 * b + c)) := by
    rw [h_cos_eq, div_pow, h_denom_sq]
  have h_denom_sq_pos : 0 < a * (a + 2 * b + c) := by
    rw [← h_norm_sq]
    exact mul_pos ha_pos (pow_pos h_norm_pos 2)
  have h_cos_sq_lb : 1 - δ ^ 2 ≤ (cosSim v (v + ε)) ^ 2 := by
    rw [h_cos_sq, le_div_iff₀ h_denom_sq_pos]
    linarith
  -- Take square roots:  √(1 - δ²) ≤ √(cosSim²) = |cosSim| = cosSim.
  have h_step :
      Real.sqrt (1 - δ ^ 2) ≤ Real.sqrt ((cosSim v (v + ε)) ^ 2) :=
    Real.sqrt_le_sqrt h_cos_sq_lb
  rwa [Real.sqrt_sq h_cos_pos.le] at h_step



-- The n-hop accumulation bound (Wilkinson `(1+δ)^n − 1`) was
-- previously deferred from this file on the grounds that real per-step
-- error is structured and the worst-case product is loose.  That
-- argument still holds for honest empirical measurement (see paper
-- §5.6), but the runtime spile suite at `experiments/spiles/` made
-- the per-step floor load-bearing in 1,301 property tests, and any
-- multi-resolution lattice composes channels.  So the worst-case
-- bound is now mechanized in `proofs/SpileLattice.lean`:
--   • `cumulative_perturbation_bound`   -- ‖∏Tᵢ v − v‖ ≤ Δ‖v‖,
--                                          Δ = ∏(1+δᵢ) − 1
--   • `chained_operational_floor_bound` -- cosSim ≥ (1−Δ)/(1+Δ) for Δ < 1
--   • `chained_operational_floor_two_delta`
--   • `chained_operational_floor_uniform` (all δᵢ equal)
-- All four are no-axiom theorems built from the per-step floor here.

-- ════════════════════════════════════════════════════════════
-- §8. WHAT THE OPERATIONAL FLOOR SAYS — AND WHAT IT DOES NOT
--
-- The operational floor (this section) bounds cosine loss from
-- *bounded additive perturbation* — the hardware contribution
-- to per-step deviation.  It is precision-dependent: shrink δ
-- and the floor rises toward 1.
--
-- §3–§5 (qualitative strict loss) covers the *non-identity*
-- contribution — any transformation that does not preserve the
-- ray of v.  Argmax onto a basis vector is the canonical
-- example, and that loss is precision-independent: it persists
-- even if all arithmetic is exact.  The empirical finding of
-- the paper (Exp 10) is that for current models at fp16 the
-- argmax-style loss dominates the perturbation loss, which is
-- the formal counterpart of "argmax is a bigger knife than
-- rounding."  The corresponding empirical decomposition is in
-- paper §5.6.
-- ════════════════════════════════════════════════════════════

end Schemen.EnglishTax
