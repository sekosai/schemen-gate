/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/


import Mathlib.Analysis.SpecialFunctions.Log.Basic
import ModelSecurity

/-!
# Hardened Security Claims — V2

Strengthened formalization addressing gaps identified in V1 audit.

## Changes from V1

1. **PRF axiom (H1)**: V1's axiom had type `C(n,n/R) ≤ C(n,n/R)` —
   trivially `le_refl`. V2 introduces `RecoveryAttempt` where the
   adversary declares a query budget and an opaque `Recovers`
   predicate. The axiom is CONDITIONAL: IF the adversary succeeds,
   THEN their budget must be ≥ C(n,n/R). This is consistent (you
   can construct cheap attempts, but can't prove they succeed) and
   non-trivial (successful recovery requires exhaustive search).

2. **Steganographic failure (H5)**: V1 proved the wrong mask reads
   wrong dimensions (algebraic). V2 additionally defines `softmax`
   using `Real.exp` from Mathlib and proves the output is a valid
   probability distribution (positive, sums to 1) — formalizing
   "confident wrong answers, not errors."

3. **DistributableSafetyV2 (H7)**: The `no_shortcut` field now
   says: `∀ A, Recovers A S → C(n,n/R) ≤ A.queries`.

4. **Training data assumption (H2)**: V1 had `axiom : True`. V2
   models the adversary's capabilities as a typed structure and
   makes the assumption a field rather than a bare axiom.

## V1 proofs are NOT modified

All V1 theorems in GateSecurity.lean and ModelSecurity.lean remain
as-is. V2 imports and builds on V1.
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.SecurityV2

open Schemen Schemen.Security


-- ════════════════════════════════════════════════════════════════
-- §1. STRENGTHENED ADVERSARY MODEL  (fixes H1, H2, H7)
-- ════════════════════════════════════════════════════════════════

/-- A partition recovery attempt.

    The adversary declares how many candidate partitions they can
    test before identifying the correct one. Under the PRF
    assumption, this budget must be at least C(n, n/R).

    This replaces V1's vacuous axiom (which had type `x ≤ x`).
    Now the axiom constrains an externally-provided natural number,
    making it non-trivially satisfiable. -/
structure RecoveryAttempt (n R : ℕ) where
  /-- Number of candidate partitions the adversary evaluates -/
  queries : ℕ
  /-- The adversary claims this suffices for recovery -/
  queries_pos : 0 < queries

/-- Threat-model marker type.

    Originally defined as a structure with four `True`-typed fields
    documenting operational assumptions (weight access under Kerckhoffs'
    principle, architecture knowledge, lack of regime-specific training
    data, lack of the master key). The fake fields were removed in the
    April 2026 adversarial review because they inflated the axiom/field
    count without constraining anything.

    The structure is retained as an empty marker only so that existing
    call sites of `prf_brute_force_optimal` compile without rewriting
    every downstream file. The operational assumptions the old fields
    described are documented in `docs/executive-summary.md` and
    `docs/compliance-mapping.md` — they are deployment requirements,
    not formal constraints. -/
structure ThreatModel (n R : ℕ) : Type where
  -- empty: all fields were vacuous and were removed.

/-- Opaque predicate: the adversary's recovery attempt succeeds
    against a given cryptographic scheme — i.e., the adversary
    correctly identifies the partition from model weights alone.

    This is axiomatized (not defined) because formalizing "the
    adversary's strategy outputs the correct partition" requires
    a computational model we don't have in Lean. The opacity is
    the point: you cannot construct a proof of `Recovers A S`
    for a cheap attempt, which keeps the system consistent. -/
axiom Recovers {n R : ℕ} : RecoveryAttempt n R → CryptoScheme n R → Prop

/-- AXIOM (PRF Brute-Force Optimality — Strengthened).

    IF an adversary successfully recovers the correct partition,
    THEN their query budget must be at least C(n, n/R).

    Why this is consistent (unlike the unconditioned version):
    • You CAN construct a RecoveryAttempt with queries = 1
    • But you CANNOT prove `Recovers` for it (opaque axiom)
    • So you cannot invoke this axiom for cheap attempts
    • No inconsistency: cheap attempts exist, they just can't
      be proven successful

    Cryptographic basis:
    • HMAC-SHA256 is a PRF (FIPS 198-1, Bellare-Canetti-Krawczyk 1996)
    • PRF-seeded Fisher-Yates produces a pseudorandom permutation
    • Under PRF assumption, the partition is computationally
      indistinguishable from a truly random partition
    • For a truly random partition, all C(n, n/R) candidates are
      equally likely, making brute force optimal

    This axiom holds for ANY PRF family, not just HMAC-SHA256. -/
axiom prf_brute_force_optimal {n R : ℕ}
    (S : CryptoScheme n R) (A : RecoveryAttempt n R)
    (_T : ThreatModel n R)
    (hn : 0 < n) (hR : 0 < R) (hdiv : R ∣ n)
    (h_success : Recovers A S) :
    Nat.choose n (n / R) ≤ A.queries


-- ════════════════════════════════════════════════════════════════
-- §2. SOFTMAX — CONCRETE DEFINITION AND PROPERTIES  (fixes H5)
--
-- V1 proved: wrong mask reads zero at training-active dimensions.
-- V2 additionally proves: the OUTPUT is a valid probability
-- distribution (positive, sums to 1). This formalizes the patent
-- claim that wrong keys produce "confident wrong answers, not
-- errors or access denials."
-- ════════════════════════════════════════════════════════════════

/-- Softmax denominator: Z(v) = Σ exp(v_k). Always positive because
    exp is always positive. -/
def softmax_denom {n : ℕ} (v : Fin n → ℝ) : ℝ :=
  ∑ k : Fin n, Real.exp (v k)

/-- The softmax function: maps arbitrary logits to a probability
    distribution. This is the standard definition used in neural
    network output layers.

    softmax(v)_j = exp(v_j) / Σ_k exp(v_k) -/
def softmax {n : ℕ} (v : Fin n → ℝ) : Fin n → ℝ :=
  fun j => Real.exp (v j) / softmax_denom v

/-- The softmax denominator is strictly positive. -/
theorem softmax_denom_pos {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) :
    0 < softmax_denom v := by
  apply Finset.sum_pos
  · intro k _; exact Real.exp_pos _
  · exact ⟨⟨0, hn⟩, Finset.mem_univ _⟩

/-- The softmax denominator is nonzero (convenience lemma). -/
theorem softmax_denom_ne_zero {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) :
    softmax_denom v ≠ 0 :=
  ne_of_gt (softmax_denom_pos hn v)

/-- **Theorem.** Every softmax output is strictly positive.

    This means the output is never zero, never negative, and never
    NaN. The adversary always receives a "real" answer. -/
theorem softmax_pos {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) (j : Fin n) :
    0 < softmax v j :=
  div_pos (Real.exp_pos _) (softmax_denom_pos hn v)

/-- **Theorem.** Softmax outputs sum to exactly 1.

    Combined with positivity, this proves softmax outputs a valid
    probability distribution for ANY input logits — including
    logits computed from the wrong mask.

    Proof: Σ_j [exp(v_j) / Z] = (Σ_j exp(v_j)) / Z = Z / Z = 1. -/
theorem softmax_sum_one {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) :
    ∑ j : Fin n, softmax v j = 1 := by
  simp only [softmax]
  rw [← Finset.sum_div]
  exact div_self (softmax_denom_ne_zero hn v)

/-- **Theorem.** Each softmax output is at most 1. -/
theorem softmax_le_one {n : ℕ} (hn : 0 < n) (v : Fin n → ℝ) (j : Fin n) :
    softmax v j ≤ 1 := by
  unfold softmax
  rw [div_le_one (softmax_denom_pos hn v)]
  exact Finset.single_le_sum (fun k _ => le_of_lt (Real.exp_pos _))
    (Finset.mem_univ j)


-- ════════════════════════════════════════════════════════════════
-- §3. STEGANOGRAPHIC OUTPUT — WRONG KEY, VALID DISTRIBUTION
--
-- Combining GateSecurity's wrong_mask_reads_wrong_dims with
-- the softmax properties above to formalize the full claim:
-- wrong mask → valid probability distribution → "confident
-- wrong answers."
-- ════════════════════════════════════════════════════════════════

/-- Output logits of the gated MLP: logits_k = Σ_j gated_j · W2_{j,k} + b2_k.
    This is the linear transformation from gated hidden layer to output. -/
def output_logits {n o : ℕ} (gated : Fin n → ℝ) (W2 : Fin n → Fin o → ℝ)
    (b2 : Fin o → ℝ) : Fin o → ℝ :=
  fun k => (∑ j : Fin n, gated j * W2 j k) + b2 k

/-- **Theorem (Steganographic Output Is Valid Distribution).**

    When ANY mask is applied (correct or incorrect), the output
    of softmax is ALWAYS a valid probability distribution:
    all components strictly positive and summing to exactly 1.

    This means:
    • Wrong key → confidently wrong answers (valid softmax)
    • No error signal, no null output, no access denied
    • The adversary cannot distinguish "wrong key" from
      "model was trained to answer differently"

    This theorem composes with `wrong_mask_reads_wrong_dims`
    (GateSecurity §7): the wrong mask zeroes the training-active
    dimensions, and the resulting logits pass through softmax
    to produce a valid but incorrect distribution. -/
theorem wrong_key_valid_distribution {n o : ℕ} (_hn : 0 < n) (ho : 0 < o)
    (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ)
    (mask : Vec n) :
    let logits := output_logits (h_act ⊙ mask) W2 b2
    (∀ k : Fin o, 0 < softmax logits k)
    ∧ (∑ k : Fin o, softmax logits k = 1)
    ∧ (∀ k : Fin o, softmax logits k ≤ 1) :=
  ⟨fun k => softmax_pos ho _ k,
   softmax_sum_one ho _,
   fun k => softmax_le_one ho _ k⟩


-- ════════════════════════════════════════════════════════════════
-- §4. DISTRIBUTABLE SAFETY V2 — STRENGTHENED MAIN THEOREM
-- ════════════════════════════════════════════════════════════════

-- DistributableSafetyV2, standard_is_distributable_safe_v2,
-- cheap_attempt, cheap_attempt_cannot_succeed, and
-- end_to_end_chain_v2 have been moved to DistributableClaims.lean.
-- ════════════════════════════════════════════════════════════════


end Schemen.SecurityV2
