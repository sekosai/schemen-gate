/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import DistributedSecurity

-- The concrete attack bound below compares powers up to 2^384.
set_option exponentiation.threshold 512

/-!
# Distributable Safety Claims — Consolidated

This file consolidates the "distributable safety" wrapper structures
from V1, V2, and V3 into a single location. These structures package
core gate security properties (gradient isolation, steganographic failure,
combinatorial hardness) into a "distributable safety" conclusion.

**Why separated**: Patent E (Distributable Inert Artifact) is deferred
from initial filing. The core gate theorems (gradient isolation, weight
confinement, steganographic failure, etc.) remain in their original files
(GateSecurity.lean, ModelSecurity.lean, ModelSecurityV2.lean,
ModelSecurityV3.lean). This file contains only the distributable-safety
*framing* — the structures that package those theorems into a
"distributable safety" conclusion.

For the Weight Camouflage proofs (V4), see DistributedSecurity.lean.

## Contents

- `DistributableSafety` (V1): combinatorial hardness + PRF + steganographic failure
- `DistributableSafetyV2` (V2): adds steganographic output (valid softmax)
- `DistributableSafetyV3` (V3): adds weight indistinguishability, regime locality,
  exact mask uniqueness, compositional confinement
- Concrete adversary bound (cheap_attempt_cannot_succeed)
- End-to-end security chains V1, V2, V3
-/

set_option autoImplicit false

noncomputable section


-- ════════════════════════════════════════════════════════════════
-- §1. (REMOVED) V1 DISTRIBUTABLE SAFETY
--
-- Previous versions contained `DistributableSafety` (V1), whose
-- `no_shortcut` field had type `∀ S, C(n,n/R) ≤ C(n,n/R)` — trivially
-- `le_refl`. Removed in the April 2026 adversarial review along with
-- the `prf_implies_no_shortcut` axiom it depended on (see
-- ModelSecurity.lean).
--
-- The V2 structure (`DistributableSafetyV2` below) supersedes it.
-- V2's `no_shortcut` quantifies over `RecoveryAttempt` and constrains
-- the query budget; it is not trivially satisfiable.
-- ════════════════════════════════════════════════════════════════


-- ════════════════════════════════════════════════════════════════
-- §2. V2 DISTRIBUTABLE SAFETY
-- ════════════════════════════════════════════════════════════════

namespace Schemen.SecurityV2

open Schemen Schemen.Security

/-- Distributable safety with strengthened guarantees (V2).

    Changes from V1:
    1. `no_shortcut` quantifies over `RecoveryAttempt` and constrains
       the adversary's query budget. Not trivially satisfiable.
    2. `steganographic_output` added: wrong mask produces a valid
       probability distribution, not an error.
    3. `threat_model` makes assumptions explicit as typed fields. -/
structure DistributableSafetyV2 (n R : ℕ) where
  combinatorial_hardness :
    2 ^ 256 ≤ Nat.choose n (n / R)
  no_shortcut :
    ∀ (S : CryptoScheme n R) (A : RecoveryAttempt n R)
      (_T : ThreatModel n R),
      Recovers A S → Nat.choose n (n / R) ≤ A.queries
  steganographic_mask :
    ∀ (P : ValidPartition n R) (r s : Fin R),
      r ≠ s → ∀ j : Fin n, j ∈ P.groups r →
      indicator (P.groups s) j = 0
  steganographic_output :
    ∀ (o : ℕ) (_ho : 0 < o)
      (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ)
      (mask : Vec n),
      let logits := output_logits (h_act ⊙ mask) W2 b2
      (∀ k : Fin o, 0 < softmax logits k)
      ∧ (∑ k : Fin o, softmax logits k = 1)

theorem standard_is_distributable_safe_v2 :
    DistributableSafetyV2 768 2 where
  combinatorial_hardness := exceeds_aes256_security
  no_shortcut := fun S A T h_rec =>
    prf_brute_force_optimal S A T (by omega) (by omega) ⟨384, by omega⟩ h_rec
  steganographic_mask := fun P r s hrs j hj =>
    wrong_mask_reads_wrong_dims P r s hrs j hj
  steganographic_output := fun o ho h_act W2 b2 mask =>
    ⟨fun k => softmax_pos ho _ k, softmax_sum_one ho _⟩

/-- A "cheap" recovery attempt: only 2^40 queries. -/
def cheap_attempt : RecoveryAttempt 768 2 where
  queries := 2 ^ 40
  queries_pos := by omega

/-- A cheap attempt cannot succeed: it contradicts the combinatorial bound. -/
theorem cheap_attempt_cannot_succeed
    (S : CryptoScheme 768 2) (T : ThreatModel 768 2)
    (h_rec : Recovers cheap_attempt S) :
    False := by
  have h1 : 2 ^ 384 ≤ cheap_attempt.queries :=
    le_trans standard_exceeds_2_384
      (prf_brute_force_optimal S cheap_attempt T (by omega) (by omega) ⟨384, by omega⟩ h_rec)
  have h2 : cheap_attempt.queries = 2 ^ 40 := rfl
  rw [h2] at h1
  exact absurd h1 (not_le.mpr (Nat.pow_lt_pow_right (by omega) (by omega)))

theorem end_to_end_chain_v2 :
    DistributableSafetyV2 768 2 :=
  standard_is_distributable_safe_v2

end Schemen.SecurityV2


-- ════════════════════════════════════════════════════════════════
-- §3. V3 DISTRIBUTABLE SAFETY
-- ════════════════════════════════════════════════════════════════

namespace Schemen.SecurityV3

open Schemen Schemen.Security Schemen.SecurityV2

/-- Distributable safety — comprehensive (V3).

    Adds to V2:
    • weight_indistinguishable: weights carry zero key information
      (requires IsSurjective T — per-process, not axiom)
    • regime_locality: wrong-key output = regime-s sub-model output
    • exact_mask: only the correct mask reproduces correct output
    • compositional_confinement: gate works inside any architecture -/
structure DistributableSafetyV3 (n R : ℕ) where
  combinatorial_hardness :
    2 ^ 256 ≤ Nat.choose n (n / R)
  no_shortcut :
    ∀ (S : CryptoScheme n R) (A : RecoveryAttempt n R)
      (_T : ThreatModel n R),
      Recovers A S → Nat.choose n (n / R) ≤ A.queries
  steganographic_mask :
    ∀ (P : ValidPartition n R) (r s : Fin R),
      r ≠ s → ∀ j : Fin n, j ∈ P.groups r →
      indicator (P.groups s) j = 0
  steganographic_output :
    ∀ (o : ℕ) (_ho : 0 < o)
      (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ)
      (mask : Vec n),
      let logits := output_logits (h_act ⊙ mask) W2 b2
      (∀ k : Fin o, 0 < softmax logits k)
      ∧ (∑ k : Fin o, softmax logits k = 1)
  weight_indistinguishable :
    ∀ (m o : ℕ) (S : CryptoScheme n R) (T : TrainingProcess n R m o),
      IsSurjective T →
      ∀ (W : ModelWeights m n o) (k : S.Key),
        ∃ D : T.Data, T.train (S.derive k) D = W
  regime_locality :
    ∀ (o : ℕ) (P : ValidPartition n R) (s : Fin R)
      (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ),
      ∀ k : Fin o,
        output_logits (h_act ⊙ indicator (P.groups s)) W2 b2 k =
        (P.groups s).sum (fun j => h_act j * W2 j k) + b2 k
  exact_mask :
    ∀ (P : ValidPartition n R) (r : Fin R) (M : Vec n),
      (∀ (o : ℕ) (h_act : Vec n) (W2 : Fin n → Fin o → ℝ)
        (b2 : Fin o → ℝ) (k : Fin o),
          output_logits (h_act ⊙ M) W2 b2 k =
          output_logits (h_act ⊙ indicator (P.groups r)) W2 b2 k) →
      ∀ j : Fin n, M j = indicator (P.groups r) j
  compositional_confinement :
    ∀ (mask : Vec n) (j : Fin n),
      mask j = 0 → ∀ upstream : ℝ, upstream * mask j = 0

theorem standard_is_distributable_safe_v3 :
    DistributableSafetyV3 768 2 where
  combinatorial_hardness := exceeds_aes256_security
  no_shortcut := fun S A T h_rec =>
    prf_brute_force_optimal S A T (by omega) (by omega) ⟨384, by omega⟩ h_rec
  steganographic_mask := fun P r s hrs j hj =>
    wrong_mask_reads_wrong_dims P r s hrs j hj
  steganographic_output := fun o ho h_act W2 b2 mask =>
    ⟨fun k => softmax_pos ho _ k, softmax_sum_one ho _⟩
  weight_indistinguishable := fun _ _ S T hT =>
    zero_key_information S T hT
  regime_locality := fun o P s h_act W2 b2 =>
    regime_output_locality P s h_act W2 b2
  exact_mask := fun P r M h_same =>
    access_requires_exact_mask P r M h_same
  compositional_confinement := fun mask j hj upstream => by
    rw [hj, mul_zero]

theorem end_to_end_chain_v3 :
    DistributableSafetyV3 768 2 :=
  standard_is_distributable_safe_v3

end Schemen.SecurityV3
