/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/


import ModelSecurityV2

/-!
# V3 Security Proofs — Weight Indistinguishability, Regime Equivalence,
# Transformer Compositionality

Addresses the four major gaps identified in the meta-audit:

1. **Weight Partition Indistinguishability (§1)**: For surjective training processes,
   the model weights carry zero information about the cryptographic key.

2. **Steganographic Regime Equivalence (§2)**: The wrong-key output is not
   "broken" — it is the output of a fully functional sub-model for a different
   regime. The sum over all dimensions collapses to a sum over the regime's
   partition.

2B. **Exact Mask Uniqueness (§2B)**: The ONLY mask that universally reproduces
    regime r's correct output is exactly indicator(groups(r)). Any other mask —
    subset, superset, partial overlap, or arbitrary — produces different logits
    whenever the differing dimensions carry nonzero signal.

3. **Transformer / Post-Residual Compositionality (§3)**: The gate confinement
   proofs hold inside residual blocks because the gate is a multiplicative
   bottleneck — any upstream gradient multiplied by mask[j]=0 is zero.

## New hypotheses

One new per-process hypothesis: `IsSurjective T`. For a given training process T,
any target weight configuration is reachable from any valid partition with
appropriate training data. This must be established for each concrete training
process — it is NOT assumed universally, avoiding the inconsistency that a bare
axiom would introduce (a trivial training process that always returns zero weights
would falsify a universal surjectivity axiom).

## Proof architecture

```
  V1 (GateSecurity):    a·0=0 chain for gradients/weights
  V1 (ModelSecurity):   combinatorial bounds, PRF axiom (deprecated)
  V2 (ModelSecurityV2): strengthened PRF, softmax validity
  V3 (this file):       weight indistinguishability
                         + regime equivalence
                         + compositionality
       │
       ├── IsSurjective T (hypothesis) → zero_key_information
       ├── regime_output_locality → steganographic equivalence
       ├── mask_decomposition → wrong_mask_corrupts → access_requires_exact_mask
       └── gate_confinement_composes → transformer compatibility
```

## Complete axiom inventory (after V3, with V4 scope note and April 2026 cleanup)

| Axiom / Hypothesis | Source | Status |
|---|---|---|
| `prf_brute_force_optimal` | V2 | Active — standard cryptographic (PRF) assumption |
| `Recovers` | V2 | Active — structural opaque predicate for adversary success |
| `IsSurjective T` | V3 | Active but unrealistic for SGD — see `zero_key_information` scope note. Retained for theorem `zero_key_information`, which is correct under its hypothesis but should not be cited in external claims |
| `IsDistributionMatched` | V4.2 (DistributedSecurity) | Active — opaque predicate for weight-camouflage distributional match |
| `camouflage_indistinguishable` | V4.2 (DistributedSecurity) | Active — conditioned on `IsDistributionMatched` |
| `gradient_probing_hard` | V4.2 (DistributedSecurity) | Active — conditioned on `IsDistributionMatched` |
| `prf_implies_no_shortcut` | V1 | **REMOVED** (April 2026) — tautology `C(n,n/R) ≤ C(n,n/R)` |
| `training_data_private` | V1 | **REMOVED** (April 2026) — was `axiom : True` |

The checked-in graph contains five project-specific `axiom` declarations:
one substantive PRF assumption, two opaque predicate declarations
(`Recovers`, `IsDistributionMatched`), and two historical statistical
consequences conditioned on `IsDistributionMatched`. The latter two remain
inspectable but are excluded from the paper's claim set. Two per-process hypotheses
(`IsSurjective T` retained for historical completeness;
`PartitionOblivious T` in V4 is the realistic replacement). Everything else
is machine-checked.
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.SecurityV3

open Schemen Schemen.Security Schemen.SecurityV2


-- ════════════════════════════════════════════════════════════════
-- §1. WEIGHT PARTITION INDISTINGUISHABILITY
--
-- "An adversary holding the model weights cannot determine
--  which partition (and therefore which key) was used."
--
-- We introduce a TrainingProcess abstraction and define
-- IsSurjective as a per-process hypothesis: any weight
-- configuration is reachable from any partition given
-- appropriate training data.
-- ════════════════════════════════════════════════════════════════

/-- Model weights: the W₁ (input→hidden) and W₂ (hidden→output) matrices.
    This is the observable artifact the adversary holds. -/
def ModelWeights (m n o : ℕ) := (Fin m → Fin n → ℝ) × (Fin n → Fin o → ℝ)

/-- A training process maps (partition, training data) to final model weights.
    This abstracts over the optimizer, learning rate, number of steps, etc.
    The only structural requirement is determinism: same inputs → same weights. -/
structure TrainingProcess (n R m o : ℕ) where
  Data : Type
  train : ValidPartition n R → Data → ModelWeights m n o

/-- HYPOTHESIS (Weight Space Surjectivity).

    A training process T is surjective if, for any target weight
    configuration W and any valid partition P, there exists training
    data D such that training with (P, D) produces exactly W.

    This is a PER-PROCESS HYPOTHESIS, not a universal axiom. It must
    be established for each concrete training process. A trivial
    training process (e.g., one that always returns zero weights)
    would NOT satisfy this predicate, and correctly so — weight
    indistinguishability only holds for training processes with
    sufficiently rich optimization landscapes.

    Why this is NOT an axiom:
    A bare `axiom weight_surjectivity (T : TrainingProcess ...)` would
    universally quantify over ALL training processes, including trivial
    ones that provably cannot reach all weight configurations. That
    would introduce an inconsistency (derive 0 = 1 via a constant-zero
    training process). Making it a hypothesis avoids this: the caller
    must demonstrate surjectivity for their specific T.

    Justification (for realistic SGD-based training):
    • Each column j of W₁ is trained exclusively by the regime
      owning dimension j (by `weight_update_confined`).
    • Each row j of W₂ likewise (by `w2_update_confined`).
    • These columns/rows are trained independently of each other.
    • For sufficiently rich training data and a capable optimizer,
      any target weight vector for a single column/row is reachable.

    The hypothesis is strictly about REACHABILITY, not about the
    probability of reaching a particular configuration. It says
    the set of achievable weights is the entire weight space,
    not that any particular weight is likely. -/
def IsSurjective {n R m o : ℕ} (T : TrainingProcess n R m o) : Prop :=
  ∀ (W : ModelWeights m n o) (P : ValidPartition n R),
    ∃ D : T.Data, T.train P D = W

/-- **Theorem (Zero Key Information) — ⚠ HYPOTHESIS IS UNREALISTIC FOR SGD.**
    For a surjective training process T, any observed weight matrix W is
    consistent with EVERY possible key k.

    Proof: for any key k, `S.derive k` produces a valid partition. By
    surjectivity of T, there exists training data D such that training with
    that partition and D produces W. Consequence:
    `∀ k, ∃ D, train(derive(k), D) = W`. Under surjectivity, the adversary's
    posterior over keys given weights equals their prior.

    **⚠ Scope / honesty note (added April 2026).** The `IsSurjective T`
    hypothesis states that any weight configuration is reachable from any
    partition. This is **false for realistic SGD-based training**: training
    trajectories form a measure-zero manifold in weight space, so the image
    of `T.train P D` over all `D` is not the full weight space.

    This theorem is kept here as a correct mathematical result under its
    stated hypothesis, not as a security claim about deployed systems. **Do
    not cite `zero_key_information` in marketing, compliance submissions, or
    patent specifications.** The public cryptographic claim Schemen makes is
    an informal reduction to the PRF assumption — see
    `proofs/ModelSecurityV4.lean` for the type-level scaffolding and
    `docs/executive-summary.md` for the operational statement. -/
theorem zero_key_information {n R m o : ℕ}
    (S : CryptoScheme n R) (T : TrainingProcess n R m o)
    (hT : IsSurjective T)
    (W : ModelWeights m n o) :
    ∀ k : S.Key, ∃ D : T.Data, T.train (S.derive k) D = W :=
  fun k => hT W (S.derive k)

/-- **Theorem (Partition Indistinguishability).**
    For any two keys k₁ and k₂, the observed weights W are
    consistent with BOTH partitions simultaneously.

    The adversary cannot determine whether k₁ or k₂ was used
    by examining the weights, because both are equally consistent
    with the observation. -/
theorem partition_indistinguishable {n R m o : ℕ}
    (S : CryptoScheme n R) (T : TrainingProcess n R m o)
    (hT : IsSurjective T)
    (W : ModelWeights m n o) (k₁ k₂ : S.Key) :
    (∃ D, T.train (S.derive k₁) D = W) ∧
    (∃ D, T.train (S.derive k₂) D = W) :=
  ⟨zero_key_information S T hT W k₁, zero_key_information S T hT W k₂⟩


-- ════════════════════════════════════════════════════════════════
-- §2. STEGANOGRAPHIC REGIME EQUIVALENCE
--
-- "The wrong-key output is not broken — it is the output of
--  regime s's sub-model, a fully functional model for a
--  different task."
--
-- We prove that the output with mask M_s depends ONLY on
-- the dimensions in groups(s). The full sum over Fin n
-- collapses to a sum over groups(s).
-- ════════════════════════════════════════════════════════════════

/-- Helper: summing f(j) · indicator(S)(j) over all j equals
    summing f(j) over just the elements of S.

    This is the algebraic core of regime equivalence:
    the indicator mask zeroes all terms outside S, leaving
    only the in-group contributions. -/
lemma sum_mul_indicator_eq {n : ℕ} (S : Finset (Fin n)) (f : Fin n → ℝ) :
    ∑ j : Fin n, f j * indicator S j = S.sum f := by
  symm
  calc S.sum f
      = S.sum (fun j => f j * indicator S j) := by
        apply Finset.sum_congr rfl; intro j hj
        rw [indicator_mem S j hj, mul_one]
    _ = ∑ j : Fin n, f j * indicator S j := by
        apply Finset.sum_subset (Finset.subset_univ S)
        intro j _ hj
        rw [indicator_not_mem S j hj, mul_zero]

/-- **Theorem (Regime Output Locality).**
    The output logits with mask M_s depend ONLY on dimensions
    in groups(s). The full sum over all n hidden dimensions
    collapses to a sum over just the |groups(s)| = n/R
    dimensions in the regime's partition.

    This means the adversary with mask M_s is running regime s's
    sub-model — a fully functional, correctly trained model for
    regime s's data. The output is not "broken" or "random";
    it is a legitimate model giving real answers for a different
    task.

    Combined with V2's `wrong_key_valid_distribution`: the output
    is a valid probability distribution (positive, sums to 1)
    that reflects regime s's learned knowledge. -/
theorem regime_output_locality {n o R : ℕ}
    (P : ValidPartition n R) (s : Fin R)
    (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ) :
    ∀ k : Fin o,
      output_logits (h_act ⊙ indicator (P.groups s)) W2 b2 k =
      (P.groups s).sum (fun j => h_act j * W2 j k) + b2 k := by
  intro k
  simp only [output_logits, hmul]
  congr 1
  simp_rw [show ∀ j, h_act j * indicator (P.groups s) j * W2 j k =
    (h_act j * W2 j k) * indicator (P.groups s) j from fun j => by ring]
  exact sum_mul_indicator_eq (P.groups s) (fun j => h_act j * W2 j k)

/-- **Theorem (Regime Independence).**
    If two hidden-layer activations agree on regime s's dimensions,
    the regime-s output is identical — regardless of what happens
    in other dimensions.

    This is the formal statement that regime s's output is a
    PURE FUNCTION of regime s's features. Modifying, randomizing,
    or zeroing the other dimensions has no effect on the output
    when mask M_s is applied.

    Together with `regime_output_locality`, this proves the
    steganographic equivalence: the wrong-key model IS regime s's
    sub-model, reading exclusively from regime s's learned features. -/
theorem regime_output_independent_of_others {n o R : ℕ}
    (P : ValidPartition n R) (s : Fin R)
    (h_act₁ h_act₂ : Vec n)
    (h_agree : ∀ j : Fin n, j ∈ P.groups s → h_act₁ j = h_act₂ j)
    (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ) :
    ∀ k : Fin o,
      output_logits (h_act₁ ⊙ indicator (P.groups s)) W2 b2 k =
      output_logits (h_act₂ ⊙ indicator (P.groups s)) W2 b2 k := by
  intro k
  simp only [output_logits, hmul]
  congr 1
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j ∈ P.groups s
  · rw [indicator_mem _ j hj]; simp only [mul_one]; rw [h_agree j hj]
  · rw [indicator_not_mem _ j hj]; simp only [mul_zero, zero_mul]


-- ════════════════════════════════════════════════════════════════
-- §2B. ONLY THE EXACT CORRECT MASK WORKS
--
-- "You don't get access to regimes you don't have keys to."
--
-- The preceding §§1-2 prove:
-- • Wrong regime mask → reads wrong dims (GateSecurity §7)
-- • Regime output depends only on the masked dims (§2 above)
--
-- But those results only cover complete regime masks (M_s where s≠r)
-- and the all-1s superset mask (V4 §3). They do NOT cover:
-- • Arbitrary masks with partial overlap
-- • Subset masks (missing some of groups(r)'s dimensions)
--
-- This section closes the gap with a GENERAL theorem: for ANY
-- mask M ≠ indicator(groups(r)), the output logits differ from
-- the correct output, provided the differing dimensions carry
-- nonzero signal. This captures all cases — superset, subset,
-- partial overlap, arbitrary construction — in a single result.
--
-- Together with weight confinement (groups(r) holds the knowledge)
-- and the PRF axiom (constructing the mask requires the key),
-- this completes the formal proof of the key claim.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (General Mask Decomposition).**
    For any two masks M₁ and M₂, the logit difference decomposes as
    the sum of per-dimension contributions weighted by (M₁[j] - M₂[j]).

    At dimensions where the masks agree: contribution = 0.
    At dimensions where M₁ has an extra 1: adds h[j] · W2[j,k].
    At dimensions where M₁ has a missing 1: subtracts h[j] · W2[j,k].

    When M₁ = all_ones and M₂ = indicator(S), this reduces to
    V4's `all_ones_logit_decomposition`. When M₁ is a different
    regime mask, the differing dims are exactly the symmetric
    difference of the two regimes. -/
theorem mask_decomposition {n o : ℕ}
    (h_act : Vec n) (M₁ M₂ : Vec n)
    (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ) (k : Fin o) :
    output_logits (h_act ⊙ M₁) W2 b2 k =
    output_logits (h_act ⊙ M₂) W2 b2 k +
    ∑ j : Fin n, h_act j * (M₁ j - M₂ j) * W2 j k := by
  simp only [output_logits, hmul]
  suffices h : ∑ j : Fin n, h_act j * M₁ j * W2 j k =
    (∑ j : Fin n, h_act j * M₂ j * W2 j k) +
    ∑ j : Fin n, h_act j * (M₁ j - M₂ j) * W2 j k by linarith
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro j _; ring

/-- **Theorem (Wrong Mask Corrupts Output).**
    If a mask M differs from the correct mask indicator(S), and the
    per-dimension difference contributes nonzero signal at output k,
    the logits at k differ from the correct output.

    This covers ALL cases:
    • Subset masks (missing dims): the missing terms subtract signal
    • Superset masks (extra dims): the extra terms add noise
    • Partial overlap: both effects combine
    • Arbitrary masks: same decomposition applies

    The hypothesis `h_diff` asks that the total contribution of
    all differing dimensions is nonzero at output k. For trained
    models with generic (non-degenerate) weights, this holds
    whenever the mask differs at any dimension with nonzero
    activation and nonzero W₂ row.

    This is the formal proof that ONLY the exact correct mask
    reproduces the correct output. Any deviation corrupts. -/
theorem wrong_mask_corrupts {n o : ℕ}
    (S : Finset (Fin n)) (M : Vec n)
    (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ)
    (k : Fin o)
    (h_diff : ∑ j : Fin n,
        h_act j * (M j - indicator S j) * W2 j k ≠ 0) :
    output_logits (h_act ⊙ M) W2 b2 k ≠
    output_logits (h_act ⊙ indicator S) W2 b2 k := by
  rw [mask_decomposition h_act M (indicator S) W2 b2 k]
  intro h_eq; exact h_diff (by linarith)

/-- **Theorem (Access Requires Exact Mask).**
    If a mask M produces the same output as indicator(groups(r))
    for ALL possible activations and ALL possible W₂ weight matrices,
    then M must agree with indicator(groups(r)) at every dimension.

    In other words: the ONLY mask that universally reproduces
    regime r's output is exactly indicator(groups(r)). There is
    no alternative mask, no approximation, no shortcut.

    Proof strategy: assume M j₀ ≠ indicator(S) j₀ for some j₀.
    Construct a specific activation (basis vector at j₀) and W₂
    matrix (identity-like at j₀) that isolates dimension j₀.
    The output logits reduce to M j₀ vs indicator(S) j₀, which
    differ by assumption. Contradiction.

    This is the formal proof of "you don't get access to regimes
    you don't have keys to." Combined with:
    • Weight confinement: regime r's knowledge lives in groups(r)
    • PRF axiom: constructing groups(r) requires the key
    The chain is: no key → can't identify groups(r) → can't
    construct indicator(groups(r)) → can't reproduce the output. -/
theorem access_requires_exact_mask {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (M : Vec n)
    (h_same : ∀ (o : ℕ) (h_act : Vec n) (W2 : Fin n → Fin o → ℝ)
      (b2 : Fin o → ℝ) (k : Fin o),
        output_logits (h_act ⊙ M) W2 b2 k =
        output_logits (h_act ⊙ indicator (P.groups r)) W2 b2 k) :
    ∀ j : Fin n, M j = indicator (P.groups r) j := by
  intro j₀
  by_contra h_ne
  have h_spec := h_same 1
    (fun j => if j = j₀ then 1 else 0)
    (fun j _ => if j = j₀ then 1 else 0)
    (fun _ => 0)
    ⟨0, Nat.one_pos⟩
  simp only [output_logits, hmul] at h_spec
  have collapse : ∀ (g : Fin n → ℝ),
      (∑ j : Fin n, (if j = j₀ then (1 : ℝ) else 0) * g j *
        (if j = j₀ then (1 : ℝ) else 0)) + 0 = g j₀ := by
    intro g
    rw [add_zero, Finset.sum_eq_single_of_mem j₀ (Finset.mem_univ _)]
    · simp
    · intro j _ hne; simp [if_neg hne]
  simp only [collapse] at h_spec
  exact h_ne h_spec


-- ════════════════════════════════════════════════════════════════
-- §3. TRANSFORMER / POST-RESIDUAL COMPOSITIONALITY
--
-- "The gate confinement proofs hold inside any architecture
--  that uses the gated MLP as a sublayer — including transformer
--  blocks with residual connections, attention, and LayerNorm."
--
-- Key insight: the existing weight_update_confined and
-- w2_update_confined proofs are parameterized by ARBITRARY
-- upstream gradients. They don't assume anything about where
-- those gradients come from. This section makes that
-- compositionality explicit and connects it to the
-- transformer architecture.
--
-- Architecture (transformer block):
--
--   input ─→ LN ─→ Attn ─→ (+) ─→ LN ─→ MLP ─→ (+) ─→ output
--            │               ↑                     ↑
--            └── residual ───┘                     │
--                              └──── residual ─────┘
--
-- The gate is applied INSIDE the MLP's hidden layer:
--   z = W1 @ input + b1
--   h = relu(z)
--   gated = h ⊙ mask    ← gate here
--   mlp_out = W2 @ gated + b2
--
-- The residual adds to the MLP's OUTPUT (outside the gate).
-- Gradient flow through the residual bypasses the MLP entirely.
-- Gradient flow through the MLP passes through the gate.
-- Therefore weight confinement holds regardless of the residual.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Multiplicative Bottleneck).**
    Any scalar multiplied by mask[j] = 0 is zero.

    This is the fundamental reason the gate confinement proofs
    compose with any architecture: the gate is a multiplicative
    bottleneck. Any upstream gradient — whether it arrives via a
    residual connection, attention layer, LayerNorm, or any other
    pathway — is zeroed when it passes through the gate at an
    inactive dimension.

    All weight confinement results (`gradient_isolation`,
    `weight_update_confined`, `w2_update_confined`) ultimately
    reduce to this fact. -/
theorem gate_confinement_composes {n : ℕ}
    (mask : Vec n) (j : Fin n) (hj : mask j = 0)
    (upstream : ℝ) :
    upstream * mask j = 0 := by
  rw [hj, mul_zero]


-- ── §3B. SUPPORT PRESERVATION — GATE PLACEMENT CONSTRAINTS ──────
--
-- Not all component placements are safe. The gate's W₁ confinement
-- depends on the backward function between the gate and the W₁
-- update preserving the zero structure: zeros at inactive dims
-- must remain zero after the backward pass.
--
-- We define "support preservation" and prove:
--   POSITIVE: support-preserving backward → W₁ confinement holds
--   NEGATIVE: non-preserving backward → W₁ confinement can break
--
-- Standard placement (act → gate → W₂) is safe because the
-- backward chain (relu_grad multiplication) preserves support.
--
-- LayerNorm backward does NOT preserve support: its mean
-- subtraction spreads gradient across ALL dimensions. Placing
-- LN between activation and gate would break W₁ confinement.
--
-- W₂ confinement is ALWAYS safe — it depends on the forward gate
-- (gated[j] = h[j]·mask[j] = 0), not on any backward path.
-- ─────────────────────────────────────────────────────────────────

/-- A function preserves support on S if: whenever the input is
    zero outside S, the output is also zero outside S.

    Captures: the function does not spread information from active
    dimensions to inactive dimensions.

    Preserves support: identity, pointwise ops with f(0)=0, ⊙.
    Does NOT preserve support: LayerNorm backward (mean subtraction
    mixes all dimensions). -/
def PreservesSupport {n : ℕ} (f : Vec n → Vec n) (S : Finset (Fin n)) : Prop :=
  ∀ (v : Vec n), (∀ j : Fin n, j ∉ S → v j = 0) →
    ∀ j : Fin n, j ∉ S → f v j = 0

/-- The identity preserves support (nothing between gate and W). -/
theorem id_preserves_support {n : ℕ} (S : Finset (Fin n)) :
    PreservesSupport id S :=
  fun _ h j hj => h j hj

/-- Pointwise application of g with g(0) = 0 preserves support.
    Covers relu backward: relu'(z) · 0 = 0. -/
theorem pointwise_preserves_support {n : ℕ}
    (g : ℝ → ℝ) (hg : g 0 = 0) (S : Finset (Fin n)) :
    PreservesSupport (fun (v : Vec n) (j : Fin n) => g (v j)) S := by
  intro v hv j hj; show g (v j) = 0; rw [hv j hj, hg]

/-- Element-wise multiplication preserves support: 0 · w[j] = 0.
    Covers the relu_grad step: d_z = d_h ⊙ relu_grad. -/
theorem hmul_preserves_support {n : ℕ} (w : Vec n) (S : Finset (Fin n)) :
    PreservesSupport (· ⊙ w) S := by
  intro v hv j hj; simp only [hmul, hv j hj, zero_mul]

/-- Composition of support-preserving functions preserves support.
    The backward pass is a chain; if each link preserves support,
    the full chain does. -/
theorem compose_preserves_support {n : ℕ}
    (f g : Vec n → Vec n) (S : Finset (Fin n))
    (hf : PreservesSupport f S) (hg : PreservesSupport g S) :
    PreservesSupport (f ∘ g) S :=
  fun v hv j hj => hf (g v) (hg v hv) j hj

/-- **KEY THEOREM (Support Preservation → W₁ Confinement).**

    If the backward function between the gate and the W₁ weight
    update preserves support on groups(r), then the W₁ gradient
    at column j is zero for all j ∉ groups(r).

    This is the formal gate placement constraint:
    • Standard (act → gate → W₂): backward chain is (· ⊙ relu_grad),
      which preserves support → W₁ SAFE
    • Broken (act → LN → gate → W₂): backward chain includes LN
      backward, which subtracts the mean → does NOT preserve support
      → W₁ UNSAFE

    Generalizes `weight_update_confined` to arbitrary architectures:
    any backward chain that preserves support maintains W₁
    confinement, regardless of what other components exist. -/
theorem support_preserving_maintains_w1_confinement {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (f_back : Vec n → Vec n)
    (hf : PreservesSupport f_back (P.groups r))
    (d_gated : Vec n) (mlp_input : Vec m)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m,
      outer mlp_input (f_back (d_gated ⊙ indicator (P.groups r))) i j = 0 := by
  intro i
  have h_supp : ∀ j', j' ∉ P.groups r → (d_gated ⊙ indicator (P.groups r)) j' = 0 :=
    fun j' hj' => gradient_isolation d_gated _ j' (indicator_not_mem _ j' hj')
  simp only [outer, hf _ h_supp j hj, mul_zero]

/-- **W₂ Confinement Is Unconditional On Placement.**

    W₂ row confinement depends only on the forward-pass gate:
    gated[j] = h[j] · mask[j] = 0 when mask[j] = 0.

    No matter what other operations exist in the network,
    d_W₂[j,k] = gated[j] · d_output[k] = 0 · d_output[k] = 0.

    This asymmetry is architecturally important:
    • W₁ confinement: depends on backward chain → needs support preservation
    • W₂ confinement: depends on forward gate → always holds -/
theorem w2_confinement_unconditional {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (hidden_act : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ (o : ℕ) (d_any : Vec o) (k : Fin o),
      outer (hidden_act ⊙ indicator (P.groups r)) d_any j k = 0 :=
  w2_rows_are_regime_confined P r j hj hidden_act

/-- **NEGATIVE RESULT (Non-Preserving Backward → Confinement Breakable).**

    If f_back can map a vector supported on groups(r) to a vector
    with nonzero values at some j ∉ groups(r), then there exist
    upstream gradients for which the post-backward gradient at j
    is nonzero — meaning the W₁ column j update would be nonzero
    despite the gate zeroing the gradient at j.

    This formalizes the LayerNorm placement hazard:
    LN backward computes d_x[j] = (1/σ)(d_h[j] - mean(d_h) - …).
    Even when the gate zeros d_h[j], mean(d_h) ≠ 0 from active
    dimensions, so d_x[j] ≠ 0. Placing LN between activation and
    gate would leak gradient to inactive W₁ columns.

    Proof: given a witness v supported on groups(r) with
    f_back(v)[j] ≠ 0, set d_gated = v. Since v is already
    supported on groups(r), v ⊙ indicator(groups r) = v,
    and f_back(v)[j] ≠ 0. -/
theorem non_preserving_breaks_w1_confinement {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (f_back : Vec n → Vec n)
    (h_break : ∃ (v : Vec n) (j : Fin n),
      (∀ i : Fin n, i ∉ P.groups r → v i = 0) ∧
      j ∉ P.groups r ∧ f_back v j ≠ 0) :
    ∃ (d_gated : Vec n) (j : Fin n),
      j ∉ P.groups r ∧
      f_back (d_gated ⊙ indicator (P.groups r)) j ≠ 0 := by
  obtain ⟨v, j, hv_supp, hj, hne⟩ := h_break
  refine ⟨v, j, hj, ?_⟩
  have h_eq : v ⊙ indicator (P.groups r) = v := by
    ext i; simp only [hmul]
    by_cases hi : i ∈ P.groups r
    · rw [indicator_mem _ i hi, mul_one]
    · rw [indicator_not_mem _ i hi, mul_zero, hv_supp i hi]
  rw [h_eq]; exact hne


/-- **Theorem (Residual Preserves W₁ Confinement).**
    In a transformer block y = x + MLP(x), the gradient that
    reaches the gated hidden layer from the loss is an arbitrary
    `d_gated : Vec n`. This gradient may have flowed through
    residual connections, attention layers, LayerNorm, or any
    composition thereof.

    The gate then applies: d_h[j] = d_gated[j] · mask[j].
    When j ∉ groups(r), mask[j] = 0 and d_h[j] = 0.
    The W₁ column update d_W1[i,j] = x[i] · d_z1[j] = 0.

    This holds REGARDLESS of how d_gated was computed. The
    residual, attention, LayerNorm, or any other upstream
    computation affects only the magnitude of the gradient,
    not the zero/nonzero structure imposed by the gate.

    Proof: direct application of `weight_update_confined`
    with mask = indicator(groups(r)). -/
theorem residual_preserves_w1_confinement {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_gated : Vec n) (relu_grad : Vec n) (mlp_input : Vec m)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m,
      let d_h := d_gated ⊙ indicator (P.groups r)
      let d_z1 := d_h ⊙ relu_grad
      outer mlp_input d_z1 i j = 0 :=
  weight_update_confined d_gated _ relu_grad mlp_input j (indicator_not_mem _ j hj)

/-- **Theorem (Residual Preserves W₂ Confinement).**
    The W₂ gradient is d_W2[j,k] = gated[j] · d_logits[k].
    When mask[j] = 0, gated[j] = h[j] · 0 = 0, so d_W2[j,k] = 0.

    This holds for ANY d_logits, including those computed via
    residual-connected loss pathways.

    Proof: direct application of `w2_rows_are_regime_confined`. -/
theorem residual_preserves_w2_confinement {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (hidden_act : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ (o : ℕ) (d_logits : Vec o) (k : Fin o),
      outer (hidden_act ⊙ indicator (P.groups r)) d_logits j k = 0 :=
  w2_rows_are_regime_confined P r j hj hidden_act

/-- **Theorem (Residual-MLP Decomposition).**
    A transformer block's output decomposes as:
      y[k] = input[k] + (∑ j ∈ groups(r), h[j] · W2[j,k] + b2[k])

    The first term `input[k]` is partition-independent — it is
    the residual stream from attention and prior layers, shared
    across all regimes.

    The second term depends ONLY on dimensions in groups(r) —
    the active partition. This is the regime-specific MLP
    contribution, proved by `regime_output_locality`.

    For weight opacity: only W₁ columns and W₂ rows indexed by
    groups(r) are trained by regime r (by weight confinement).
    The remaining MLP weights carry information from other regimes
    but are invisible to regime r's forward pass. -/
theorem residual_mlp_decomposition {n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (input : Vec o) (h_act : Vec n)
    (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ) :
    ∀ k : Fin o,
      input k + output_logits (h_act ⊙ indicator (P.groups r)) W2 b2 k =
      input k + ((P.groups r).sum (fun j => h_act j * W2 j k) + b2 k) := by
  intro k
  congr 1
  exact regime_output_locality P r h_act W2 b2 k


-- ── §3C. RESIDUAL BLOCK BACKWARD — EXPLICIT FORMALIZATION ───────
--
-- We define the residual backward split and the W₂ backprop step,
-- then prove the complete confinement chain through a residual
-- transformer block:
--
--   d_block_output → residual split → W₂.T → gate → zero
--
-- The residual split is the identity (copies gradient to both
-- branches). W₂ backprop is matrix-vector multiplication. The
-- gate then zeros all inactive dimensions.
-- ─────────────────────────────────────────────────────────────────

/-- Backpropagation through W₂: d_gated[j] = Σ_k d_output[k] · W₂[j,k].
    The gradient flowing from the output layer back to the gated
    hidden layer. In a residual block, d_output = d_block_output
    (the residual copies the gradient identically). -/
def backprop_through_w2 {n o : ℕ} (d_output : Vec o) (W2 : Fin n → Fin o → ℝ) : Vec n :=
  fun j => ∑ k : Fin o, d_output k * W2 j k

/-- **Theorem (Residual Block Complete Confinement).**

    The full backward chain through a residual transformer block:

    1. d_block_output arrives from upstream (loss, later layers)
    2. Residual split: d_mlp = d_block_output (gradient copy)
    3. W₂ backprop: d_gated[j] = Σ_k d_block_output[k] · W₂[j,k]
    4. Gate: d_h[j] = d_gated[j] · mask[j] = 0 when j ∉ groups(r)
    5. ReLU backward: d_z[j] = d_h[j] · relu'(z)[j] = 0 · _ = 0
    6. W₁ update: d_W₁[i,j] = input[i] · d_z[j] = _ · 0 = 0

    And independently (forward path):
    7. W₂ update: d_W₂[j,k] = gated[j] · d_mlp[k] = 0
       (because gated[j] = h[j] · mask[j] = 0 in the forward pass)

    All weight updates at column/row j are zero for j ∉ groups(r),
    regardless of what d_block_output contains. The residual
    connection copies the gradient but does not alter the zero
    structure imposed by the gate.

    This explicitly composes the residual split, W₂ backprop,
    and gate confinement into a single end-to-end theorem for
    the full transformer block backward pass. -/
theorem residual_block_full_confinement {m n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_block_output : Vec o) (W2 : Fin n → Fin o → ℝ)
    (relu_grad : Vec n) (mlp_input : Vec m) (hidden_act : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    -- (a) Gate gradient is zero at j
    ((backprop_through_w2 d_block_output W2 ⊙ indicator (P.groups r)) j = 0)
    -- (b) Every entry in W₁ column j has zero gradient
    ∧ (∀ i : Fin m,
        outer mlp_input
          ((backprop_through_w2 d_block_output W2 ⊙ indicator (P.groups r)) ⊙ relu_grad)
          i j = 0)
    -- (c) Every entry in W₂ row j has zero gradient
    ∧ (∀ k : Fin o,
        outer (hidden_act ⊙ indicator (P.groups r)) d_block_output j k = 0) := by
  have hmask := indicator_not_mem (P.groups r) j hj
  refine ⟨?_, ?_, ?_⟩
  · simp only [backprop_through_w2, hmul, hmask, mul_zero]
  · intro i
    simp only [backprop_through_w2, outer, hmul, hmask, mul_zero, zero_mul]
  · intro k
    simp only [outer, hmul, hmask, mul_zero, zero_mul]


-- ════════════════════════════════════════════════════════════════
-- §4. DISTRIBUTABLE SAFETY V3 — MOVED TO DistributableClaims.lean
--
-- DistributableSafetyV3, standard_is_distributable_safe_v3, and
-- end_to_end_chain_v3 have been moved to DistributableClaims.lean.
-- ════════════════════════════════════════════════════════════════


end Schemen.SecurityV3
