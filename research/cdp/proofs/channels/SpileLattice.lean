/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import EnglishTax

/-!
# SpileLattice -- Composed Operational-Floor Bound

Mechanizes the composition theorem for bounded relative-perturbation
channels.  This retires the deferral on line ~375 of `EnglishTax.lean`,
where the n-hop accumulation bound was deliberately not formalized.
The Python suite at `experiments/spiles/` makes the per-step
operational floor load-bearing in 1,301 property tests; multi-resolution
lattices stack channels, so we now need the chained bound to be a
theorem rather than a conjecture.

## Statement of Results

### Channel Composition

- `BoundedChannel`: a transport map `T : E → E` with relative-perturbation
  bound `δ`, i.e. `‖T u - u‖ ≤ δ ‖u‖` for every `u ∈ E`.

- `applyChain`: left-fold application of a list of channels.

- `cumulativeDelta`: the Wilkinson-style product factor `∏(1 + δᵢ) - 1`.

### Theorems

- `cumulative_perturbation_bound`: composing k bounded channels gives a
  channel with relative-perturbation bound `cumulativeDelta`, i.e.
  `‖applyChain channels v - v‖ ≤ cumulativeDelta(δ-list) · ‖v‖`.

- `chained_operational_floor_bound`: when `cumulativeDelta < 1`,
  the cosine between input and the chained output is bounded below by
  `(1 - cumulativeDelta) / (1 + cumulativeDelta)`.

- `chained_operational_floor_two_delta`: the cosine deviation
  `1 - cos(v, u_k)` is bounded above by `2·Δ / (1 + Δ)`.

- `cumulative_delta_uniform`: when all δᵢ are equal to `δ`,
  `cumulativeDelta = (1 + δ)^k - 1`.

## Axioms

ZERO new axioms.  Everything reduces to:
- `EnglishTax.operational_floor_bound` (per-step floor, EnglishTax §7),
- `norm_add_le` (triangle inequality, mathlib),
- list induction on the channel sequence.

## Honesty Statement

This theorem gives the *worst-case* product bound.  Real channels in a
realistic transformer or spile lattice are not adversarial in their
error directions; LayerNorm and softmax periodically renormalize the
dynamic range, so the empirical error growth is far below the
Wilkinson product.  We retain the formal worst-case bound because it
is the *only* claim that lifts to "any chain of bounded channels" --
empirical structure is per-architecture and does not generalize.
-/

set_option autoImplicit false
-- Three lemmas (`applyChain_nil`, `applyChain_cons`,
-- `cumulative_perturbation_bound`) do not use `InnerProductSpace ℝ E`
-- and one does not use `NormedAddCommGroup E` either; they are stated
-- in this scope so they can compose with the floor theorem below
-- without re-introducing variables.  The unused-section-variables
-- linter is silenced for that reason; correctness is not affected.
set_option linter.unusedSectionVars false

noncomputable section

namespace Schemen.SpileLattice

open Schemen.EnglishTax
open scoped RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]


-- ════════════════════════════════════════════════════════════
-- §1. BOUNDED-PERTURBATION CHANNELS
-- ════════════════════════════════════════════════════════════

/-- A bounded relative-perturbation channel.

    A channel `T : E → E` is δ-bounded if for every input `u : E`,
    `‖T u - u‖ ≤ δ · ‖u‖`.  This is the channel-level abstraction of
    operational-floor compliance: the channel adds at most relative
    error δ to its input, in any direction.

    The Python primitive `Codebook` (with `per_coord_delta`) and the
    `IdentityChannel` (with `δ = 0`) are the canonical instances. -/
structure BoundedChannel (E : Type*) [NormedAddCommGroup E] where
  T : E → E
  δ : ℝ
  δ_nn : 0 ≤ δ
  bound : ∀ u : E, ‖T u - u‖ ≤ δ * ‖u‖


-- ════════════════════════════════════════════════════════════
-- §2. CHANNEL CHAIN APPLICATION
-- ════════════════════════════════════════════════════════════

/-- Apply a list of channels left-to-right:
    `applyChain [T₁, T₂, T₃] v = T₃ (T₂ (T₁ v))`. -/
def applyChain : List (E → E) → E → E
  | [],      v => v
  | T :: Ts, v => applyChain Ts (T v)

@[simp] lemma applyChain_nil (v : E) : applyChain ([] : List (E → E)) v = v := rfl

@[simp] lemma applyChain_cons (T : E → E) (Ts : List (E → E)) (v : E) :
    applyChain (T :: Ts) v = applyChain Ts (T v) := rfl


-- ════════════════════════════════════════════════════════════
-- §3. CUMULATIVE PERTURBATION FACTOR
-- ════════════════════════════════════════════════════════════

/-- The cumulative relative-perturbation factor of a list of bounds:
    `Δ([δ₁, …, δₖ]) := (1 + δ₁)(1 + δ₂)⋯(1 + δₖ) - 1`.

    For the empty list, Δ = 0 (no channels, no perturbation).  This is
    the Wilkinson-style product factor that controls the chained
    operational floor.  We define it by direct recursion to avoid a
    dependency on Mathlib's BigOperators infrastructure. -/
def cumulativeDelta : List ℝ → ℝ
  | []      => 0
  | δ :: δs => (1 + δ) * (1 + cumulativeDelta δs) - 1

@[simp] lemma cumulativeDelta_nil :
    cumulativeDelta ([] : List ℝ) = 0 := rfl

@[simp] lemma cumulativeDelta_cons (δ : ℝ) (δs : List ℝ) :
    cumulativeDelta (δ :: δs) =
      (1 + δ) * (1 + cumulativeDelta δs) - 1 := rfl

/-- The cumulative factor is non-negative when each component is. -/
lemma cumulativeDelta_nonneg {δs : List ℝ}
    (hδ : ∀ δ ∈ δs, 0 ≤ δ) :
    0 ≤ cumulativeDelta δs := by
  induction δs with
  | nil => simp
  | cons δ tail ih =>
    have h_head : 0 ≤ δ := hδ δ List.mem_cons_self
    have h_tail_nn : ∀ x ∈ tail, 0 ≤ x :=
      fun x hx => hδ x (List.mem_cons_of_mem _ hx)
    have h_tail_cum : 0 ≤ cumulativeDelta tail := ih h_tail_nn
    rw [cumulativeDelta_cons]
    nlinarith [h_head, h_tail_cum,
               mul_nonneg (by linarith : (0 : ℝ) ≤ δ)
                          (by linarith : (0 : ℝ) ≤ cumulativeDelta tail)]


-- ════════════════════════════════════════════════════════════
-- §4. CUMULATIVE PERTURBATION BOUND
--
-- The core induction.  Composing k channels with bounds δ₁, …, δₖ
-- gives a transport whose deviation from the identity is bounded by
-- the Wilkinson product factor `cumulativeDelta`.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Cumulative perturbation bound).**
    Composing a list of bounded channels yields a channel that
    deviates from the identity by at most `cumulativeDelta` of the
    component bounds, in relative norm.

    Specifically, for every `v : E`,
    `‖applyChain (channels.map T) v - v‖ ≤ cumulativeDelta (channels.map δ) · ‖v‖`.

    The proof is a direct induction on the channel list: the inductive
    step combines the head's bound with the tail's bound (applied to
    the head's output) via the triangle inequality. -/
theorem cumulative_perturbation_bound
    (channels : List (BoundedChannel E)) :
    ∀ (v : E),
      ‖applyChain (channels.map BoundedChannel.T) v - v‖ ≤
        cumulativeDelta (channels.map BoundedChannel.δ) * ‖v‖ := by
  induction channels with
  | nil =>
    intro v
    simp
  | cons head tail ih =>
    intro v
    -- Notation:
    --   u  := head.T v          -- output after the first channel
    --   δh := head.δ            -- head's bound
    --   T_tail := tail.map T    -- remaining channels' transports
    --   δ_tail := cumulativeDelta (tail.map δ)
    -- We bound ‖applyChain T_tail u - v‖ via triangle decomposition.
    have h_v_nn : (0 : ℝ) ≤ ‖v‖ := norm_nonneg v
    have h_head_pert : ‖head.T v - v‖ ≤ head.δ * ‖v‖ := head.bound v
    -- ‖head.T v‖ ≤ (1 + head.δ) ‖v‖.
    have h_norm_head : ‖head.T v‖ ≤ (1 + head.δ) * ‖v‖ := by
      have h_split : head.T v = (head.T v - v) + v := by abel
      calc ‖head.T v‖ = ‖(head.T v - v) + v‖ := by rw [← h_split]
        _ ≤ ‖head.T v - v‖ + ‖v‖ := norm_add_le _ _
        _ ≤ head.δ * ‖v‖ + ‖v‖ := by linarith
        _ = (1 + head.δ) * ‖v‖ := by ring
    -- Inductive hypothesis applied at u = head.T v.
    have h_tail_at_u :
        ‖applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v‖ ≤
          cumulativeDelta (tail.map BoundedChannel.δ) * ‖head.T v‖ :=
      ih (head.T v)
    -- δ_tail ≥ 0.
    have h_tail_cum_nn :
        0 ≤ cumulativeDelta (tail.map BoundedChannel.δ) := by
      apply cumulativeDelta_nonneg
      intro x hx
      simp only [List.mem_map] at hx
      obtain ⟨c, _, rfl⟩ := hx
      exact c.δ_nn
    -- Replace ‖head.T v‖ in the tail bound by (1 + head.δ) ‖v‖.
    have h_tail_in_v :
        ‖applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v‖ ≤
          cumulativeDelta (tail.map BoundedChannel.δ) *
            ((1 + head.δ) * ‖v‖) := by
      calc ‖applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v‖
          ≤ cumulativeDelta (tail.map BoundedChannel.δ) * ‖head.T v‖ :=
            h_tail_at_u
        _ ≤ cumulativeDelta (tail.map BoundedChannel.δ) *
              ((1 + head.δ) * ‖v‖) :=
            mul_le_mul_of_nonneg_left h_norm_head h_tail_cum_nn
    -- Triangle inequality: ‖A - v‖ ≤ ‖A - u‖ + ‖u - v‖.
    have h_triangle :
        ‖applyChain (tail.map BoundedChannel.T) (head.T v) - v‖ ≤
          ‖applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v‖ +
            ‖head.T v - v‖ := by
      have h_split :
          applyChain (tail.map BoundedChannel.T) (head.T v) - v =
            (applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v) +
              (head.T v - v) := by abel
      calc ‖applyChain (tail.map BoundedChannel.T) (head.T v) - v‖
          = ‖(applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v) +
              (head.T v - v)‖ := by rw [← h_split]
        _ ≤ ‖applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v‖ +
              ‖head.T v - v‖ := norm_add_le _ _
    -- Combine the three bounds into a single inequality.
    have h_assembled :
        ‖applyChain (tail.map BoundedChannel.T) (head.T v) - v‖ ≤
          (cumulativeDelta (tail.map BoundedChannel.δ) * (1 + head.δ) +
              head.δ) * ‖v‖ := by
      calc ‖applyChain (tail.map BoundedChannel.T) (head.T v) - v‖
          ≤ ‖applyChain (tail.map BoundedChannel.T) (head.T v) - head.T v‖ +
              ‖head.T v - v‖ := h_triangle
        _ ≤ cumulativeDelta (tail.map BoundedChannel.δ) *
              ((1 + head.δ) * ‖v‖) + head.δ * ‖v‖ := by linarith
        _ = (cumulativeDelta (tail.map BoundedChannel.δ) * (1 + head.δ) +
              head.δ) * ‖v‖ := by ring
    -- The (head :: tail) form of cumulativeDelta equals the assembled
    -- coefficient.
    have h_cum_eq :
        cumulativeDelta ((head :: tail).map BoundedChannel.δ) =
          cumulativeDelta (tail.map BoundedChannel.δ) * (1 + head.δ) +
            head.δ := by
      rw [List.map_cons, cumulativeDelta_cons]
      ring
    -- Rewrite the LHS via applyChain_cons and conclude.
    have h_chain_eq :
        applyChain ((head :: tail).map BoundedChannel.T) v =
          applyChain (tail.map BoundedChannel.T) (head.T v) := by
      simp [List.map_cons]
    rw [h_chain_eq, h_cum_eq]
    exact h_assembled


-- ════════════════════════════════════════════════════════════
-- §5. CHAINED OPERATIONAL FLOOR -- the headline corollary
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Chained operational floor).**
    A composition of `k` bounded channels with bounds `δ₁, …, δₖ`
    satisfies a single operational-floor cosine bound with total
    perturbation `Δ := (1 + δ₁)⋯(1 + δₖ) - 1`, provided `Δ < 1`:

    `cosSim v (applyChain channels v) ≥ (1 - Δ) / (1 + Δ)`.

    The proof reduces to:
      (i)  the cumulative-perturbation bound (the composed channel is
           Δ-bounded), and
      (ii) the per-step `EnglishTax.operational_floor_bound`.
    No new axioms. -/
theorem chained_operational_floor_bound
    {v : E} (hv : v ≠ 0)
    (channels : List (BoundedChannel E))
    (h_total_lt_one :
      cumulativeDelta (channels.map BoundedChannel.δ) < 1) :
    (1 - cumulativeDelta (channels.map BoundedChannel.δ)) /
        (1 + cumulativeDelta (channels.map BoundedChannel.δ)) ≤
      cosSim v (applyChain (channels.map BoundedChannel.T) v) := by
  -- The composed channel deviates from v by at most Δ ‖v‖.
  have h_pert :
      ‖applyChain (channels.map BoundedChannel.T) v - v‖ ≤
        cumulativeDelta (channels.map BoundedChannel.δ) * ‖v‖ :=
    cumulative_perturbation_bound channels v
  -- Δ ≥ 0.
  have h_Δ_nn :
      0 ≤ cumulativeDelta (channels.map BoundedChannel.δ) := by
    apply cumulativeDelta_nonneg
    intro x hx
    simp only [List.mem_map] at hx
    obtain ⟨c, _, rfl⟩ := hx
    exact c.δ_nn
  -- Apply EnglishTax floor with ε = (composed v) - v, so v + ε = composed v.
  set ε := applyChain (channels.map BoundedChannel.T) v - v with hε_def
  have h_eq : v + ε = applyChain (channels.map BoundedChannel.T) v := by
    simp [hε_def]
  rw [← h_eq]
  exact operational_floor_bound hv h_Δ_nn h_total_lt_one h_pert


-- ════════════════════════════════════════════════════════════
-- §6. DEVIATION FORM (CHAINED)
-- ════════════════════════════════════════════════════════════

/-- **Corollary (Chained two-delta floor).**
    Under the hypotheses of `chained_operational_floor_bound`, the
    cosine deviation `1 - cosSim v (applyChain channels v)` is at most
    `2Δ / (1 + Δ)`.  This is the loose-but-cheap form for
    error-budget arguments across channel chains. -/
theorem chained_operational_floor_two_delta
    {v : E} (hv : v ≠ 0)
    (channels : List (BoundedChannel E))
    (h_total_lt_one :
      cumulativeDelta (channels.map BoundedChannel.δ) < 1) :
    1 - cosSim v (applyChain (channels.map BoundedChannel.T) v) ≤
      2 * cumulativeDelta (channels.map BoundedChannel.δ) /
        (1 + cumulativeDelta (channels.map BoundedChannel.δ)) := by
  set Δ := cumulativeDelta (channels.map BoundedChannel.δ) with hΔ_def
  have h_floor := chained_operational_floor_bound hv channels h_total_lt_one
  have h_Δ_nn : 0 ≤ Δ := by
    apply cumulativeDelta_nonneg
    intro x hx
    simp only [List.mem_map] at hx
    obtain ⟨c, _, rfl⟩ := hx
    exact c.δ_nn
  have h_one_plus_pos : (0 : ℝ) < 1 + Δ := by linarith
  have h_id : (1 : ℝ) - 2 * Δ / (1 + Δ) = (1 - Δ) / (1 + Δ) := by
    field_simp
    ring
  linarith [h_floor, h_id]


-- ════════════════════════════════════════════════════════════
-- §7. UNIFORM-BOUND CASE
-- ════════════════════════════════════════════════════════════

/-- **Lemma (Cumulative delta -- uniform bound).**
    When every channel has the same bound `δ`, the cumulative factor
    reduces to `(1 + δ)^k - 1`, where `k` is the chain length.

    This is the form most often invoked in engineering analyses: a
    chain of k identical-quality channels behaves like a single
    `(1 + δ)^k - 1`-bounded channel. -/
lemma cumulative_delta_uniform (δ : ℝ) (k : ℕ) :
    cumulativeDelta (List.replicate k δ) = (1 + δ) ^ k - 1 := by
  induction k with
  | zero => simp
  | succ n ih =>
    rw [List.replicate_succ, cumulativeDelta_cons, ih, pow_succ]
    ring

/-- **Theorem (Chained operational floor, uniform).**
    A chain of `k` channels each with bound `δ ≥ 0`, where
    `(1 + δ)^k < 2`, has cosine fidelity at least
    `(2 - (1+δ)^k) / (1+δ)^k`. -/
theorem chained_operational_floor_uniform
    {v : E} (hv : v ≠ 0)
    {δ : ℝ} (hδ_nn : 0 ≤ δ) {k : ℕ}
    (channels : List (BoundedChannel E))
    (h_len : channels.length = k)
    (h_uniform : ∀ c ∈ channels, c.δ = δ)
    (h_lt_two : (1 + δ) ^ k < 2) :
    (2 - (1 + δ) ^ k) / (1 + δ) ^ k ≤
      cosSim v (applyChain (channels.map BoundedChannel.T) v) := by
  -- Step 1: the channel-δ list is List.replicate k δ.
  have h_δ_list :
      channels.map BoundedChannel.δ = List.replicate k δ := by
    apply List.eq_replicate_iff.mpr
    refine ⟨?_, ?_⟩
    · rw [List.length_map, h_len]
    · intro x hx
      simp only [List.mem_map] at hx
      obtain ⟨c, hc_mem, rfl⟩ := hx
      exact h_uniform c hc_mem
  -- Step 2: cumulativeDelta of that list is (1+δ)^k - 1.
  have h_Δ_eq :
      cumulativeDelta (channels.map BoundedChannel.δ) = (1 + δ) ^ k - 1 := by
    rw [h_δ_list, cumulative_delta_uniform]
  -- Step 3: cumulativeDelta < 1 follows from (1+δ)^k < 2.
  have h_lt_one :
      cumulativeDelta (channels.map BoundedChannel.δ) < 1 := by
    rw [h_Δ_eq]; linarith
  -- Step 4: apply the general chained floor and simplify the bound.
  have h_floor := chained_operational_floor_bound hv channels h_lt_one
  rw [h_Δ_eq] at h_floor
  have h_pos : (0 : ℝ) < (1 + δ) ^ k :=
    pow_pos (by linarith : (0 : ℝ) < 1 + δ) k
  have h_id :
      (1 - ((1 + δ) ^ k - 1)) / (1 + ((1 + δ) ^ k - 1)) =
        (2 - (1 + δ) ^ k) / (1 + δ) ^ k := by
    field_simp
    ring
  rw [h_id] at h_floor
  exact h_floor


-- ════════════════════════════════════════════════════════════
-- §8. TIGHT CHAINED FLOOR
--
-- Same composition argument applied to the tight per-step floor
-- `operational_floor_tight` (which gives `cosSim ≥ √(1 - δ²)`).
-- Materially better than the loose `(1-Δ)/(1+Δ)` form once the
-- chain has more than a handful of channels.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Tight chained operational floor).**
    A composition of bounded channels with cumulative
    perturbation factor `Δ < 1` satisfies the tight cosine bound

       cosSim v (applyChain channels v) ≥ √(1 − Δ²).

    Direct corollary: `cumulative_perturbation_bound` gives
    `‖compose(v) − v‖ ≤ Δ‖v‖`, then `operational_floor_tight`
    delivers the bound on cosine fidelity.  Same induction skeleton
    as `chained_operational_floor_bound`; same zero-axiom proof. -/
theorem chained_operational_floor_tight
    {v : E} (hv : v ≠ 0)
    (channels : List (BoundedChannel E))
    (h_total_lt_one :
      cumulativeDelta (channels.map BoundedChannel.δ) < 1) :
    Real.sqrt
        (1 - cumulativeDelta (channels.map BoundedChannel.δ) ^ 2) ≤
      cosSim v (applyChain (channels.map BoundedChannel.T) v) := by
  have h_pert :
      ‖applyChain (channels.map BoundedChannel.T) v - v‖ ≤
        cumulativeDelta (channels.map BoundedChannel.δ) * ‖v‖ :=
    cumulative_perturbation_bound channels v
  have h_Δ_nn :
      0 ≤ cumulativeDelta (channels.map BoundedChannel.δ) := by
    apply cumulativeDelta_nonneg
    intro x hx
    simp only [List.mem_map] at hx
    obtain ⟨c, _, rfl⟩ := hx
    exact c.δ_nn
  set ε := applyChain (channels.map BoundedChannel.T) v - v with hε_def
  have h_eq : v + ε = applyChain (channels.map BoundedChannel.T) v := by
    simp [hε_def]
  rw [← h_eq]
  exact operational_floor_tight hv h_Δ_nn h_total_lt_one h_pert


-- ════════════════════════════════════════════════════════════
-- §9. ADDITIVE-PERTURBATION CHAINS
--
-- The chained bounds above assume each channel has a *relative*
-- perturbation bound (`‖T u - u‖ ≤ δ ‖u‖`).  Coordinate-wise
-- quantizers (codebooks) instead satisfy an *additive* bound
-- (`‖Q v - v‖ ≤ ε_abs`), and there is no uniform relative bound.
-- The chained version is then a sum, by triangle inequality.
-- ════════════════════════════════════════════════════════════

/-- A bounded *additive*-perturbation channel.

    A channel `T : E → E` is `ε_abs`-additively-bounded if
    `‖T u - u‖ ≤ ε_abs` for every `u ∈ E`.  This is the natural
    abstraction of coordinate-wise quantizers, where
    `ε_abs = (step / 2) · √d` is independent of `‖u‖`. -/
structure AdditiveChannel (E : Type*) [NormedAddCommGroup E] where
  T : E → E
  εAbs : ℝ
  εAbs_nn : 0 ≤ εAbs
  bound : ∀ u : E, ‖T u - u‖ ≤ εAbs

/-- The cumulative additive bound of a chain: simple sum. -/
def cumulativeEpsAbs : List ℝ → ℝ
  | []          => 0
  | ε :: εs     => ε + cumulativeEpsAbs εs

@[simp] lemma cumulativeEpsAbs_nil :
    cumulativeEpsAbs ([] : List ℝ) = 0 := rfl

@[simp] lemma cumulativeEpsAbs_cons (ε : ℝ) (εs : List ℝ) :
    cumulativeEpsAbs (ε :: εs) = ε + cumulativeEpsAbs εs := rfl

lemma cumulativeEpsAbs_nonneg {εs : List ℝ}
    (h : ∀ ε ∈ εs, 0 ≤ ε) :
    0 ≤ cumulativeEpsAbs εs := by
  induction εs with
  | nil => simp
  | cons ε tail ih =>
    have h_head : 0 ≤ ε := h ε List.mem_cons_self
    have h_tail : ∀ x ∈ tail, 0 ≤ x :=
      fun x hx => h x (List.mem_cons_of_mem _ hx)
    rw [cumulativeEpsAbs_cons]
    linarith [ih h_tail]

/-- **Theorem (Additive cumulative bound).**
    A chain of `k` additively-bounded channels has additive bound
    equal to the sum of component bounds:
    `‖compose(v) - v‖ ≤ ε₁ + ε₂ + ... + εₖ`.

    Proof by triangle inequality and induction.  No relative
    hypothesis, no Wilkinson product — coordinate-wise quantizers
    really do compose by simple addition in the worst case. -/
theorem additive_cumulative_bound
    (channels : List (AdditiveChannel E)) :
    ∀ (v : E),
      ‖applyChain (channels.map AdditiveChannel.T) v - v‖ ≤
        cumulativeEpsAbs (channels.map AdditiveChannel.εAbs) := by
  induction channels with
  | nil =>
    intro v
    simp
  | cons head tail ih =>
    intro v
    -- u := head.T v;  ‖u - v‖ ≤ head.εAbs by hypothesis.
    have h_head : ‖head.T v - v‖ ≤ head.εAbs := head.bound v
    -- IH on the tail at u:  ‖applyChain (tail.T) u - u‖ ≤ Σ tail.εAbs.
    have h_tail := ih (head.T v)
    -- Triangle:
    have h_triangle :
        ‖applyChain (tail.map AdditiveChannel.T) (head.T v) - v‖ ≤
          ‖applyChain (tail.map AdditiveChannel.T) (head.T v) - head.T v‖ +
            ‖head.T v - v‖ := by
      have h_split :
          applyChain (tail.map AdditiveChannel.T) (head.T v) - v =
            (applyChain (tail.map AdditiveChannel.T) (head.T v) - head.T v) +
              (head.T v - v) := by abel
      calc ‖applyChain (tail.map AdditiveChannel.T) (head.T v) - v‖
          = ‖(applyChain (tail.map AdditiveChannel.T) (head.T v) - head.T v) +
              (head.T v - v)‖ := by rw [← h_split]
        _ ≤ ‖applyChain (tail.map AdditiveChannel.T) (head.T v) - head.T v‖ +
              ‖head.T v - v‖ := norm_add_le _ _
    -- Goal rewrite + assemble.
    have h_chain_eq :
        applyChain ((head :: tail).map AdditiveChannel.T) v =
          applyChain (tail.map AdditiveChannel.T) (head.T v) := by
      simp [List.map_cons]
    have h_eps_eq :
        cumulativeEpsAbs ((head :: tail).map AdditiveChannel.εAbs) =
          head.εAbs + cumulativeEpsAbs (tail.map AdditiveChannel.εAbs) := by
      rw [List.map_cons, cumulativeEpsAbs_cons]
    rw [h_chain_eq, h_eps_eq]
    linarith [h_triangle, h_head, h_tail]


-- ════════════════════════════════════════════════════════════
-- §10. ENERGY-BUDGETED FOLDING
--
-- Folding (compressing a high-D residual onto a lower-D
-- representation) should be governed by the *error budget* and the
-- *energy in error-causing directions*, not by the availability of
-- dimensional bits.  Concretely: an orthogonal projection P onto a
-- subspace S satisfies an operational floor whose tightness is set
-- by the fraction of v's energy that lives in S^perp.
--
-- This is a direct corollary of `operational_floor_tight`; we
-- restate it under the natural name so callers can use it without
-- replaying the substitution.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Subspace projection floor / Energy fold).**

    If `p` is any vector with `‖v − p‖ ≤ ε · ‖v‖` for some
    `0 ≤ ε < 1`, then

       cosSim v p ≥ √(1 − ε²).

    The intended specialisation is `p` = the orthogonal projection
    of `v` onto a subspace `S`.  By the Pythagorean identity for
    orthogonal projections, `‖v − p‖² = ‖v‖² − ‖p‖²` is exactly the
    energy of `v` in `S^perp` -- the "directions that would cause
    error" if we folded the representation onto `S`.  The cosine
    floor therefore depends on the tail energy *in those
    directions*, not on the dimensional bits available.

    Proof: trivial substitution into `operational_floor_tight` with
    perturbation `eps := p - v`.  Same axiom set:
    `[propext, Classical.choice, Quot.sound]`. -/
theorem subspace_projection_floor
    {v p : E} (hv : v ≠ 0)
    {ε : ℝ}
    (hε_nn : 0 ≤ ε) (hε1 : ε < 1)
    (h_tail : ‖v - p‖ ≤ ε * ‖v‖) :
    Real.sqrt (1 - ε ^ 2) ≤ cosSim v p := by
  -- Re-orient the perturbation: ε_vec := p - v has same norm.
  have h_pert : ‖p - v‖ ≤ ε * ‖v‖ := by
    rw [norm_sub_rev]; exact h_tail
  have h_eq : v + (p - v) = p := by abel
  rw [← h_eq]
  exact operational_floor_tight hv hε_nn hε1 h_pert


end Schemen.SpileLattice
