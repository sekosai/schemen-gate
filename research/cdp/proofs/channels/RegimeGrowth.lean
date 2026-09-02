/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/


import CapacitySecurity

/-!
# Regime Growth — Dynamic Partition Validity

## Statement of Results

Regimes can grow, shrink, and transfer dimensions while preserving
all Tier 1 isolation guarantees. This file proves that dynamic
reallocation operations produce valid partitions.

**Theorem (Dynamic Isolation).** A DynamicPartition (disjoint groups
without exhaustiveness or equal-size constraints) preserves all
isolation properties: gradient zeroing, weight confinement, mask
orthogonality, and steganographic failure.

**Theorem (Annex Preserves Isolation).** Adding reserve dims
(dims not in any group) to a regime produces a new DynamicPartition
whose groups remain disjoint. All untouched regimes' masks are
unchanged.

**Theorem (Transfer Preserves Isolation).** Moving dims from one
regime to another (after zeroing in the donor) produces a new
DynamicPartition with disjoint groups.

**Theorem (Destruction Returns to Reserve).** Removing a regime's
dims from the partition produces a DynamicPartition where those
dims are in no group (available for reallocation).

## Axioms

ZERO new axioms. All theorems derive from V1 foundations.

## Key Insight

The V1 isolation theorems (gradient_isolation, weight_update_confined,
masks_orthogonal, forward_isolation) depend ONLY on mask disjointness.
They do NOT depend on exhaustiveness or equal group sizes. Therefore,
any disjoint group assignment — regardless of group sizes, unassigned
dims, or dynamic reallocation history — inherits the full isolation
guarantee.

## Proof Architecture

```
V1 (GateSecurity) ─── indicator, masks_orthogonal, gradient_isolation,
│                     forward_isolation, weight_update_confined
│
V6 (this file)
 ├── §1  DynamicPartition (relaxed ValidPartition)
 ├── §2  Dynamic isolation (V1 proofs apply to DynamicPartition)
 ├── §3  Annex from reserve
 ├── §4  Transfer between regimes
 ├── §5  Destruction (return to reserve)
 └── §6  ValidPartition embeds into DynamicPartition
```
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.RegimeGrowth

open Schemen Schemen.Security Schemen.SecurityV4


-- ════════════════════════════════════════════════════════════════
-- §1. DYNAMIC PARTITION
-- ════════════════════════════════════════════════════════════════

/-- A dynamic partition of Fin n into R groups. Relaxes
    ValidPartition by dropping exhaustiveness and equal-size
    requirements. Only disjointness is required — this is
    the single property that all isolation proofs depend on.

    Dims not in any group are "reserve" — available for
    allocation by the authority. -/
structure DynamicPartition (n R : ℕ) where
  groups : Fin R → Finset (Fin n)
  disjoint : ∀ r s : Fin R, r ≠ s → Disjoint (groups r) (groups s)

/-- Reserve dims: those not belonging to any group. -/
def DynamicPartition.isReserve {n R : ℕ}
    (P : DynamicPartition n R) (j : Fin n) : Prop :=
  ∀ r : Fin R, j ∉ P.groups r


-- ════════════════════════════════════════════════════════════════
-- §2. DYNAMIC ISOLATION
--
-- All V1 isolation theorems work with DynamicPartition because
-- they depend only on disjointness, which DynamicPartition
-- preserves. These theorems demonstrate that the full Tier 1
-- Citizen Rights hold for any disjoint group assignment.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Dynamic Gradient Isolation).**
    Dimensions outside the active group receive zero gradient.
    Identical to V1's partition_isolates, using DynamicPartition. -/
theorem dynamic_gradient_isolation {n R : ℕ}
    (P : DynamicPartition n R) (d_gated : Vec n)
    (r : Fin R) (j : Fin n) (hj : j ∉ P.groups r) :
    (d_gated ⊙ indicator (P.groups r)) j = 0 :=
  gradient_isolation d_gated (indicator (P.groups r)) j (indicator_not_mem _ j hj)

/-- **Theorem (Dynamic Mask Orthogonality).**
    Masks from distinct groups have zero Hadamard product. -/
theorem dynamic_masks_orthogonal {n R : ℕ}
    (P : DynamicPartition n R) (r s : Fin R) (hrs : r ≠ s)
    (j : Fin n) :
    (indicator (P.groups r) ⊙ indicator (P.groups s)) j = 0 :=
  masks_orthogonal _ _ (P.disjoint r s hrs) j

/-- **Theorem (Dynamic Weight Confinement — W1).**
    Training under group r's mask cannot update W1 columns
    outside group r. -/
theorem dynamic_w1_confined {m n R : ℕ}
    (P : DynamicPartition n R) (r : Fin R)
    (d_gated relu_grad : Vec n) (x : Vec m)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m,
      outer x ((d_gated ⊙ indicator (P.groups r)) ⊙ relu_grad) i j = 0 :=
  weight_update_confined d_gated (indicator (P.groups r)) relu_grad x j
    (indicator_not_mem _ j hj)

/-- **Theorem (Dynamic Weight Confinement — W2).**
    Training under group r's mask cannot update W2 rows
    outside group r. -/
theorem dynamic_w2_confined {n o R : ℕ}
    (P : DynamicPartition n R) (r : Fin R)
    (h : Vec n) (d_logits : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ k : Fin o,
      outer (h ⊙ indicator (P.groups r)) d_logits j k = 0 :=
  w2_update_confined h (indicator (P.groups r)) d_logits j
    (indicator_not_mem _ j hj)

/-- **Theorem (Dynamic Steganographic Failure).**
    Wrong group's mask reads zero at every dim of the correct
    group. -/
theorem dynamic_wrong_mask {n R : ℕ}
    (P : DynamicPartition n R) (r s : Fin R) (hrs : r ≠ s)
    (h : Vec n) (j : Fin n) (hj : j ∈ P.groups r) :
    (h ⊙ indicator (P.groups s)) j = 0 := by
  have hj_not_s : j ∉ P.groups s :=
    fun hmem => by
      have := Finset.disjoint_left.mp (P.disjoint r s hrs) hj
      exact this hmem
  exact forward_isolation h (indicator (P.groups s)) j (indicator_not_mem _ j hj_not_s)

/-- **Theorem (Dynamic Mutual Invisibility).**
    For any two distinct groups, forward pass through one
    produces zero at every dimension of the other. -/
theorem dynamic_mutual_invisibility {n R : ℕ}
    (P : DynamicPartition n R) (h : Vec n)
    (r s : Fin R) (hrs : r ≠ s) :
    ∀ j : Fin n, j ∈ P.groups s →
      (h ⊙ indicator (P.groups r)) j = 0 :=
  fun j hj => dynamic_wrong_mask P s r (Ne.symm hrs) h j hj


-- ════════════════════════════════════════════════════════════════
-- §3. ANNEX FROM RESERVE
--
-- Adding reserve dims to a group preserves disjointness.
-- The key insight: reserve dims are not in ANY group, so
-- adding them to one group cannot create overlap with another.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Annex Preserves Disjointness).**
    If new_dims are in reserve (not in any group) and we add
    them to group r, the resulting groups are still disjoint.

    This is the formal basis for regime expansion: the authority
    can grant reserve dims to a regime, and the isolation
    guarantee is preserved for all other regimes. -/
theorem annex_disjoint {n R : ℕ}
    (P : DynamicPartition n R) (r : Fin R)
    (new_dims : Finset (Fin n))
    (h_reserve : ∀ j ∈ new_dims, P.isReserve j)
    (s t : Fin R) (hst : s ≠ t) :
    let new_groups := fun x => if x = r then P.groups r ∪ new_dims else P.groups x
    Disjoint (new_groups s) (new_groups t) := by
  simp only
  rw [Finset.disjoint_left]
  intro j hj_s hj_t
  have h_pairwise : ∀ (r1 r2 : Fin R), r1 ≠ r2 →
      j ∈ P.groups r1 → j ∉ P.groups r2 :=
    fun r1 r2 h12 hj1 hj2 =>
      Finset.disjoint_left.mp (P.disjoint r1 r2 h12) hj1 hj2
  by_cases hsr : s = r
  · by_cases htr : t = r
    · exact hst (hsr.trans htr.symm)
    · rw [if_pos hsr] at hj_s
      rw [if_neg htr] at hj_t
      rcases Finset.mem_union.mp hj_s with hs_groups | hs_newdims
      · exact h_pairwise r t (Ne.symm htr) (hsr ▸ hs_groups) hj_t
      · exact h_reserve j hs_newdims t hj_t
  · by_cases htr : t = r
    · rw [if_neg hsr] at hj_s
      rw [if_pos htr] at hj_t
      rcases Finset.mem_union.mp hj_t with ht_groups | ht_newdims
      · exact h_pairwise s r hsr hj_s (htr ▸ ht_groups)
      · exact h_reserve j ht_newdims s hj_s
    · rw [if_neg hsr] at hj_s
      rw [if_neg htr] at hj_t
      exact h_pairwise s t hst hj_s hj_t

/-- Construct the annexed DynamicPartition. -/
def DynamicPartition.annex {n R : ℕ}
    (P : DynamicPartition n R) (r : Fin R)
    (new_dims : Finset (Fin n))
    (h_reserve : ∀ j ∈ new_dims, P.isReserve j) :
    DynamicPartition n R where
  groups := fun s => if s = r then P.groups r ∪ new_dims else P.groups s
  disjoint s t hst := annex_disjoint P r new_dims h_reserve s t hst

/-- **Theorem (Annex Preserves Other Groups).**
    After annexing to group r, every other group s (s ≠ r)
    is identical to before. Their masks, and therefore all
    their isolation properties, are unchanged. -/
theorem annex_preserves_others {n R : ℕ}
    (P : DynamicPartition n R) (r : Fin R)
    (new_dims : Finset (Fin n))
    (h_reserve : ∀ j ∈ new_dims, P.isReserve j)
    (s : Fin R) (hsr : s ≠ r) :
    (P.annex r new_dims h_reserve).groups s = P.groups s := by
  simp [DynamicPartition.annex, if_neg hsr]


-- ════════════════════════════════════════════════════════════════
-- §4. TRANSFER BETWEEN REGIMES
--
-- Removing dims from one group and adding them to another
-- preserves disjointness, provided the transferred dims are
-- zeroed in the donor (which we enforce operationally, not
-- algebraically — the zeroing is the Right of Destruction).
-- ════════════════════════════════════════════════════════════════

/-- Transfer dims: remove from source, add to target.

    Disjointness is proven element-wise via `Finset.disjoint_left`.
    The key facts: (1) members of `P.groups src \ dims` belong to
    `P.groups src` and not to `dims`; (2) any element of `dims` is
    in `P.groups src` (by `h_src`), so it's disjoint from every
    other regime; (3) the base partition is pairwise disjoint. -/
def DynamicPartition.transfer {n R : ℕ}
    (P : DynamicPartition n R) (src tgt : Fin R)
    (dims : Finset (Fin n))
    (h_src : dims ⊆ P.groups src) (h_ne : src ≠ tgt) :
    DynamicPartition n R where
  groups := fun s =>
    if s = src then P.groups src \ dims
    else if s = tgt then P.groups tgt ∪ dims
    else P.groups s
  disjoint := by
    intro s t hst
    rw [Finset.disjoint_left]
    intro j hj_s hj_t
    have h_pairwise : ∀ (r1 r2 : Fin R), r1 ≠ r2 →
        j ∈ P.groups r1 → j ∉ P.groups r2 :=
      fun r1 r2 h12 hj1 hj2 =>
        Finset.disjoint_left.mp (P.disjoint r1 r2 h12) hj1 hj2
    -- Classify what s's new group tells us about j
    by_cases hs_src : s = src
    · rw [if_pos hs_src] at hj_s
      have hj_src : j ∈ P.groups src := (Finset.mem_sdiff.mp hj_s).1
      have hj_not_dims : j ∉ dims := (Finset.mem_sdiff.mp hj_s).2
      by_cases ht_src : t = src
      · exact hst (hs_src.trans ht_src.symm)
      · by_cases ht_tgt : t = tgt
        · rw [if_neg ht_src, if_pos ht_tgt] at hj_t
          rcases Finset.mem_union.mp hj_t with hjt_tgt | hjt_dims
          · exact h_pairwise src tgt h_ne hj_src (ht_tgt ▸ hjt_tgt)
          · exact hj_not_dims hjt_dims
        · rw [if_neg ht_src, if_neg ht_tgt] at hj_t
          exact h_pairwise src t (Ne.symm ht_src) hj_src hj_t
    · by_cases hs_tgt : s = tgt
      · rw [if_neg hs_src, if_pos hs_tgt] at hj_s
        rcases Finset.mem_union.mp hj_s with hjs_tgt | hjs_dims
        · by_cases ht_src : t = src
          · rw [if_pos ht_src] at hj_t
            have hj_t_src : j ∈ P.groups src := (Finset.mem_sdiff.mp hj_t).1
            exact h_pairwise tgt src h_ne.symm (hs_tgt ▸ hjs_tgt) hj_t_src
          · by_cases ht_tgt : t = tgt
            · exact hst (hs_tgt.trans ht_tgt.symm)
            · rw [if_neg ht_src, if_neg ht_tgt] at hj_t
              exact h_pairwise tgt t (Ne.symm ht_tgt)
                (hs_tgt ▸ hjs_tgt) hj_t
        · -- j ∈ dims, so j ∈ P.groups src
          have hj_src : j ∈ P.groups src := h_src hjs_dims
          by_cases ht_src : t = src
          · rw [if_pos ht_src] at hj_t
            exact (Finset.mem_sdiff.mp hj_t).2 hjs_dims
          · by_cases ht_tgt : t = tgt
            · exact hst (hs_tgt.trans ht_tgt.symm)
            · rw [if_neg ht_src, if_neg ht_tgt] at hj_t
              exact h_pairwise src t (Ne.symm ht_src) hj_src hj_t
      · rw [if_neg hs_src, if_neg hs_tgt] at hj_s
        by_cases ht_src : t = src
        · rw [if_pos ht_src] at hj_t
          have hj_t_src : j ∈ P.groups src := (Finset.mem_sdiff.mp hj_t).1
          exact h_pairwise s src hs_src hj_s hj_t_src
        · by_cases ht_tgt : t = tgt
          · rw [if_neg ht_src, if_pos ht_tgt] at hj_t
            rcases Finset.mem_union.mp hj_t with hjt_tgt | hjt_dims
            · exact h_pairwise s tgt hs_tgt hj_s (ht_tgt ▸ hjt_tgt)
            · exact h_pairwise s src hs_src hj_s (h_src hjt_dims)
          · rw [if_neg ht_src, if_neg ht_tgt] at hj_t
            exact h_pairwise s t hst hj_s hj_t


-- ════════════════════════════════════════════════════════════════
-- §5. DESTRUCTION (Return to Reserve)
--
-- Removing a group's dims makes them available for reallocation.
-- The resulting partition has an empty group at the destroyed
-- regime's index.
-- ════════════════════════════════════════════════════════════════

/-- Destroy regime r: set its group to empty.
    Dims return to reserve. -/
def DynamicPartition.destroy {n R : ℕ}
    (P : DynamicPartition n R) (r : Fin R) :
    DynamicPartition n R where
  groups := fun s => if s = r then ∅ else P.groups s
  disjoint := by
    intro s t hst
    by_cases hsr : s = r <;> by_cases htr : t = r
    · exact absurd (hsr.trans htr.symm) hst
    · subst hsr; simp [Finset.disjoint_empty_left]
    · subst htr; simp [Finset.disjoint_empty_right]
    · simp [if_neg hsr, if_neg htr]; exact P.disjoint s t hst

/-- **Theorem (Destroyed Dims are Reserve).**
    After destruction, the destroyed regime's dims are
    not in any group. -/
theorem destroy_dims_are_reserve {n R : ℕ}
    (P : DynamicPartition n R) (r : Fin R)
    (j : Fin n) (hj : j ∈ P.groups r) :
    (P.destroy r).isReserve j := by
  intro s
  simp [DynamicPartition.destroy]
  by_cases hsr : s = r
  · subst hsr; simp
  · simp [if_neg hsr]
    intro hmem
    exact absurd (Finset.disjoint_left.mp (P.disjoint r s (Ne.symm hsr)) hj) (not_not.mpr hmem)

/-- **Theorem (Destruction Preserves Others).**
    Destroying regime r does not change any other group's dims. -/
theorem destroy_preserves_others {n R : ℕ}
    (P : DynamicPartition n R) (r s : Fin R) (hrs : s ≠ r) :
    (P.destroy r).groups s = P.groups s := by
  simp [DynamicPartition.destroy, if_neg hrs]


-- ════════════════════════════════════════════════════════════════
-- §6. VALID PARTITION EMBEDS INTO DYNAMIC PARTITION
-- ════════════════════════════════════════════════════════════════

/-- Every ValidPartition is a DynamicPartition.
    Static partitions are a special case of dynamic ones. -/
def validToDynamic {n R : ℕ} (P : ValidPartition n R) :
    DynamicPartition n R where
  groups := P.groups
  disjoint := P.disjoint

/-- **Theorem (Static Rights Imply Dynamic Rights).**
    Any isolation property proven for DynamicPartition
    automatically holds for ValidPartition via embedding. -/
theorem static_isolation_from_dynamic {n R : ℕ}
    (P : ValidPartition n R) (d_gated : Vec n)
    (r : Fin R) (j : Fin n) (hj : j ∉ P.groups r) :
    (d_gated ⊙ indicator (P.groups r)) j = 0 :=
  dynamic_gradient_isolation (validToDynamic P) d_gated r j hj


end Schemen.RegimeGrowth
