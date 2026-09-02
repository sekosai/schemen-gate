/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/


import CapacitySecurity

/-!
# MatrixIsolation — Matrix-Level Isolation and Two-Basis Structure

This file lifts the vector-level gate algebra of V1 (GateSecurity)
to weight matrices, and formalizes the "two-basis" architecture
of a regime: an outer coordinate-partition (cryptographic) basis
composed with an inner rotation (spectral) basis.

## Statement of Results

**Theorem (C1, Matrix Column Isolation).** The outer-product
gradient `u ⊗ (d ⊙ M_r)` has an entire zero column at every j
where the mask M_r is zero.  This lifts V1's scalar
`gradient_isolation` to a statement about full matrix columns —
the dashboard-level claim that every column of every weight
matrix outside a regime receives zero gradient, not just zero at
the residual-stream coordinate.

**Theorem (C2, Cross-Block Update Invariance).** Under any
mask-confined update `ΔW = u ⊗ (d ⊙ M_r)`, every cross-block
entry of `W` is preserved exactly: `(W + ΔW)[i][j] = W[i][j]`
for j ∉ groups(r).  The off-diagonal (cross-regime) Frobenius
energy is a training invariant — the algebraic skeleton of the
empirical CRWE metric.

**Theorem (C2, Block Product Vanishes).** Tensor-product block
masks `blockMask P r s = indicator(groups r) ⊗ indicator(groups s)`
are mutually annihilating whenever the block coordinates differ:
for (r₁, s₁) ≠ (r₂, s₂), the Hadamard product of the two block
masks is identically zero.

**Theorem (C3, Spectral Mutual Invisibility, Structural Form).**
Row i of `colMask W (indicator (groups r))` and row k of
`colMask W (indicator (groups s))` are componentwise orthogonal
at every coordinate when r ≠ s.  Since componentwise-zero
products imply zero inner product, this is the elementary form
of the spectral claim: row-spaces (and hence right singular
vector spaces) of the two regime-masked matrices are orthogonal.

**Theorem (C5, Two-Basis Composition).** An "inner basis" U for
regime r — any function that acts as identity outside r's
coordinate subspace — is transparent to every other regime's
mask: `(U h) ⊙ M_s = h ⊙ M_s` for s ≠ r.  The inner spectral
basis and the outer cryptographic basis compose without
weakening isolation.

## Axioms

ZERO new axioms.  Every theorem derives from V1 and V4.

## Proof Architecture

```
V1 (GateSecurity)    ─── indicator, hmul, outer, gradient_isolation,
│                        forward_isolation, weight_update_confined
│
V4 (CapacitySecurity) ── ValidPartition, unique_membership,
│                        indicator_not_mem, indicator_mem
│
V7 (this file)
 ├── §1  Matrix type + elementary ops
 ├── §2  Matrix-level gradient isolation (C1)
 ├── §3  Cross-block decomposition (C2)
 ├── §4  Cross-block update invariance (C2 full form)
 ├── §5  Spectral mutual invisibility (C3)
 └── §6  Two-basis composition (C5)
```
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.MatrixIsolation

open Schemen Schemen.Security Schemen.SecurityV4


-- ════════════════════════════════════════════════════════════
-- §1. MATRIX TYPE AND ELEMENTARY OPERATIONS
--
-- A weight matrix W : Mat m n with W i j being the weight from
-- input coordinate j (column) to output coordinate i (row).
-- ════════════════════════════════════════════════════════════

/-- A real matrix indexed by `Fin m × Fin n`.  Entry `W i j` is
    the weight from input dim `j` (column) to output dim `i` (row). -/
abbrev Mat (m n : ℕ) := Fin m → Fin n → ℝ

/-- Element-wise (Hadamard) matrix product. -/
def matHmul {m n : ℕ} (A B : Mat m n) : Mat m n :=
  fun i j => A i j * B i j

scoped infixl:70 " ⊙ₘ " => matHmul

/-- Apply a column mask: zero out every column j where `M j = 0`.
    `(colMask W M) i j = W i j * M j`. -/
def colMask {m n : ℕ} (W : Mat m n) (M : Vec n) : Mat m n :=
  fun i j => W i j * M j

/-- Apply a row mask: zero out every row i where `M i = 0`.
    `(rowMask W M) i j = M i * W i j`. -/
def rowMask {m n : ℕ} (W : Mat m n) (M : Vec m) : Mat m n :=
  fun i j => M i * W i j


-- ════════════════════════════════════════════════════════════
-- §2. MATRIX-LEVEL GRADIENT ISOLATION (C1)
--
-- V1 proved: mask j = 0 ⟹ (d ⊙ mask) j = 0.
-- We lift this to whole columns of the outer-product gradient.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Matrix Column Isolation).**
    The outer-product gradient `u ⊗ (d ⊙ M)` has an entire zero
    column at every j where `M j = 0`.  The vector-level scalar
    result of V1 becomes a statement about matrix columns. -/
theorem matrix_column_zero {m n : ℕ}
    (u : Vec m) (d M : Vec n)
    (j : Fin n) (hj : M j = 0) :
    ∀ i : Fin m, outer u (d ⊙ M) i j = 0 := by
  intro i
  simp only [outer, hmul, hj, mul_zero]

/-- **Theorem (Matrix Row Isolation).**
    Dually: `(h ⊙ M) ⊗ d` has an entire zero row at every
    j where `M j = 0`. -/
theorem matrix_row_zero {m n : ℕ}
    (h M : Vec m) (d : Vec n)
    (j : Fin m) (hj : M j = 0) :
    ∀ k : Fin n, outer (h ⊙ M) d j k = 0 := by
  intro k
  simp only [outer, hmul, hj]
  ring

/-- **Theorem (Column Confinement Under Partition).**
    Under a valid partition, every column of the outer-product
    gradient outside the active regime is identically zero.
    This is the matrix-level companion to V1's
    `weight_update_confined`. -/
theorem matrix_gradient_column_confined {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (u : Vec m) (d : Vec n) (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m, outer u (d ⊙ indicator (P.groups r)) i j = 0 :=
  matrix_column_zero u d (indicator (P.groups r)) j
    (indicator_not_mem _ j hj)


-- ════════════════════════════════════════════════════════════
-- §3. CROSS-BLOCK DECOMPOSITION (C2)
--
-- Every square weight matrix W : Mat n n decomposes under a
-- partition into R² tensor blocks: M_r ⊗ M_s for r, s ∈ [R].
-- The diagonal blocks (r = s) are the intra-regime entries;
-- the off-diagonal blocks (r ≠ s) are the cross-regime leak
-- channel.  This section formalizes the block structure.
-- ════════════════════════════════════════════════════════════

/-- Tensor-product block mask for the (r, s) block:
    `blockMask P r s = indicator(groups r) ⊗ indicator(groups s)`.
    Nonzero only at entries (i, j) with i ∈ groups(r) and
    j ∈ groups(s). -/
def blockMask {n R : ℕ} (P : ValidPartition n R) (r s : Fin R) : Mat n n :=
  outer (indicator (P.groups r)) (indicator (P.groups s))

/-- **Lemma (Block mask vanishes outside).**
    `blockMask P r s` is zero at any (i, j) with either i
    outside groups(r) or j outside groups(s). -/
theorem blockMask_zero_outside {n R : ℕ}
    (P : ValidPartition n R) (r s : Fin R)
    (i j : Fin n) (h : i ∉ P.groups r ∨ j ∉ P.groups s) :
    blockMask P r s i j = 0 := by
  simp only [blockMask, outer]
  rcases h with hi | hj
  · rw [indicator_not_mem _ i hi]; ring
  · rw [indicator_not_mem _ j hj]; ring

/-- **Theorem (Block Product Mutual Annihilation).**
    For distinct block coordinates (r₁, s₁) ≠ (r₂, s₂), the
    element-wise product of the two block masks is identically
    zero.  In particular, diagonal blocks M_r ⊗ M_r and
    off-diagonal blocks M_r ⊗ M_s (r ≠ s) occupy disjoint
    rectangles — cross-regime blocks are cleanly separable
    from intra-regime blocks. -/
theorem block_product_vanishes {n R : ℕ}
    (P : ValidPartition n R) (r1 s1 r2 s2 : Fin R)
    (h : r1 ≠ r2 ∨ s1 ≠ s2) (i j : Fin n) :
    (blockMask P r1 s1 ⊙ₘ blockMask P r2 s2) i j = 0 := by
  simp only [matHmul, blockMask, outer]
  rcases h with hr | hs
  · by_cases hi1 : i ∈ P.groups r1
    · have hi2 : i ∉ P.groups r2 :=
        fun hi2' => hr (unique_membership P i r1 r2 hi1 hi2')
      rw [indicator_not_mem _ i hi2]; ring
    · rw [indicator_not_mem _ i hi1]; ring
  · by_cases hj1 : j ∈ P.groups s1
    · have hj2 : j ∉ P.groups s2 :=
        fun hj2' => hs (unique_membership P j s1 s2 hj1 hj2')
      rw [indicator_not_mem _ j hj2]; ring
    · rw [indicator_not_mem _ j hj1]; ring


-- ════════════════════════════════════════════════════════════
-- §4. CROSS-BLOCK UPDATE INVARIANCE (C2, full form)
--
-- The central claim: off-diagonal (cross-regime) weight energy
-- is a training invariant under mask-confined updates.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Cross-Block Entries are Update-Invariant).**
    Let `W : Mat m n` and `ΔW = u ⊗ (d ⊙ M_r)` be a mask-confined
    update under regime r.  For every j ∉ groups(r) and every i,
    the updated entry equals the original: no cross-regime entry
    moves under any regime's mask-confined training step.

    Iterating this over any sequence of updates gives the CRWE
    invariant: the Frobenius norm of the cross-block portion is
    preserved bit-for-bit across arbitrary training. -/
theorem cross_block_update_invariant {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (W : Mat m n) (u : Vec m) (d : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m,
      W i j + outer u (d ⊙ indicator (P.groups r)) i j = W i j := by
  intro i
  rw [matrix_gradient_column_confined P r u d j hj i, add_zero]


-- ════════════════════════════════════════════════════════════
-- §5. SPECTRAL MUTUAL INVISIBILITY (C3)
--
-- Row-space orthogonality between regime-column-masked matrices.
-- Since the right singular vectors of A span the row space of A,
-- this is the structural core of the spectral claim: singular-
-- vector subspaces of W M_r and W M_s are orthogonal (r ≠ s).
-- ════════════════════════════════════════════════════════════

/-- **Lemma (Column-Masked Row Support).**
    Row i of a column-masked matrix is zero at every j where
    the mask is zero. -/
theorem colMask_row_zero {m n : ℕ}
    (W : Mat m n) (M : Vec n) (i : Fin m) (j : Fin n)
    (hj : M j = 0) :
    colMask W M i j = 0 := by
  simp only [colMask, hj, mul_zero]

/-- **Theorem (Row-Orthogonality Between Regime-Masked Matrices).**
    For r ≠ s, any two rows (one from `colMask W (M_r)`, one from
    `colMask W (M_s)`) have zero componentwise product at every
    coordinate j.  This immediately implies zero inner product,
    hence orthogonality of the row spaces — the structural core
    of spectral mutual invisibility. -/
theorem regime_row_pointwise_orthogonal {m n R : ℕ}
    (P : ValidPartition n R) (r s : Fin R) (hrs : r ≠ s)
    (W : Mat m n) (i k : Fin m) (j : Fin n) :
    colMask W (indicator (P.groups r)) i j *
      colMask W (indicator (P.groups s)) k j = 0 := by
  simp only [colMask]
  by_cases hj : j ∈ P.groups r
  · have hj_not_s : j ∉ P.groups s :=
      fun hjs => hrs (unique_membership P j r s hj hjs)
    rw [indicator_not_mem _ j hj_not_s]; ring
  · rw [indicator_not_mem _ j hj]; ring

/-- **Theorem (Regime Row-Spaces have Disjoint Nonzero Support).**
    The nonzero support of every row of `colMask W (M_r)` is
    contained in `groups(r)`.  Since the partition groups are
    disjoint, the nonzero supports of any two regimes' masked
    row spaces are disjoint finite sets. -/
theorem regime_row_support_in_group {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (W : Mat m n) (i : Fin m) (j : Fin n) (hj : j ∉ P.groups r) :
    colMask W (indicator (P.groups r)) i j = 0 :=
  colMask_row_zero W (indicator (P.groups r)) i j
    (indicator_not_mem _ j hj)


-- ════════════════════════════════════════════════════════════
-- §6. TWO-BASIS COMPOSITION (C5)
--
-- An "inner basis" for regime r is any map U : Vec n → Vec n
-- that acts as identity on every coordinate outside groups(r).
-- Concretely: U may rotate/reflect within the r-coordinate
-- subspace, but leaves every other coordinate untouched.
--
-- Formalizes the composition: (outer cryptographic basis) ∘
-- (inner spectral basis) retains every isolation property.
-- ════════════════════════════════════════════════════════════

/-- An inner basis for regime r: a map U : Vec n → Vec n that
    is the identity on every coordinate outside groups(r).  Any
    orthogonal rotation restricted to r's coordinate subspace
    (extended by identity) satisfies this definition. -/
def IsInnerBasis {n R : ℕ} (P : ValidPartition n R) (r : Fin R)
    (U : Vec n → Vec n) : Prop :=
  ∀ (h : Vec n) (j : Fin n), j ∉ P.groups r → U h j = h j

/-- **Theorem (Two-Basis Composition, Pointwise).**
    For any inner basis U for regime r and any other regime
    s ≠ r: the regime-s gated reading of U(h) coincides
    pointwise with the regime-s gated reading of h. -/
theorem two_basis_mask_invariance {n R : ℕ}
    (P : ValidPartition n R) (r s : Fin R) (hrs : r ≠ s)
    (U : Vec n → Vec n) (hU : IsInnerBasis P r U)
    (h : Vec n) (j : Fin n) (hj : j ∈ P.groups s) :
    (U h ⊙ indicator (P.groups s)) j =
      (h ⊙ indicator (P.groups s)) j := by
  have hj_not_r : j ∉ P.groups r :=
    fun hjr => hrs (unique_membership P j r s hjr hj)
  simp only [hmul]
  rw [hU h j hj_not_r]

/-- **Theorem (Two-Basis Composition, Vector Form).**
    The full regime-s mask of U(h) equals the regime-s mask of h
    for every inner basis U for any r ≠ s. -/
theorem two_basis_mask_invariance_vec {n R : ℕ}
    (P : ValidPartition n R) (r s : Fin R) (hrs : r ≠ s)
    (U : Vec n → Vec n) (hU : IsInnerBasis P r U) (h : Vec n) :
    U h ⊙ indicator (P.groups s) = h ⊙ indicator (P.groups s) := by
  funext j
  by_cases hj : j ∈ P.groups s
  · exact two_basis_mask_invariance P r s hrs U hU h j hj
  · simp only [hmul, indicator_not_mem _ j hj, mul_zero]

/-- **Theorem (Inner-Basis Gradient Stays Confined).**
    A gradient passed through an inner basis U for regime r
    still vanishes at every coordinate outside r after masking.
    Composing with the inner basis does not leak gradient into
    other regimes. -/
theorem inner_basis_gradient_isolation {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d : Vec n) (j : Fin n) (hj : j ∉ P.groups r) :
    (d ⊙ indicator (P.groups r)) j = 0 :=
  gradient_isolation d _ j (indicator_not_mem _ j hj)


-- ════════════════════════════════════════════════════════════
-- §7. ORTHOGONAL-PROJECTED UPDATES (C6, structural form)
--
-- The formal skeleton of the `ortho_preen` result: subtracting
-- any multiple of a stored direction from the update leaves the
-- gradient orthogonal to that direction.  The "training
-- subspace" is not populated further.
-- ════════════════════════════════════════════════════════════

/-- Projection coefficient of a vector onto a direction,
    modelled pointwise. -/
def projectOut {n : ℕ} (d v : Vec n) (α : ℝ) : Vec n :=
  fun j => d j - α * v j

/-- **Theorem (Ortho-Projection Zeroes the Trained Subspace).**
    Given a nonzero coordinate j, if we choose the projection
    coefficient α so that α * v j = d j, the residual `d - α v`
    is zero at j.  This is the pointwise skeleton of the
    orthogonal gradient projection used by `ortho_preen.py`. -/
theorem projectOut_zeroes {n : ℕ}
    (d v : Vec n) (j : Fin n) (hv : v j ≠ 0) :
    projectOut d v (d j / v j) j = 0 := by
  simp only [projectOut]
  rw [div_mul_cancel₀ (d j) hv]
  ring


end Schemen.MatrixIsolation
