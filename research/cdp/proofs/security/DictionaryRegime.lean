/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import MatrixIsolation
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Matrix.Mul

/-!
# DictionaryRegime — Frobenius-Isometric Factored Storage

Formalizes the Dictionary Regime primitive described in the April 16
journal entry.  Every tenant regime r stores only a compressed
coefficient vector / matrix `z`; the reconstructor `B` lives in the
operator's Regime_0.  Head reconstitution is `W_r = B · z`, with
`B`'s rows zero outside `groups(r)` so the output is automatically
confined to the regime's coordinate subspace.

## Statement of Results

**Theorem (Reconstruction Support).** If the reconstructor `B`
has every row zero outside a set `S ⊆ Fin n`, then for any
coefficient vector `z`, the reconstructed output `(B · z) i = 0`
for every `i ∉ S`.  The support of the reconstruction lives
entirely inside `S`.

**Theorem (Reconstruction Confined to Regime).** Specialized to a
ValidPartition: if `B` is row-supported on `P.groups r`, then the
reconstruction has support inside `groups(r)`.  This is the
dictionary-regime restatement of `gradient_isolation` from V1 and
`matrix_row_zero` from V7; it says *reconstitution*, not just
training, preserves the isolation partition.

**Theorem (Rotation Identity).** For any orthogonal `T ∈ O(k)`:
  `(B · T) · (Tᵀ · z) = B · z`.
Applying a dictionary rotation jointly to the reconstructor and
the tenant's coefficients leaves the reconstructed output
bit-identical.  The proof is pure matrix associativity plus the
orthogonality identity `T · Tᵀ = I`.

**Theorem (Rotation Composition).** For any two `k × k` matrices
`T₁`, `T₂`:
  `rotate(rotate(B, z, T₁), T₂) = rotate(B, z, T₁ · T₂)`.
Rotating by T₁ then T₂ is equivalent to a single rotation by
their product.  This is the audit-trail compaction claim: a chain
of rotation events in the Rosetta Stone can be summarized by the
product of its transforms.

**Theorem (Identity Rotation).** Rotating by `I` is a no-op.
Sanity check; proves the rotation primitive degenerates correctly.

## Axioms and dependencies

ZERO new axioms.  Theorems derive from:
- `matrix_row_zero` in `MatrixIsolation.lean` (V7)
- `unique_membership` in `CapacitySecurity.lean` (V4)
- Mathlib's `Matrix.mul_assoc`, `Matrix.mul_mulVec`,
  `Matrix.transpose_mul_self_eq_one` (for orthogonal matrices)

## Proof Architecture

```
V1 (GateSecurity)     ── gradient_isolation, forward_isolation
V4 (CapacitySecurity) ── ValidPartition, unique_membership
V7 (MatrixIsolation)  ── matrix_row_zero, matrix_column_zero
│
V8 (this file)
 ├── §1  Reconstruction support (row-supported B ⟹ confined output)
 ├── §2  Partition-level corollary
 ├── §3  Rotation identity (matMul associativity + T Tᵀ = I)
 └── §4  Rotation composition + identity rotation
```
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.DictionaryRegime

open Schemen Schemen.Security Schemen.SecurityV4 Schemen.MatrixIsolation
open scoped Matrix


-- ════════════════════════════════════════════════════════════════
-- §1. RECONSTRUCTION SUPPORT
--
-- Reconstruction from a row-supported dictionary stays inside the
-- row-support set.  This is the core confinement claim for the
-- dictionary regime.
-- ════════════════════════════════════════════════════════════════

/-- A reconstructor B : Mat n k is "row-supported in S" if every
    row outside S is identically zero. -/
def RowSupportedIn {n k : ℕ} (B : Mat n k) (S : Finset (Fin n)) : Prop :=
  ∀ i : Fin n, i ∉ S → ∀ j : Fin k, B i j = 0

/-- Reconstruction of a head row by linear combination of B's
    columns weighted by z.  This is the scalar form of `B · z`. -/
def reconstruct {n k : ℕ} (B : Mat n k) (z : Vec k) : Vec n :=
  fun i => Finset.univ.sum (fun j : Fin k => B i j * z j)

/-- **Theorem (Reconstruction Support Zero Outside).**
    If B is row-supported in S, the reconstructed output is zero
    at every coordinate outside S.  The support of `B · z` is a
    subset of S regardless of z. -/
theorem reconstruction_support_zero_outside {n k : ℕ}
    (B : Mat n k) (S : Finset (Fin n))
    (hB : RowSupportedIn B S)
    (z : Vec k) (i : Fin n) (hi : i ∉ S) :
    reconstruct B z i = 0 := by
  unfold reconstruct
  apply Finset.sum_eq_zero
  intro j _
  rw [hB i hi j]
  ring


-- ════════════════════════════════════════════════════════════════
-- §2. PARTITION-LEVEL CONFINEMENT
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Reconstruction Confined to Regime).**
    Under a ValidPartition P and regime r, a reconstructor that is
    row-supported on `P.groups r` produces an output with support
    entirely inside `P.groups r`.  This is the dictionary-regime
    companion of the V7 matrix isolation theorems. -/
theorem reconstruction_confined_to_regime {n k R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (B : Mat n k)
    (hB : RowSupportedIn B (P.groups r))
    (z : Vec k) (i : Fin n) (hi : i ∉ P.groups r) :
    reconstruct B z i = 0 :=
  reconstruction_support_zero_outside B (P.groups r) hB z i hi

/-- **Theorem (Cross-Regime Reconstruction Invisibility).**
    A regime-r-supported reconstructor produces zero at every
    coordinate belonging to any other regime s ≠ r.  Two regimes'
    dictionary-regime heads have disjoint support. -/
theorem reconstruction_cross_regime_zero {n k R : ℕ}
    (P : ValidPartition n R) (r s : Fin R) (hrs : r ≠ s)
    (B : Mat n k)
    (hB : RowSupportedIn B (P.groups r))
    (z : Vec k) (i : Fin n) (hi : i ∈ P.groups s) :
    reconstruct B z i = 0 := by
  have hi_not_r : i ∉ P.groups r :=
    fun hir => hrs (unique_membership P i r s hir hi)
  exact reconstruction_confined_to_regime P r B hB z i hi_not_r


-- ════════════════════════════════════════════════════════════════
-- §3. ROTATION IDENTITY
--
-- Orthogonal T ∈ O(k) applied jointly to (B, z) leaves the
-- reconstructed output invariant.  The proof uses Mathlib's
-- `Matrix.mul_mulVec` (associativity) and the orthogonality
-- condition `T · Tᵀ = 1`.
-- ════════════════════════════════════════════════════════════════

/-- A matrix is orthogonal if its transpose is its inverse on one side.
    For square real matrices this implies both sides. -/
def IsOrthogonal {k : ℕ} (T : Matrix (Fin k) (Fin k) ℝ) : Prop :=
  T * Tᵀ = 1

/-- **Theorem (Rotation Identity).**
    For any orthogonal T, the joint rotation `(B · T, Tᵀ · z)`
    reconstructs the same vector as `(B, z)`.

    Structural content: dictionary rotation is a free change of
    basis on the compression coordinates; tenant output is
    invariant.  Security content: the operator can re-key the
    dictionary at will; tenant behaviour is unchanged. -/
theorem rotation_identity {n k : ℕ}
    (B : Matrix (Fin n) (Fin k) ℝ)
    (T : Matrix (Fin k) (Fin k) ℝ)
    (z : Fin k → ℝ)
    (hT : IsOrthogonal T) :
    (B * T).mulVec (Tᵀ.mulVec z) = B.mulVec z := by
  have hBT : (B * T) * Tᵀ = B := by
    rw [Matrix.mul_assoc, show T * Tᵀ = 1 from hT, Matrix.mul_one]
  calc (B * T).mulVec (Tᵀ.mulVec z)
      = ((B * T) * Tᵀ).mulVec z := by rw [Matrix.mulVec_mulVec]
    _ = B.mulVec z              := by rw [hBT]

/-- **Theorem (Rotation by Identity is a No-Op).**
    `(B · I, Iᵀ · z) = (B, z)` trivially.  A sanity check that
    rotating by the identity matrix does nothing. -/
theorem rotation_by_identity {n k : ℕ}
    (B : Matrix (Fin n) (Fin k) ℝ)
    (z : Fin k → ℝ) :
    (B * (1 : Matrix (Fin k) (Fin k) ℝ)).mulVec
      ((1 : Matrix (Fin k) (Fin k) ℝ)ᵀ.mulVec z)
    = B.mulVec z := by
  rw [Matrix.mul_one, Matrix.transpose_one, Matrix.one_mulVec]


-- ════════════════════════════════════════════════════════════════
-- §4. ROTATION COMPOSITION
--
-- Rotating by T₁ then T₂ is equivalent to rotating by T₁ · T₂.
-- A chain of Rosetta Stone entries can be summarized by the
-- product of its transforms.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Rotation Composition).**
    For any two square k × k matrices T₁, T₂, applying them in
    sequence is equivalent to a single application of their
    product:

      (B · T₁) · T₂  =  B · (T₁ · T₂)
      (T₁ · T₂)ᵀ · z  =  T₂ᵀ · (T₁ᵀ · z)

    Therefore rotating by T₁ then T₂ on the (B, z) pair is
    algebraically identical to rotating by the single matrix
    `T₁ · T₂`.  This is associativity-as-audit-compaction: a chain
    of rotation events can be condensed to a single entry whose
    transform is the matrix product of the chain. -/
theorem rotation_composition {n k : ℕ}
    (B : Matrix (Fin n) (Fin k) ℝ)
    (T1 T2 : Matrix (Fin k) (Fin k) ℝ)
    (z : Fin k → ℝ) :
    ((B * T1) * T2).mulVec (T2ᵀ.mulVec (T1ᵀ.mulVec z))
    = (B * (T1 * T2)).mulVec ((T1 * T2)ᵀ.mulVec z) := by
  calc ((B * T1) * T2).mulVec (T2ᵀ.mulVec (T1ᵀ.mulVec z))
      = ((B * T1) * T2 * T2ᵀ).mulVec (T1ᵀ.mulVec z)      := by
        rw [Matrix.mulVec_mulVec]
    _ = ((B * T1) * T2 * T2ᵀ * T1ᵀ).mulVec z             := by
        rw [Matrix.mulVec_mulVec]
    _ = ((B * (T1 * T2)) * (T2ᵀ * T1ᵀ)).mulVec z          := by
        rw [show (B * T1) * T2 * T2ᵀ * T1ᵀ
              = (B * (T1 * T2)) * (T2ᵀ * T1ᵀ) by
            rw [Matrix.mul_assoc B T1 T2, Matrix.mul_assoc,
                Matrix.mul_assoc]]
    _ = ((B * (T1 * T2)) * (T1 * T2)ᵀ).mulVec z          := by
        rw [← Matrix.transpose_mul]
    _ = (B * (T1 * T2)).mulVec ((T1 * T2)ᵀ.mulVec z)     := by
        rw [← Matrix.mulVec_mulVec]


-- ════════════════════════════════════════════════════════════════
-- §5. INTEGRATION WITH V7 MATRIX ISOLATION
--
-- The reconstruction theorem from §1 is a direct corollary of
-- matrix_row_zero from V7 when applied coordinate-by-coordinate.
-- We state this connection explicitly to make the "zero new
-- axioms" claim auditable.
-- ════════════════════════════════════════════════════════════════

/-- **Lemma (Row-supported reduces to indicator masking).**
    A reconstructor that is row-supported in S is equivalent
    (pointwise) to the product of the indicator of S with an
    arbitrary matrix.  The hmul-indicator form is what V7's
    isolation theorems consume. -/
theorem row_supported_via_indicator {n k : ℕ}
    (B : Mat n k) (S : Finset (Fin n))
    (hB : RowSupportedIn B S) (i : Fin n) (j : Fin k) :
    B i j = indicator S i * B i j := by
  by_cases hi : i ∈ S
  · rw [indicator_mem _ i hi]; ring
  · rw [hB i hi j, indicator_not_mem _ i hi]; ring


-- ════════════════════════════════════════════════════════════════
-- §6. LINEARITY OF RECONSTRUCTION
--
-- Reconstruction `reconstruct B z` is linear in z and in B.
-- These are not confinement theorems; they are structural
-- identities that let us reason about reconstruction as a
-- well-behaved linear map.  Useful downstream when composing
-- dictionary reconstruction with other linear transforms
-- (adapter injection, spectral projection, migration rotation).
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Reconstruction is Additive in z) — Pointwise.** -/
theorem reconstruct_add_pointwise {n k : ℕ}
    (B : Mat n k) (z1 z2 : Vec k) (i : Fin n) :
    reconstruct B (fun j => z1 j + z2 j) i
    = reconstruct B z1 i + reconstruct B z2 i := by
  unfold reconstruct
  simp only [mul_add, Finset.sum_add_distrib]

/-- **Theorem (Reconstruction is Additive in z).**
    reconstruct B (z₁ + z₂) = reconstruct B z₁ + reconstruct B z₂. -/
theorem reconstruct_add {n k : ℕ} (B : Mat n k) (z1 z2 : Vec k) :
    reconstruct B (fun j => z1 j + z2 j)
    = fun i => reconstruct B z1 i + reconstruct B z2 i := by
  funext i
  exact reconstruct_add_pointwise B z1 z2 i

/-- **Theorem (Reconstruction is Homogeneous in z) — Pointwise.** -/
theorem reconstruct_smul_pointwise {n k : ℕ}
    (B : Mat n k) (c : ℝ) (z : Vec k) (i : Fin n) :
    reconstruct B (fun j => c * z j) i = c * reconstruct B z i := by
  unfold reconstruct
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro j _
  ring

/-- **Theorem (Reconstruction is Homogeneous in z).**
    reconstruct B (c · z) = c · reconstruct B z. -/
theorem reconstruct_smul {n k : ℕ} (B : Mat n k) (c : ℝ) (z : Vec k) :
    reconstruct B (fun j => c * z j)
    = fun i => c * reconstruct B z i := by
  funext i
  exact reconstruct_smul_pointwise B c z i

/-- **Theorem (Zero Coefficients ⟹ Zero Output).**
    reconstruct B 0 = 0 regardless of the dictionary. -/
theorem reconstruct_zero_coeffs {n k : ℕ} (B : Mat n k) (i : Fin n) :
    reconstruct B (fun _ => (0 : ℝ)) i = 0 := by
  unfold reconstruct
  apply Finset.sum_eq_zero
  intro j _
  ring

/-- **Theorem (Zero Dictionary ⟹ Zero Output — the Permanent
    Connection claim).**  With B = 0, reconstruction is
    identically zero regardless of the coefficients.  This is the
    formal form of the operator-required-for-inference property:
    without Regime_0's reconstructor, the tenant's coefficients
    produce no signal. -/
theorem reconstruct_zero_dict {n k : ℕ} (z : Vec k) (i : Fin n) :
    reconstruct (fun _ _ => (0 : ℝ)) z i = 0 := by
  unfold reconstruct
  apply Finset.sum_eq_zero
  intro j _
  ring

/-- **Theorem (Reconstruction is Linear in z).**
    Combined additivity + homogeneity.  A direct consequence of
    the two pointwise lemmas above. -/
theorem reconstruct_linear_in_z {n k : ℕ} (B : Mat n k)
    (c1 c2 : ℝ) (z1 z2 : Vec k) (i : Fin n) :
    reconstruct B (fun j => c1 * z1 j + c2 * z2 j) i
    = c1 * reconstruct B z1 i + c2 * reconstruct B z2 i := by
  rw [reconstruct_add_pointwise B (fun j => c1 * z1 j) (fun j => c2 * z2 j) i]
  rw [reconstruct_smul_pointwise B c1 z1 i]
  rw [reconstruct_smul_pointwise B c2 z2 i]


-- ════════════════════════════════════════════════════════════════
-- §7. ROTATION PRESERVES ROW SUPPORT
--
-- Applying a dictionary rotation T to a row-supported B produces
-- a matrix that is still row-supported in the same set.  This is
-- the fact that lets us rotate a tenant's dictionary without
-- leaving its coordinate subspace: every V1-V8 isolation theorem
-- continues to apply after rotation.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Rotation Preserves Row Support).**
    If `B : Mat n k` is row-supported in `S`, then for any
    `T : Mat k k`, the product `B * T` (as matrix multiplication)
    is also row-supported in `S`.

    Proof: `(B * T) i l = ∑_j B i j * T j l`.  If `i ∉ S` then
    every `B i j = 0`, so every term is zero, so the sum is zero.

    Security consequence: operator-driven dictionary rotation
    leaves tenant output fully confined to the tenant's coordinate
    subspace. -/
theorem rotation_preserves_row_support {n k : ℕ}
    (B : Matrix (Fin n) (Fin k) ℝ)
    (T : Matrix (Fin k) (Fin k) ℝ)
    (S : Finset (Fin n))
    (hB : ∀ i : Fin n, i ∉ S → ∀ j : Fin k, B i j = 0)
    (i : Fin n) (hi : i ∉ S) (l : Fin k) :
    (B * T) i l = 0 := by
  rw [Matrix.mul_apply]
  apply Finset.sum_eq_zero
  intro j _
  rw [hB i hi j]
  ring


-- ════════════════════════════════════════════════════════════════
-- §8. FROBENIUS-VEC ISOMETRY
--
-- The headline identity: the Frobenius norm of a matrix equals
-- the L² norm of its flattening.  We prove it at the level of
-- squared norms (which keeps everything in ℝ-arithmetic without
-- requiring Mathlib's InnerProductSpace machinery):
--
--   ∑_{i,j} (A i j)²  =  ∑_k (A (π₁(k)) (π₂(k)))²
--
-- where π is `finProdFinEquiv.symm : Fin (m·n) ≃ Fin m × Fin n`.
--
-- This is the formal underpinning of the claim that a matrix
-- of shape m × n fits losslessly into a coordinate subspace of
-- dimension m·n, and that Frobenius-similarity is preserved in
-- the compressed representation.
-- ════════════════════════════════════════════════════════════════

/-- The column-major flattening of a matrix via Mathlib's
    canonical Fin × Fin ≃ Fin bijection. -/
def flat {m n : ℕ} (A : Mat m n) : Vec (m * n) :=
  fun k => A (finProdFinEquiv.symm k).1 (finProdFinEquiv.symm k).2

/-- **Theorem (Double Sum Equals Product Sum).**
    The Frobenius-style double sum decomposes into a single sum
    over the Cartesian product index. -/
theorem frobenius_sq_as_product_sum {m n : ℕ} (A : Mat m n) :
    (∑ i : Fin m, ∑ j : Fin n, (A i j) ^ 2)
    = ∑ p : Fin m × Fin n, (A p.1 p.2) ^ 2 :=
  (Fintype.sum_prod_type (fun p : Fin m × Fin n => (A p.1 p.2) ^ 2)).symm

/-- **Theorem (Frobenius Squared = Flat L² Squared).**
    The sum of squares of all matrix entries equals the sum of
    squares of the flattened vector.  This is the norm-squared
    form of `||A||_F = ||vec(A)||_2`. -/
theorem frobenius_sq_eq_flat_l2_sq {m n : ℕ} (A : Mat m n) :
    (∑ i : Fin m, ∑ j : Fin n, (A i j) ^ 2)
    = ∑ k : Fin (m * n), (flat A k) ^ 2 := by
  rw [frobenius_sq_as_product_sum]
  rw [← Equiv.sum_comp finProdFinEquiv
        (fun k => (flat A k) ^ 2)]
  apply Finset.sum_congr rfl
  intro p _
  unfold flat
  rw [Equiv.symm_apply_apply]

/-- **Corollary (Frobenius Inner Product = Flat Dot Product).**
    For two matrices A, B of the same shape:
      ∑_{i,j} A i j * B i j  =  ∑_k (flat A k) * (flat B k).

    The inner-product form of the isometry — it's what makes
    Frobenius-based similarity (CKA, subspace alignment) preserved
    under vec-compression. -/
theorem frobenius_inner_eq_flat_inner {m n : ℕ} (A B : Mat m n) :
    (∑ i : Fin m, ∑ j : Fin n, A i j * B i j)
    = ∑ k : Fin (m * n), flat A k * flat B k := by
  rw [show (∑ i : Fin m, ∑ j : Fin n, A i j * B i j)
      = ∑ p : Fin m × Fin n, A p.1 p.2 * B p.1 p.2 from
      (Fintype.sum_prod_type (fun p : Fin m × Fin n =>
        A p.1 p.2 * B p.1 p.2)).symm]
  rw [← Equiv.sum_comp finProdFinEquiv
        (fun k => flat A k * flat B k)]
  apply Finset.sum_congr rfl
  intro p _
  unfold flat
  rw [Equiv.symm_apply_apply]

/-- **Theorem (Flat preserves zero).**
    A matrix is zero iff its flattening is zero.  This is the
    identity that lets us reason about "a regime slot is empty"
    interchangeably in matrix form and vector form. -/
theorem flat_eq_zero_iff {m n : ℕ} (A : Mat m n) :
    (∀ k, flat A k = 0) ↔ (∀ i j, A i j = 0) := by
  constructor
  · intro h i j
    have := h (finProdFinEquiv (i, j))
    unfold flat at this
    rwa [Equiv.symm_apply_apply] at this
  · intro h k
    unfold flat
    apply h


end Schemen.DictionaryRegime
