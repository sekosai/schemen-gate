/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import Mathlib.Data.Fin.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Card
import Mathlib.Order.Disjoint
import Mathlib.Tactic

/-!
# Serial Gate Security — Formal Verification (Tier 1.5)

Machine-checked proofs for the serial gate architecture, where each
regime gets a dedicated adapter that reads the full hidden state but
whose output is masked to the regime's coordinate partition.

The adapter is modeled as an opaque function `f : Vec n → Vec n`.
No assumptions are made about its internals.  All guarantees follow
from the mask multiplication on the adapter's output.

## What is proven

### Output Confinement
- `serial_output_confined`: masked adapter output = 0 outside partition
  (holds for ANY adapter function)

### Composition
- `serial_compose_disjoint`: the pointwise sum of two masked adapter
  outputs equals the piecewise function (f(h) on S, g(h) on T,
  0 outside S∪T).  This is the correct "unions compose" statement
  for black-box adapters; the earlier stronger form
  `(f(h)+g(h)) ⊙ 1_{S∪T}` was false without confinement hypotheses.
- `serial_compose_confined_to_union`: corollary — the combined
  output is zero outside S∪T.
- `serial_compound_excludes`: non-participating regimes receive zero
  signal from all adapters in a compound

### Orthogonality
- `serial_orthogonal`: two adapters with disjoint masks produce
  outputs whose pointwise product is zero everywhere

### Preservation
- `serial_output_preserves`: at active positions, the mask is
  transparent — adapter output passes through unchanged

### Additive Injection
- `serial_injection_confined`: additive injection h + α·(f(h) ⊙ mask)
  equals h outside the partition — the original hidden state is
  untouched at non-regime dimensions

## Relationship to GateSecurity.lean

These theorems extend GateSecurity.lean §4-§5 (orthogonality,
composability) to the serial gate setting.  The key difference:
GateSecurity.lean masks the hidden state directly (h ⊙ mask),
while SerialGateSecurity.lean masks the adapter output (f(h) ⊙ mask).
The output-level properties are identical because the mask
multiplication is the same operation in both cases.

## Empirical validation

All claims verified at two scales:
- Toy model (64 dims, 4 regimes): `poc/serial_gate_poc.py`
  - Cross-regime: 0%, output confinement: 0.0, union dot: 0.0
- DistilBERT (768 dims, 4 regimes): `poc/serial_gate_distilbert.py`
  - Leakage: 0.000000, output confinement: 0.0, union dot: 0.0
-/

set_option autoImplicit false

namespace Schemen.SerialGate

-- ════════════════════════════════════════════════════════════
-- §0. DEFINITIONS (reuse from GateSecurity where possible)
-- ════════════════════════════════════════════════════════════

/-- A vector of reals indexed by Fin n. -/
abbrev Vec (n : ℕ) := Fin n → ℝ

/-- Element-wise (Hadamard) product. -/
@[simp]
def hmul {n : ℕ} (a b : Vec n) : Vec n := fun j => a j * b j

scoped infixl:70 " ⊙ " => hmul

/-- Binary indicator function for a finite set. -/
@[simp]
def indicator {n : ℕ} (S : Finset (Fin n)) : Vec n :=
  fun j => if j ∈ S then (1 : ℝ) else (0 : ℝ)

theorem indicator_mem {n : ℕ} (S : Finset (Fin n)) (j : Fin n) (hj : j ∈ S) :
    indicator S j = 1 := by simp [indicator, hj]

theorem indicator_not_mem {n : ℕ} (S : Finset (Fin n)) (j : Fin n) (hj : j ∉ S) :
    indicator S j = 0 := by simp [indicator, hj]

/-- An adapter is an arbitrary function from Vec n to Vec n.
    We make no assumptions about its internals — it is a black box.
    The serial gate's guarantees hold for ANY adapter. -/
def Adapter (n : ℕ) := Vec n → Vec n


-- ════════════════════════════════════════════════════════════
-- §1. OUTPUT CONFINEMENT
--
-- The core serial gate property: the adapter can compute
-- anything it wants internally, but its output, when multiplied
-- by the mask, is zero outside the regime's partition.
--
-- This is the "plank with holes" — inside the plank, full
-- bandwidth; at the holes, only the regime's dims pass through.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Serial Output Confinement).**
    For ANY adapter function f, the masked output is zero at
    positions outside the regime's partition S.

    This is the fundamental serial gate guarantee.  The proof is
    trivial because indicator(S, j) = 0 when j ∉ S, and
    anything × 0 = 0.  But the trivially is the point: the
    guarantee is STRUCTURAL, not dependent on f's behavior. -/
theorem serial_output_confined {n : ℕ} (f : Adapter n)
    (h : Vec n) (S : Finset (Fin n)) (j : Fin n) (hj : j ∉ S) :
    (f h ⊙ indicator S) j = 0 := by
  have hmask : indicator S j = 0 := indicator_not_mem S j hj
  simp only [hmul, hmask, mul_zero]


-- ════════════════════════════════════════════════════════════
-- §2. PRESERVATION AT ACTIVE POSITIONS
--
-- At positions INSIDE the regime's partition, the mask is
-- transparent: the adapter output passes through unchanged.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Serial Output Preserves Active).**
    At active positions (j ∈ S), the mask is 1.0 and the
    adapter output is unchanged. -/
theorem serial_output_preserves {n : ℕ} (f : Adapter n)
    (h : Vec n) (S : Finset (Fin n)) (j : Fin n) (hj : j ∈ S) :
    (f h ⊙ indicator S) j = f h j := by
  have hmask : indicator S j = 1 := indicator_mem S j hj
  simp only [hmul, hmask, mul_one]


-- ════════════════════════════════════════════════════════════
-- §3. ORTHOGONALITY
--
-- Two serial gate adapters with disjoint masks produce
-- orthogonal outputs.  This is the same property as spatial
-- gate orthogonality (GateSecurity.lean §4), extended to
-- arbitrary adapter functions.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Serial Orthogonality).**
    Two adapters with disjoint masks produce outputs whose
    pointwise product is zero at every position.

    At any position j: either j ∈ S (so indicator(T,j)=0 by
    disjointness) or j ∉ S (so indicator(S,j)=0).  Either way,
    one factor is zero. -/
theorem serial_orthogonal {n : ℕ} (f g : Adapter n)
    (h : Vec n) (S T : Finset (Fin n))
    (hdisj : Disjoint S T) (j : Fin n) :
    (f h ⊙ indicator S) j * (g h ⊙ indicator T) j = 0 := by
  by_cases hs : j ∈ S
  · have ht : j ∉ T := Finset.disjoint_left.mp hdisj hs
    simp only [hmul, indicator_not_mem T j ht, mul_zero]
  · simp only [hmul, indicator_not_mem S j hs, mul_zero, zero_mul]

/-- Pointwise orthogonality as a function equality. -/
theorem serial_orthogonal_vec {n : ℕ} (f g : Adapter n)
    (h : Vec n) (S T : Finset (Fin n))
    (hdisj : Disjoint S T) :
    (fun j => (f h ⊙ indicator S) j * (g h ⊙ indicator T) j) =
    (fun _ => (0 : ℝ)) := by
  ext j; exact serial_orthogonal f g h S T hdisj j


-- ════════════════════════════════════════════════════════════
-- §4. COMPOSITION (UNIONS)
--
-- Two serial gate adapters compose additively: the sum of
-- their masked outputs equals the union indicator applied to
-- the pointwise sum of their raw outputs.
--
-- This is the Bill of Rights composition property: each
-- regime contributes its partition to the union, and the
-- union is the sum of contributions.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Serial Compose Disjoint — Piecewise).**
    The pointwise sum of two masked adapter outputs evaluates to:
      f(h)[j]        when j ∈ S
      g(h)[j]        when j ∈ T  (and j ∉ S, under disjointness)
      0              when j ∉ S ∪ T

    Each adapter contributes independently inside its own partition,
    and the combined output is confined to S ∪ T.  The statement is
    an unconditional pointwise equality to a piecewise function;
    disjointness is not needed for the equality (it is needed for
    the interpretation "f on S, g on T" without overlap).

    This replaces the earlier statement
    `(f h + g h) ⊙ indicator (S ∪ T)`, which did NOT hold for
    arbitrary black-box adapters — at j ∈ S the LHS is f(h)[j]
    while that formulation gave f(h)[j] + g(h)[j], which differ
    unless g(h)[j] = 0 (a confinement hypothesis f, g do not
    enjoy). -/
theorem serial_compose_disjoint {n : ℕ} (f g : Adapter n)
    (h : Vec n) (S T : Finset (Fin n))
    (j : Fin n) :
    (f h ⊙ indicator S) j + (g h ⊙ indicator T) j =
    (if j ∈ S then f h j else 0) + (if j ∈ T then g h j else 0) := by
  by_cases hs : j ∈ S <;> by_cases ht : j ∈ T <;>
    simp [hmul, indicator, hs, ht]

/-- **Corollary (Serial Compose Confined To Union).**
    For any two adapters f, g and any two partitions S, T, the
    pointwise sum of their masked outputs is zero outside S ∪ T.
    No disjointness hypothesis needed. -/
theorem serial_compose_confined_to_union {n : ℕ} (f g : Adapter n)
    (h : Vec n) (S T : Finset (Fin n))
    (j : Fin n) (hj : j ∉ S ∪ T) :
    (f h ⊙ indicator S) j + (g h ⊙ indicator T) j = 0 := by
  rw [Finset.mem_union, not_or] at hj
  rw [serial_output_confined f h S j hj.1,
      serial_output_confined g h T j hj.2, add_zero]


-- ════════════════════════════════════════════════════════════
-- §5. COMPOUND EXCLUSION
--
-- A regime not participating in a compound receives zero
-- signal from all adapters in the compound.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Serial Compound Excludes).**
    If regime U is disjoint from both S and T (the compound
    participants), then the total signal from both adapters
    is zero at every position j ∈ U. -/
theorem serial_compound_excludes {n : ℕ} (f g : Adapter n)
    (h : Vec n) (S T U : Finset (Fin n))
    (hdSU : Disjoint S U) (hdTU : Disjoint T U)
    (j : Fin n) (hj : j ∈ U) :
    (f h ⊙ indicator S) j + (g h ⊙ indicator T) j = 0 := by
  have hjS : j ∉ S := Finset.disjoint_right.mp hdSU hj
  have hjT : j ∉ T := Finset.disjoint_right.mp hdTU hj
  simp only [hmul, indicator_not_mem S j hjS, indicator_not_mem T j hjT, mul_zero, add_zero]

/-- Generalized to three adapters. -/
theorem serial_compound_excludes_three {n : ℕ} (f g k : Adapter n)
    (h : Vec n) (S T V U : Finset (Fin n))
    (hdSU : Disjoint S U) (hdTU : Disjoint T U) (hdVU : Disjoint V U)
    (j : Fin n) (hj : j ∈ U) :
    (f h ⊙ indicator S) j + (g h ⊙ indicator T) j + (k h ⊙ indicator V) j = 0 := by
  have hjS : j ∉ S := Finset.disjoint_right.mp hdSU hj
  have hjT : j ∉ T := Finset.disjoint_right.mp hdTU hj
  have hjV : j ∉ V := Finset.disjoint_right.mp hdVU hj
  simp only [hmul, indicator_not_mem S j hjS, indicator_not_mem T j hjT,
             indicator_not_mem V j hjV, mul_zero, add_zero]


-- ════════════════════════════════════════════════════════════
-- §6. ADDITIVE INJECTION CONFINEMENT
--
-- The production deployment pattern: h + α·(f(h) ⊙ mask).
-- At positions outside the regime, this equals h — the
-- original hidden state is untouched.
-- ════════════════════════════════════════════════════════════

/-- **Theorem (Serial Injection Confined).**
    The additive injection pattern `h + α·(f(h) ⊙ mask)` equals
    `h` at every position outside the regime's partition.
    The original hidden state is provably preserved outside
    the regime. -/
theorem serial_injection_confined {n : ℕ} (f : Adapter n)
    (h : Vec n) (α : ℝ) (S : Finset (Fin n)) (j : Fin n) (hj : j ∉ S) :
    h j + α * (f h ⊙ indicator S) j = h j := by
  have hzero : (f h ⊙ indicator S) j = 0 := serial_output_confined f h S j hj
  rw [hzero, mul_zero, add_zero]

/-- Two adapters injected additively: the compound injection
    preserves h outside both partitions. -/
theorem serial_compound_injection_confined {n : ℕ} (f g : Adapter n)
    (h : Vec n) (α β : ℝ) (S T : Finset (Fin n))
    (j : Fin n) (hjS : j ∉ S) (hjT : j ∉ T) :
    h j + α * (f h ⊙ indicator S) j + β * (g h ⊙ indicator T) j = h j := by
  have hzS : (f h ⊙ indicator S) j = 0 := serial_output_confined f h S j hjS
  have hzT : (g h ⊙ indicator T) j = 0 := serial_output_confined g h T j hjT
  rw [hzS, hzT, mul_zero, mul_zero, add_zero, add_zero]


end Schemen.SerialGate
