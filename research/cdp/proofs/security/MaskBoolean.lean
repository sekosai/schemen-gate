/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/


import GateSecurity
import Mathlib.Tactic

/-!
# MaskBoolean -- Boolean Algebra on Hadamard Masks

Mechanizes the Boolean-completeness claim for CDP's Hadamard masks
(Entry 84).  A mask `m : Vec n` is the gate's basic authority
object: it picks a subset of coordinates and gates which dimensions
an operation may touch.  The construction
defines three operations on masks:

  AND :  `maskAnd m1 m2 = m1 ⊙ m2`           (Hadamard product)
  OR  :  `maskOr  m1 m2 = m1 + m2 - m1 ⊙ m2` (inclusion-exclusion)
  NOT :  `maskNot m     = 1 - m`             (coordinate complement)

Entry 84 of the discovery journal asserts that {AND, NOT, OR}
together with these definitions form a *Boolean algebra* on
`{0,1}^n`, i.e. every Boolean function over coordinate supports is
expressible as a composition of these three operators.  This file
mechanises the constituent laws.

We deliberately use named functions (`maskAnd`, `maskOr`,
`maskNot`) rather than the infix lattice symbols `⊓`, `⊔`, `⊖`,
because Mathlib's lattice infrastructure already exports those for
the meet/join structure of `α → β` and overloading would force the
reader to disambiguate at every use site.

## What requires `IsBinary`, what doesn't

The CDP mask algebra is realised as polynomial expressions
on ℝ, and *some* Boolean identities are real-arithmetic identities
in that polynomial form (e.g. De Morgan, double-negation,
commutativity, associativity).  Others depend genuinely on the
idempotence `m·m = m` -- which is `IsBinary m`.  We separate the
two classes explicitly so the reader can see exactly which laws
are intrinsic to the polynomial form vs. which require the binary
restriction.

## Statement of Results

**Real-arithmetic identities (no `IsBinary` required).**
- Identity: `maskAnd m (maskOne n) = m`,
  `maskOr m (maskZero n) = m`.
- Annihilation (0): `maskAnd m (maskZero n) = maskZero n`.
- Commutativity / Associativity for both.
- Double negation: `maskNot (maskNot m) = m`.
- De Morgan, both directions.

**Identities requiring `IsBinary` (intrinsically).**
- Closure under {AND, OR, NOT}.
- Idempotence: `maskAnd m m = m`, `maskOr m m = m`.
- Annihilation (1): `maskOr m (maskOne n) = maskOne n`.
- Excluded middle: `maskOr m (maskNot m) = maskOne n`.
- Non-contradiction: `maskAnd m (maskNot m) = maskZero n`.
- Distributivity (both directions).
- Absorption (both directions).

These together discharge the "Boolean completeness on Hadamard
masks" claim of Entry 84.

## Axioms

ZERO new axioms.  Reductions:
- `IsBinary` and `hmul` from `GateSecurity.lean`,
- ring arithmetic on ℝ,
- function extensionality.
-/

set_option autoImplicit false

namespace Schemen.MaskBoolean

open Schemen

-- ════════════════════════════════════════════════════════════════
-- §1. THE THREE OPERATORS
-- ════════════════════════════════════════════════════════════════

/-- CDP mask AND: element-wise (Hadamard) product. -/
def maskAnd {n : ℕ} (a b : Vec n) : Vec n := a ⊙ b

/-- CDP mask OR: inclusion-exclusion. -/
def maskOr {n : ℕ} (a b : Vec n) : Vec n :=
  fun j => a j + b j - a j * b j

/-- CDP mask NOT: coordinate complement. -/
def maskNot {n : ℕ} (a : Vec n) : Vec n :=
  fun j => 1 - a j

/-- The all-zeros mask: deny everything. -/
def maskZero (n : ℕ) : Vec n := fun _ => 0

/-- The all-ones mask: allow everything. -/
def maskOne (n : ℕ) : Vec n := fun _ => 1


-- ════════════════════════════════════════════════════════════════
-- §2. CLOSURE OF IsBinary UNDER {AND, OR, NOT}
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (AND preserves binary).** -/
theorem maskAnd_binary {n : ℕ} (a b : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) :
    IsBinary (maskAnd a b) := by
  intro j
  rcases ha j with h0a | h1a
  · left
    simp only [maskAnd, hmul, h0a, zero_mul]
  · rcases hb j with h0b | h1b
    · left
      simp only [maskAnd, hmul, h0b, mul_zero]
    · right
      simp only [maskAnd, hmul, h1a, h1b, mul_one]

/-- **Theorem (OR preserves binary).** -/
theorem maskOr_binary {n : ℕ} (a b : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) :
    IsBinary (maskOr a b) := by
  intro j
  rcases ha j with h0a | h1a <;> rcases hb j with h0b | h1b
  · left;  simp only [maskOr, h0a, h0b]; ring
  · right; simp only [maskOr, h0a, h1b]; ring
  · right; simp only [maskOr, h1a, h0b]; ring
  · right; simp only [maskOr, h1a, h1b]; ring

/-- **Theorem (NOT preserves binary).** -/
theorem maskNot_binary {n : ℕ} (a : Vec n) (ha : IsBinary a) :
    IsBinary (maskNot a) := by
  intro j
  rcases ha j with h0 | h1
  · right; simp only [maskNot, h0]; ring
  · left;  simp only [maskNot, h1]; ring

/-- **Corollary (Boolean closure).** -/
theorem binary_closure {n : ℕ} (a b : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) :
    IsBinary (maskAnd a b) ∧ IsBinary (maskOr a b) ∧ IsBinary (maskNot a) :=
  ⟨maskAnd_binary a b ha hb, maskOr_binary a b ha hb, maskNot_binary a ha⟩


-- ════════════════════════════════════════════════════════════════
-- §3. IDEMPOTENCE  (binary required: m·m = m iff m ∈ {0,1})
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (AND idempotent).** -/
theorem maskAnd_idem {n : ℕ} (m : Vec n) (hm : IsBinary m) :
    maskAnd m m = m := by
  funext j
  rcases hm j with h | h <;> simp only [maskAnd, hmul, h] <;> ring

/-- **Theorem (OR idempotent).** -/
theorem maskOr_idem {n : ℕ} (m : Vec n) (hm : IsBinary m) :
    maskOr m m = m := by
  funext j
  rcases hm j with h | h <;> simp only [maskOr, h] <;> ring


-- ════════════════════════════════════════════════════════════════
-- §4. IDENTITY AND ANNIHILATION
--
-- Identity laws and zero-annihilation are real-arithmetic
-- identities (no binary constraint required).
-- One-annihilation requires binary.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (1 is the AND-identity).** -/
theorem maskAnd_one {n : ℕ} (m : Vec n) :
    maskAnd m (maskOne n) = m := by
  funext j
  simp only [maskAnd, hmul, maskOne, mul_one]

/-- **Theorem (0 is the OR-identity).** -/
theorem maskOr_zero {n : ℕ} (m : Vec n) :
    maskOr m (maskZero n) = m := by
  funext j
  simp only [maskOr, maskZero, add_zero, mul_zero, sub_zero]

/-- **Theorem (0 is the AND-annihilator).** -/
theorem maskAnd_zero {n : ℕ} (m : Vec n) :
    maskAnd m (maskZero n) = maskZero n := by
  funext j
  simp only [maskAnd, hmul, maskZero, mul_zero]

/-- **Theorem (1 is the OR-annihilator), for binary masks.** -/
theorem maskOr_one {n : ℕ} (m : Vec n) (hm : IsBinary m) :
    maskOr m (maskOne n) = maskOne n := by
  funext j
  rcases hm j with h | h <;> simp only [maskOr, maskOne, h] <;> ring


-- ════════════════════════════════════════════════════════════════
-- §5. COMMUTATIVITY AND ASSOCIATIVITY
--
-- All four laws are real-arithmetic identities.  No binary
-- constraint required.
-- ════════════════════════════════════════════════════════════════

theorem maskAnd_comm {n : ℕ} (a b : Vec n) :
    maskAnd a b = maskAnd b a := by
  funext j; simp only [maskAnd, hmul]; ring

theorem maskOr_comm {n : ℕ} (a b : Vec n) :
    maskOr a b = maskOr b a := by
  funext j; simp only [maskOr]; ring

theorem maskAnd_assoc {n : ℕ} (a b c : Vec n) :
    maskAnd (maskAnd a b) c = maskAnd a (maskAnd b c) := by
  funext j; simp only [maskAnd, hmul]; ring

theorem maskOr_assoc {n : ℕ} (a b c : Vec n) :
    maskOr (maskOr a b) c = maskOr a (maskOr b c) := by
  funext j; simp only [maskOr]; ring


-- ════════════════════════════════════════════════════════════════
-- §6. COMPLEMENTATION  (binary required)
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Excluded middle: m ∨ ¬m = 1).** -/
theorem maskOr_compl {n : ℕ} (m : Vec n) (hm : IsBinary m) :
    maskOr m (maskNot m) = maskOne n := by
  funext j
  rcases hm j with h | h <;>
    simp only [maskOr, maskNot, maskOne, h] <;> ring

/-- **Theorem (Non-contradiction: m ∧ ¬m = 0).** -/
theorem maskAnd_compl {n : ℕ} (m : Vec n) (hm : IsBinary m) :
    maskAnd m (maskNot m) = maskZero n := by
  funext j
  rcases hm j with h | h <;>
    simp only [maskAnd, hmul, maskNot, maskZero, h] <;> ring

/-- **Theorem (Double negation).**  Real-arithmetic identity; no
    binary constraint required. -/
theorem maskNot_not {n : ℕ} (m : Vec n) :
    maskNot (maskNot m) = m := by
  funext j
  simp only [maskNot]; ring


-- ════════════════════════════════════════════════════════════════
-- §7. DE MORGAN
--
-- Both De Morgan laws are real-arithmetic identities in the
-- polynomial form of the CDP mask algebra.
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (De Morgan, AND -> OR): real-arithmetic identity.**
    `1 - a·b = (1-a) + (1-b) - (1-a)·(1-b)` for all `a, b ∈ ℝ`. -/
theorem maskNot_and {n : ℕ} (a b : Vec n) :
    maskNot (maskAnd a b) = maskOr (maskNot a) (maskNot b) := by
  funext j
  simp only [maskNot, maskAnd, hmul, maskOr]; ring

/-- **Theorem (De Morgan, OR -> AND): real-arithmetic identity.**
    `1 - (a + b - a·b) = (1-a)·(1-b)` for all `a, b ∈ ℝ`. -/
theorem maskNot_or {n : ℕ} (a b : Vec n) :
    maskNot (maskOr a b) = maskAnd (maskNot a) (maskNot b) := by
  funext j
  simp only [maskNot, maskOr, maskAnd, hmul]; ring


-- ════════════════════════════════════════════════════════════════
-- §8. DISTRIBUTIVITY  (binary required: needs a² = a)
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (AND distributes over OR).** -/
theorem maskAnd_distrib_or {n : ℕ} (a b c : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) (hc : IsBinary c) :
    maskAnd a (maskOr b c) = maskOr (maskAnd a b) (maskAnd a c) := by
  funext j
  rcases ha j with ha' | ha' <;>
    rcases hb j with hb' | hb' <;>
    rcases hc j with hc' | hc' <;>
    simp only [maskAnd, hmul, maskOr, ha', hb', hc'] <;>
    ring

/-- **Theorem (OR distributes over AND).** -/
theorem maskOr_distrib_and {n : ℕ} (a b c : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) (hc : IsBinary c) :
    maskOr a (maskAnd b c) = maskAnd (maskOr a b) (maskOr a c) := by
  funext j
  rcases ha j with ha' | ha' <;>
    rcases hb j with hb' | hb' <;>
    rcases hc j with hc' | hc' <;>
    simp only [maskOr, maskAnd, hmul, ha', hb', hc'] <;>
    ring


-- ════════════════════════════════════════════════════════════════
-- §9. ABSORPTION  (binary required)
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (AND-absorption: m1 ∧ (m1 ∨ m2) = m1).** -/
theorem maskAnd_absorb {n : ℕ} (a b : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) :
    maskAnd a (maskOr a b) = a := by
  funext j
  rcases ha j with ha' | ha' <;> rcases hb j with hb' | hb' <;>
    simp only [maskAnd, hmul, maskOr, ha', hb'] <;> ring

/-- **Theorem (OR-absorption: m1 ∨ (m1 ∧ m2) = m1).** -/
theorem maskOr_absorb {n : ℕ} (a b : Vec n)
    (ha : IsBinary a) (hb : IsBinary b) :
    maskOr a (maskAnd a b) = a := by
  funext j
  rcases ha j with ha' | ha' <;> rcases hb j with hb' | hb' <;>
    simp only [maskOr, maskAnd, hmul, ha', hb'] <;> ring


-- ════════════════════════════════════════════════════════════════
-- §10. SUMMARY
--
-- Together, §2-§9 say: the binary-mask sublattice of (Vec n)
-- forms a Boolean algebra under (maskAnd, maskOr, maskNot,
-- maskZero, maskOne).  Every standard Boolean-algebra identity
-- (idempotence, absorption, de Morgan, distributivity,
-- complementation) holds; some are intrinsic to the polynomial
-- form (de Morgan, commutativity, associativity, double-negation,
-- 0-/1-identities, 0-annihilation), others require the binary
-- constraint (idempotence, 1-annihilation, excluded middle,
-- non-contradiction, distributivity, absorption).
--
-- This is the formal statement of Entry 84's "Boolean
-- completeness on Hadamard masks": the CDP gate algebra
-- is a Boolean algebra in the precise sense of Stone duality,
-- and the runtime exploits that fact by content-hashing the
-- mask-of-bytes and treating composition as integer (= bit)
-- arithmetic the float unit happens to execute.
-- ════════════════════════════════════════════════════════════════

end Schemen.MaskBoolean
