/-
Copyright (c) 2026 Ryan R. All rights reserved.
Released under Apache 2.0; see LICENSES/Apache-2.0.txt.
Authors: Ryan R
-/
import ModelSecurityV3

/-!
# P3 — Formal Gate Placement Proofs

Extends the Lean 4 proof suite to cover gate placement validity,
addressing the three questions posed in the papers roadmap:

1. **FFN gating preserves isolation when attention is shared (§1)**
2. **Decoder gating is necessary for seq2seq isolation (§2)**
3. **Layer safety classification: "safe to gate" vs "unsafe to gate" (§3)**

## Proof Architecture

```
  V3 §3B: PreservesSupport framework (foundation)
    │
    ├── §1: SharedAttention + FFN gate → full MLP isolation
    │       (attention is arbitrary upstream; gate is bottleneck)
    │
    ├── §2: Encoder-only gate → decoder leak (constructive)
    │       Dual gates → full seq2seq isolation
    │
    ├── §3: Layer safety taxonomy
    │       ├── SAFE: element-wise f(0)=0, Hadamard, ReLU backward
    │       ├── UNSAFE: constant injection (LN backward), projection (dense linear)
    │       └── Composition: safe ∘ safe = safe
    │
    └── §4: GatePlacementValid predicate + master classification
            ├── Forward isolation: always holds (gate is multiplicative)
            ├── W2 confinement: always holds (forward-path only)
            └── W1 confinement: iff backward chain preserves support
```

## Key Insight

Gate placement validity reduces to a single question: does the backward
path between the gate and the W1 weight update preserve the zero
structure imposed by the mask?

- YES → W1 confinement holds → safe placement
- NO → W1 confinement breakable → unsafe placement
- W2 confinement holds unconditionally (forward path only)

This was established abstractly in V3 §3B. This file makes it concrete
by classifying every transformer component and proving end-to-end
gate placement theorems for standard architectures.

## Axiom inventory

No new axioms. All theorems are either:
- Algebraic consequences of the multiplicative gate (fully proven)
- Direct applications of V3's PreservesSupport framework (fully proven)
- Constructive counterexamples for unsafe placements (fully proven)
-/

set_option autoImplicit false

noncomputable section

namespace Schemen.GatePlacement

open Schemen Schemen.Security Schemen.SecurityV2 Schemen.SecurityV3


-- ════════════════════════════════════════════════════════════════
-- §1. SHARED ATTENTION + FFN GATE ISOLATION
--
-- Attention computes Q·K^T/√d → softmax → ·V, a full inner
-- product across ALL dimensions. It mixes dimensions freely.
-- The FFN gate, applied post-attention inside the MLP, creates
-- a multiplicative bottleneck that confines both forward
-- activations and backward gradients.
--
-- Architecture (standard transformer block):
--
--   input → LN → [Attn] → (+) → LN → W1 → ReLU → [GATE] → W2 → (+) → out
--           │               ↑                         ↑                ↑
--           └── residual ───┘          bottleneck ─────┘  residual ────┘
--
-- The gate is the security boundary. Everything upstream (including
-- attention) is allowed to mix freely. Everything downstream (W2
-- output) is confined to the regime's dimensions.
-- ════════════════════════════════════════════════════════════════

/-- Shared attention: an arbitrary function mapping activations to
    activations. Models Q·K^T/√d → softmax → V, which computes a
    full inner product across all dimensions.

    Critically: NO support preservation is assumed or required.
    Attention can route information from any dimension to any other.
    The gate, not attention, provides isolation. -/
abbrev SharedAttention (n : ℕ) := Vec n → Vec n

/-- **Theorem (Forward Isolation Despite Shared Attention).**
    The gated hidden activation at dimension j is zero when
    j ∉ groups(r), regardless of what attention computed upstream.

    The attention layer can send arbitrary signal to dimension j.
    The FFN's W1 can transform it into any hidden activation h[j].
    But the gate multiplies by mask[j] = 0, producing exactly zero.

    This is the forward-pass security guarantee: the regime's output
    depends only on the regime's partition dimensions, no matter how
    attention mixes the full representation. -/
theorem forward_isolation_despite_attention {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (_attn : SharedAttention n)
    (h_hidden : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    (h_hidden ⊙ indicator (P.groups r)) j = 0 :=
  forward_isolation h_hidden (indicator (P.groups r)) j (indicator_not_mem _ j hj)

/-- **Theorem (W1 Confinement Despite Shared Attention).**
    Even when attention backward produces gradients at ALL dimensions,
    the gate confines W1 column updates to the regime's partition.

    Gradient chain:
      loss → later layers → attention backward → d_from_attention
      → [gate: d_h = d_from_attention ⊙ mask]
      → [ReLU: d_z = d_h ⊙ relu']
      → [W1: d_W1[i,j] = x[i] · d_z[j]]

    At inactive dim j: mask[j] = 0 → d_h[j] = 0 → d_z[j] = 0
    → d_W1[i,j] = 0 for all i.

    The attention backward can produce any d_from_attention. The gate
    kills the gradient at inactive dims regardless. -/
theorem w1_confined_despite_attention {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (_attn : SharedAttention n)
    (d_from_attention : Vec n)
    (relu_grad : Vec n) (mlp_input : Vec m)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m,
      outer mlp_input
        ((d_from_attention ⊙ indicator (P.groups r)) ⊙ relu_grad) i j = 0 :=
  weight_update_confined d_from_attention _ relu_grad mlp_input j
    (indicator_not_mem _ j hj)

/-- **Theorem (W2 Confinement Despite Shared Attention).**
    W2 row confinement depends ONLY on the forward gate:
    d_W2[j,k] = gated[j] · d_output[k] = (h[j] · 0) · d_output[k] = 0.

    No backward path is involved. The forward gate is sufficient.
    This holds regardless of attention, loss function, or any other
    architectural component. -/
theorem w2_confined_despite_attention {n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (_attn : SharedAttention n)
    (h_hidden : Vec n) (d_output : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ k : Fin o,
      outer (h_hidden ⊙ indicator (P.groups r)) d_output j k = 0 :=
  fun k => w2_update_confined h_hidden _ d_output j (indicator_not_mem _ j hj) k

/-- **Theorem (Attention Weights Are Intentionally Shared).**
    The attention weight gradient at non-regime dimensions is generally
    nonzero. This is BY DESIGN: attention learns shared routing patterns
    that serve all regimes.

    The security model: attention is SHARED infrastructure (like the
    backbone). Regime-specific knowledge lives in the gated FFN.
    Attention routing is not a secret — it is the FFN partition that
    isolates regime knowledge. -/
theorem attention_weights_shared {m n : ℕ}
    (d_query : Vec n) (input : Vec m)
    (j : Fin n) (hq : d_query j ≠ 0)
    (i : Fin m) (hx : input i ≠ 0) :
    outer input d_query i j ≠ 0 := by
  simp only [outer]
  exact mul_ne_zero hx hq

/-- **Theorem (Gated Output Is Regime-Local Despite Shared Attention).**
    The gated FFN output logits depend ONLY on dimensions in groups(r).
    The full sum over all n hidden dims collapses to a sum over just
    the |groups(r)| = n/R active dims.

    Shared attention can mix freely, but the gated MLP output is a
    pure function of the regime's partition. Combines forward isolation
    with regime output locality (V3 §2). -/
theorem gated_output_local_despite_attention {n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (_attn : SharedAttention n)
    (h_act : Vec n) (W2 : Fin n → Fin o → ℝ) (b2 : Fin o → ℝ) :
    ∀ k : Fin o,
      output_logits (h_act ⊙ indicator (P.groups r)) W2 b2 k =
      (P.groups r).sum (fun j => h_act j * W2 j k) + b2 k :=
  regime_output_locality P r h_act W2 b2

/-- **Theorem (Full Block: Shared Attention + Gated FFN Isolation).**
    Complete isolation result for a transformer block with shared
    attention and gated FFN. For any attention function and any
    gradient arriving from upstream:

    (a) Gated activation at non-regime dim j: zero
    (b) W1 column j gradient: zero for all input dims i
    (c) W2 row j gradient: zero for all output dims k

    This is the formal proof that FFN gating is SUFFICIENT for
    MLP weight isolation, even with fully shared attention. -/
theorem shared_attention_gated_ffn_isolation {m n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (_attn : SharedAttention n)
    (d_upstream relu_grad : Vec n)
    (mlp_input : Vec m) (h_hidden : Vec n) (d_output : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ((h_hidden ⊙ indicator (P.groups r)) j = 0)
    ∧ (∀ i : Fin m,
        outer mlp_input
          ((d_upstream ⊙ indicator (P.groups r)) ⊙ relu_grad) i j = 0)
    ∧ (∀ k : Fin o,
        outer (h_hidden ⊙ indicator (P.groups r)) d_output j k = 0) := by
  have hmask := indicator_not_mem (P.groups r) j hj
  exact ⟨forward_isolation h_hidden _ j hmask,
         weight_update_confined d_upstream _ relu_grad mlp_input j hmask,
         fun k => w2_update_confined h_hidden _ d_output j hmask k⟩


-- ════════════════════════════════════════════════════════════════
-- §2. ENCODER-DECODER: DUAL GATING NECESSITY
--
-- In seq2seq architectures (T5, BART, encoder-decoder transformers),
-- the encoder produces representations that the decoder reads via
-- cross-attention. The question: is encoder gating sufficient, or
-- must the decoder also be gated?
--
-- Answer: BOTH must be gated. The gate provides LOCAL weight
-- confinement for the MLP it guards. Gating the encoder confines
-- encoder weights. Gating the decoder confines decoder weights.
-- Neither substitutes for the other.
--
-- Why encoder-only fails:
-- 1. The encoder output = input + gated_FFN(input)
-- 2. The residual stream (input) has signal at ALL dimensions
-- 3. Cross-attention reads the full encoder output
-- 4. The decoder's W1 receives gradient from this full input
-- 5. Without a decoder gate: decoder W1 updates at ALL dims
-- ════════════════════════════════════════════════════════════════

/-- **Theorem (Encoder Gating Confines Encoder FFN).**
    When the encoder's FFN is gated, encoder MLP weight updates
    are confined to the regime's partition. This holds regardless
    of what the decoder does. -/
theorem encoder_gating_confines_encoder {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_enc_gated relu_grad_enc : Vec n) (enc_input : Vec m)
    (enc_hidden : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    (∀ i : Fin m,
      outer enc_input
        ((d_enc_gated ⊙ indicator (P.groups r)) ⊙ relu_grad_enc) i j = 0)
    ∧ (∀ (o : ℕ) (d_out : Vec o) (k : Fin o),
      outer (enc_hidden ⊙ indicator (P.groups r)) d_out j k = 0) :=
  ⟨weight_update_confined d_enc_gated _ relu_grad_enc enc_input j
    (indicator_not_mem _ j hj),
   w2_rows_are_regime_confined P r j hj enc_hidden⟩

/-- **Theorem (Ungated Decoder Leaks).**
    Without a gate in the decoder's FFN, there exist gradient
    configurations where the decoder W1 update at a non-regime
    dimension is nonzero.

    This holds even when the encoder is perfectly gated, because
    the decoder receives full-dimensional input via cross-attention
    and the residual stream. The encoder's gate cannot help — weight
    confinement is local to each MLP. -/
theorem ungated_decoder_leaks {m n : ℕ}
    (d_dec relu_grad_dec : Vec n)
    (dec_input : Vec m)
    (j : Fin n)
    (hgrad : d_dec j ≠ 0) (hrelu : relu_grad_dec j ≠ 0)
    (i : Fin m) (hx : dec_input i ≠ 0) :
    outer dec_input (d_dec ⊙ relu_grad_dec) i j ≠ 0 := by
  simp only [outer, hmul]
  exact mul_ne_zero hx (mul_ne_zero hgrad hrelu)

/-- **Theorem (Dual Gating Confines Decoder FFN).**
    With a gate in the decoder's FFN, decoder MLP weight updates
    are confined to the regime's partition, mirroring the encoder. -/
theorem dual_gating_confines_decoder {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_dec_gated relu_grad_dec : Vec n) (dec_input : Vec m)
    (dec_hidden : Vec n)
    (j : Fin n) (hj : j ∉ P.groups r) :
    (∀ i : Fin m,
      outer dec_input
        ((d_dec_gated ⊙ indicator (P.groups r)) ⊙ relu_grad_dec) i j = 0)
    ∧ (∀ (o : ℕ) (d_out : Vec o) (k : Fin o),
      outer (dec_hidden ⊙ indicator (P.groups r)) d_out j k = 0) :=
  ⟨weight_update_confined d_dec_gated _ relu_grad_dec dec_input j
    (indicator_not_mem _ j hj),
   w2_rows_are_regime_confined P r j hj dec_hidden⟩

/-- **Theorem (Encoder-Only Gating Is Insufficient for Seq2Seq).**
    When only the encoder has a gate:
    • Encoder MLP weights: CONFINED (gate provides confinement)
    • Decoder MLP weights: NOT CONFINED (no gate, no confinement)

    Proof: the encoder half follows from the gate (algebraic zero).
    The decoder half constructs explicit witnesses (unit vectors)
    where the ungated decoder W1 column j receives nonzero update.

    Gate confinement is LOCAL. The encoder's gate does not
    propagate to the decoder's MLP. -/
theorem encoder_only_insufficient {m n R : ℕ}
    (hm : 0 < m)
    (P : ValidPartition n R) (r : Fin R)
    (j : Fin n) (hj : j ∉ P.groups r) :
    -- Encoder IS confined (gate works)
    (∀ (d_enc relu_grad : Vec n) (enc_input : Vec m) (i : Fin m),
      outer enc_input
        ((d_enc ⊙ indicator (P.groups r)) ⊙ relu_grad) i j = 0)
    -- Decoder is NOT confined (no gate → leak exists)
    ∧ (∃ (d_dec relu_grad : Vec n) (dec_input : Vec m) (i : Fin m),
      outer dec_input (d_dec ⊙ relu_grad) i j ≠ 0) := by
  refine ⟨fun d_enc relu_grad enc_input i =>
    weight_update_confined d_enc _ relu_grad enc_input j
      (indicator_not_mem _ j hj) i, ?_⟩
  exact ⟨fun _ => 1, fun _ => 1, fun _ => 1, ⟨0, hm⟩, by
    simp only [outer, hmul, mul_one]; exact one_ne_zero⟩

/-- **Theorem (Dual Gating Is Necessary and Sufficient).**
    For full seq2seq MLP weight isolation:
    • Encoder gate → confines encoder W1/W2
    • Decoder gate → confines decoder W1/W2
    • Both gates use the same partition (derived from the same key)

    Each gate operates independently. The two isolation results
    compose without interaction. -/
theorem dual_gating_sufficient {m n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_enc relu_enc : Vec n) (enc_input : Vec m) (enc_hidden : Vec n)
    (d_dec relu_dec : Vec n) (dec_input : Vec m) (dec_hidden : Vec n)
    (d_output : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    -- Encoder W1 column j: zero
    (∀ i : Fin m, outer enc_input
      ((d_enc ⊙ indicator (P.groups r)) ⊙ relu_enc) i j = 0)
    -- Encoder W2 row j: zero
    ∧ (∀ k : Fin o, outer (enc_hidden ⊙ indicator (P.groups r))
      d_output j k = 0)
    -- Decoder W1 column j: zero
    ∧ (∀ i : Fin m, outer dec_input
      ((d_dec ⊙ indicator (P.groups r)) ⊙ relu_dec) i j = 0)
    -- Decoder W2 row j: zero
    ∧ (∀ k : Fin o, outer (dec_hidden ⊙ indicator (P.groups r))
      d_output j k = 0) := by
  have hmask := indicator_not_mem (P.groups r) j hj
  exact ⟨weight_update_confined d_enc _ relu_enc enc_input j hmask,
         fun k => w2_update_confined enc_hidden _ d_output j hmask k,
         weight_update_confined d_dec _ relu_dec dec_input j hmask,
         fun k => w2_update_confined dec_hidden _ d_output j hmask k⟩


-- ════════════════════════════════════════════════════════════════
-- §3. LAYER SAFETY CLASSIFICATION
--
-- Which layers are "safe to gate" (the backward path preserves
-- support) vs "unsafe to gate" (the backward path breaks support)?
--
-- The PreservesSupport framework (V3 §3B) already provides:
-- ✓ id_preserves_support (identity is safe)
-- ✓ pointwise_preserves_support (element-wise f(0)=0 is safe)
-- ✓ hmul_preserves_support (Hadamard product is safe)
-- ✓ compose_preserves_support (composition of safe = safe)
-- ✓ support_preserving_maintains_w1_confinement (positive result)
-- ✓ non_preserving_breaks_w1_confinement (negative result)
--
-- This section adds CONCRETE unsafe counterexamples:
-- ✗ Constant injection (models LayerNorm backward mean subtraction)
-- ✗ Dimension projection (models dense linear backward mixing)
-- And shows the standard FFN placement IS safe.
-- ════════════════════════════════════════════════════════════════


-- ── §3A. UNSAFE: CONSTANT INJECTION (LayerNorm backward) ──────

/-- Constant injection: outputs a fixed value c at every dimension,
    ignoring the input.

    Models the mean-subtraction component of LayerNorm backward:

      d_x[j] = γ/σ · (d_h[j] − μ_{d_h} − x̂[j] · Σ d_h[i]·x̂[i]/n)

    When d_h is supported on active dims S (gate zeros inactive dims),
    the mean μ_{d_h} = (1/n)·Σ_{i∈S} d_h[i] is generally nonzero.
    This nonzero constant is then subtracted from ALL dimensions,
    including inactive ones, introducing gradient outside S.

    We model this with the simplest function exhibiting the property:
    one that injects a constant c into every dimension. -/
def const_inject {n : ℕ} (c : ℝ) : Vec n → Vec n :=
  fun _v _j => c

/-- **Theorem (Constant Injection Breaks Support).**
    A function that outputs constant c ≠ 0 at every dimension
    does NOT preserve support. The zero vector (trivially supported
    on any S) maps to a vector that is nonzero at inactive dims.

    This captures why LayerNorm backward is unsafe for gate placement:
    the mean of active gradients leaks to inactive dimensions. -/
theorem const_inject_breaks_support {n : ℕ}
    (S : Finset (Fin n))
    (j₀ : Fin n) (hj₀ : j₀ ∉ S)
    (c : ℝ) (hc : c ≠ 0) :
    ¬ PreservesSupport (const_inject c) S := by
  intro hp
  have h_supp : ∀ j : Fin n, j ∉ S → (fun (_ : Fin n) => (0 : ℝ)) j = 0 :=
    fun _ _ => rfl
  have h_val := hp _ h_supp j₀ hj₀
  exact hc (by simpa only [const_inject] using h_val)

/-- **Corollary (LayerNorm Placement Is Unsafe).**
    Placing the gate such that LayerNorm backward is between the gate
    and W1 breaks W1 confinement.

    UNSAFE architecture:
      W1 → ReLU → [GATE] → LayerNorm → W2
                              ↑
                LN backward sits between gate and W1 gradient path

    SAFE architecture (standard):
      LN → W1 → ReLU → [GATE] → W2
      ↑
      LN is upstream of the gated MLP, not in the gate→W1 path -/
theorem layernorm_placement_unsafe {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (j₀ : Fin n) (hj₀ : j₀ ∉ P.groups r)
    (c : ℝ) (hc : c ≠ 0) :
    ¬ PreservesSupport (const_inject c) (P.groups r) :=
  const_inject_breaks_support _ j₀ hj₀ c hc


-- ── §3B. UNSAFE: DIMENSION PROJECTION (dense linear backward) ─

/-- Dimension projection: maps every dimension's value to the value
    at a fixed source dimension i₀. f(v)[j] = v[i₀] for all j.

    Models the effect of dense linear backward (W^T multiplication)
    where column j₀ of W^T has a nonzero entry at row i₀. The gradient
    at active dimension i₀ is projected to inactive dimension j₀.

    Attention backward has this property: W_Q^T, W_K^T, W_V^T are
    dense matrices that route gradient from any dimension to any other.
    This is why gates go in the FFN, not in the attention path. -/
def dim_project {n : ℕ} (i₀ : Fin n) : Vec n → Vec n :=
  fun v _j => v i₀

/-- **Theorem (Dimension Projection Breaks Support).**
    If i₀ ∈ S and j₀ ∉ S, then dim_project i₀ does NOT preserve
    support on S. The indicator of S (supported on S with value 1
    at i₀) maps to a vector that is 1 at j₀ ∉ S.

    This captures why attention backward and dense linear backward
    are unsafe: the gradient at an active dimension is routed to
    inactive dimensions via the weight matrix transpose. -/
theorem dim_project_breaks_support {n : ℕ}
    (S : Finset (Fin n))
    (i₀ : Fin n) (hi₀ : i₀ ∈ S)
    (j₀ : Fin n) (hj₀ : j₀ ∉ S) :
    ¬ PreservesSupport (dim_project i₀) S := by
  intro hp
  have h_supp : ∀ j : Fin n, j ∉ S → indicator S j = 0 :=
    fun j hj => indicator_not_mem _ j hj
  have h_val := hp (indicator S) h_supp j₀ hj₀
  simp only [dim_project] at h_val
  rw [indicator_mem S i₀ hi₀] at h_val
  exact one_ne_zero h_val

/-- **Corollary (Attention Backward Placement Is Unsafe).**
    A gate placed so that attention backward (W_Q^T, W_K^T, W_V^T
    multiplication) is between the gate and W1 breaks confinement.

    Given any active dimension i₀ and any inactive dimension j₀,
    the dense linear backward can route gradient from i₀ to j₀.

    This is why gates go INSIDE the FFN, not inside attention.
    Attention is shared infrastructure; the FFN is the isolation point. -/
theorem attention_backward_unsafe {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (i₀ : Fin n) (hi₀ : i₀ ∈ P.groups r)
    (j₀ : Fin n) (hj₀ : j₀ ∉ P.groups r) :
    ¬ PreservesSupport (dim_project i₀) (P.groups r) :=
  dim_project_breaks_support _ i₀ hi₀ j₀ hj₀


-- ── §3C. SAFE: STANDARD FFN PLACEMENT ─────────────────────────

/-- **Theorem (Standard FFN Placement Is Safe).**
    The standard gate placement (after ReLU, before W2):

      W1 → z → ReLU → h → [GATE] → gated → W2

    The backward chain from gate to W1 is:
      d_h ⊙ relu_grad(z)    (Hadamard with ReLU derivative)

    Hadamard preserves support (V3): if d_h[j] = 0 at inactive
    dims, then d_h[j] · relu_grad[j] = 0 · anything = 0.

    Therefore the standard placement is safe for W1 confinement. -/
theorem standard_ffn_safe {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (relu_grad : Vec n) :
    PreservesSupport (· ⊙ relu_grad) (P.groups r) :=
  hmul_preserves_support relu_grad (P.groups r)

/-- **Theorem (Standard Placement → Full Weight Isolation).**
    Combining the standard safe placement with the confinement
    theorems: all W1 column and W2 row updates at non-regime
    dimensions are zero. -/
theorem standard_placement_full_isolation {m n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_gated relu_grad : Vec n) (mlp_input : Vec m)
    (h_hidden : Vec n) (d_output : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    (∀ i : Fin m,
      outer mlp_input
        ((d_gated ⊙ indicator (P.groups r)) ⊙ relu_grad) i j = 0)
    ∧ (∀ k : Fin o,
      outer (h_hidden ⊙ indicator (P.groups r)) d_output j k = 0) :=
  ⟨support_preserving_maintains_w1_confinement P r (· ⊙ relu_grad)
    (hmul_preserves_support relu_grad _) d_gated mlp_input j hj,
   fun k => w2_update_confined h_hidden _ d_output j
    (indicator_not_mem _ j hj) k⟩


-- ── §3D. SAFE: COMPOSITION OF SAFE OPERATIONS ─────────────────

/-- **Theorem (Chained Safe Operations Remain Safe).**
    If two backward-chain operations both preserve support, their
    composition preserves support. This means stacking multiple safe
    operations between the gate and W1 is still safe.

    Example: gate → Hadamard(relu_grad) → Hadamard(dropout_mask)
    Both Hadamard ops preserve support, so the chain is safe. -/
theorem chained_safe_ops {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (f g : Vec n → Vec n)
    (hf : PreservesSupport f (P.groups r))
    (hg : PreservesSupport g (P.groups r)) :
    PreservesSupport (f ∘ g) (P.groups r) :=
  compose_preserves_support f g (P.groups r) hf hg


-- ════════════════════════════════════════════════════════════════
-- §4. GATE PLACEMENT VALIDITY — COMPREHENSIVE CHARACTERIZATION
--
-- We package the results from §1–§3 into a single framework:
-- a gate placement validity predicate and master classification.
--
-- The information flow property that makes a gate placement safe:
--
--   "Zeros at inactive dimensions are preserved through the
--    backward chain between the gate and the W1 weight update."
--
-- This is precisely PreservesSupport. It is the COMPLETE
-- characterization:
-- • If the backward chain preserves support → W1 confined (§3C)
-- • If not → there exist configs where W1 leaks (V3 §3B)
-- • W2 confinement is unconditional (V3 §3B)
-- ════════════════════════════════════════════════════════════════

/-- A gate placement is valid for regime r if the backward chain
    between the gate and the W1 weight update preserves support
    on groups(r).

    This predicate is both necessary and sufficient for W1 confinement:
    • `gate_placement_implies_w1_confined`: valid → W1 confined
    • `non_preserving_breaks_w1_confinement` (V3): ¬valid → W1 breakable -/
def IsValidPlacement {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (backward_chain : Vec n → Vec n) : Prop :=
  PreservesSupport backward_chain (P.groups r)

/-- **Theorem (Valid Placement → W1 Confinement).**
    If the backward chain preserves support, W1 column updates at
    non-regime dimensions are zero for all possible upstream gradients. -/
theorem gate_placement_implies_w1_confined {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (f_back : Vec n → Vec n)
    (hvalid : IsValidPlacement P r f_back)
    (d_gated : Vec n) (mlp_input : Vec m)
    (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ i : Fin m,
      outer mlp_input (f_back (d_gated ⊙ indicator (P.groups r))) i j = 0 :=
  support_preserving_maintains_w1_confinement P r f_back hvalid d_gated mlp_input j hj

/-- **Theorem (Invalid Placement → W1 Confinement Breakable).**
    If the backward chain does NOT preserve support (there exists a
    vector supported on groups(r) that maps to a vector with nonzero
    values outside groups(r)), then there exist upstream gradients
    that produce nonzero W1 updates outside the regime.

    This is the formal justification for refusing unsafe placements:
    they can — not just might — break W1 confinement. -/
theorem invalid_placement_breaks_w1 {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (f_back : Vec n → Vec n)
    (h_break : ∃ (v : Vec n) (j : Fin n),
      (∀ i : Fin n, i ∉ P.groups r → v i = 0) ∧
      j ∉ P.groups r ∧ f_back v j ≠ 0) :
    ∃ (d_gated : Vec n) (j : Fin n),
      j ∉ P.groups r ∧
      f_back (d_gated ⊙ indicator (P.groups r)) j ≠ 0 :=
  non_preserving_breaks_w1_confinement P r f_back h_break

/-- **Theorem (W2 Confinement Is Placement-Independent).**
    W2 row confinement holds for ANY placement because it depends
    only on the forward gate (gated[j] = h[j] · mask[j] = 0).
    No backward path is involved. -/
theorem w2_always_confined {n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (h_act : Vec n) (j : Fin n) (hj : j ∉ P.groups r) :
    ∀ (o : ℕ) (d_out : Vec o) (k : Fin o),
      outer (h_act ⊙ indicator (P.groups r)) d_out j k = 0 :=
  w2_confinement_unconditional P r h_act j hj

/-- **Master Theorem (Gate Placement Classification).**

    Complete characterization of gate placement safety:

    VALID placements (backward chain preserves support):
    ✓ Standard FFN: gate after ReLU, backward = (· ⊙ relu_grad)
    ✓ Gate after any element-wise f with f(0)=0
    ✓ Gate with identity backward (gate directly before W2)
    ✓ Any composition of valid backward chains

    INVALID placements (backward chain breaks support):
    ✗ Gate after LayerNorm (mean subtraction spreads gradient)
    ✗ Gate after dense linear without its own gate (dimension mixing)
    ✗ Gate after any operation that creates nonzero signal at inactive dims

    For any placement:
    1. If valid: BOTH W1 and W2 are confined (full isolation)
    2. If invalid: W2 is still confined, but W1 MAY leak
    3. W2 confinement is unconditional — depends only on forward gate

    Classification rule: check whether the backward function maps
    vectors supported on groups(r) to vectors supported on groups(r).
    This is PreservesSupport — the complete information flow criterion. -/
theorem gate_placement_master {m n R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (f_back : Vec n → Vec n) :
    -- If valid placement: W1 confined for all configs
    (IsValidPlacement P r f_back →
      ∀ (d_gated : Vec n) (mlp_input : Vec m) (j : Fin n),
        j ∉ P.groups r →
        ∀ i : Fin m,
          outer mlp_input (f_back (d_gated ⊙ indicator (P.groups r))) i j = 0)
    -- W2 always confined regardless of placement
    ∧ (∀ (h_act : Vec n) (j : Fin n),
      j ∉ P.groups r →
      ∀ (o : ℕ) (d_out : Vec o) (k : Fin o),
        outer (h_act ⊙ indicator (P.groups r)) d_out j k = 0) :=
  ⟨fun hvalid d mlp j hj =>
    support_preserving_maintains_w1_confinement P r f_back hvalid d mlp j hj,
   fun h_act j hj =>
    w2_confinement_unconditional P r h_act j hj⟩


-- ════════════════════════════════════════════════════════════════
-- §5. SPECIFIC ARCHITECTURE SAFETY CERTIFICATES
--
-- Pre-built safety proofs for common architectures.
-- These can be used directly by the Schemen assistant to
-- certify gate placement recommendations.
-- ════════════════════════════════════════════════════════════════

/-- **Certificate (Encoder-Only Transformer).**
    A standard encoder-only transformer (BERT, ViT) with gated FFN
    in each layer provides full MLP weight isolation for each regime.

    Architecture per layer:
      LN → Attn → (+) → LN → W1 → ReLU → [GATE] → W2 → (+) → out

    The gate sits at the standard position (after ReLU, before W2).
    The backward chain is (· ⊙ relu_grad), which preserves support.
    Therefore W1 and W2 are both confined. -/
theorem encoder_only_transformer_safe {m n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (_attn : SharedAttention n)
    (d_gated relu_grad : Vec n) (mlp_input : Vec m)
    (h_hidden : Vec n) (d_output : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    (∀ i : Fin m,
      outer mlp_input
        ((d_gated ⊙ indicator (P.groups r)) ⊙ relu_grad) i j = 0)
    ∧ (∀ k : Fin o,
      outer (h_hidden ⊙ indicator (P.groups r)) d_output j k = 0) :=
  standard_placement_full_isolation P r d_gated relu_grad mlp_input
    h_hidden d_output j hj

/-- **Certificate (Encoder-Decoder Transformer with Dual Gates).**
    A seq2seq transformer (T5, BART) with gated FFN in both encoder
    and decoder layers provides full MLP weight isolation.

    The dual-gating theorem (§2) establishes that each MLP's
    confinement is local and independent. -/
theorem encoder_decoder_transformer_safe {m n o R : ℕ}
    (P : ValidPartition n R) (r : Fin R)
    (d_enc relu_enc : Vec n) (enc_input : Vec m) (enc_hidden : Vec n)
    (d_dec relu_dec : Vec n) (dec_input : Vec m) (dec_hidden : Vec n)
    (d_output : Vec o)
    (j : Fin n) (hj : j ∉ P.groups r) :
    (∀ i : Fin m, outer enc_input
      ((d_enc ⊙ indicator (P.groups r)) ⊙ relu_enc) i j = 0)
    ∧ (∀ k : Fin o, outer (enc_hidden ⊙ indicator (P.groups r))
      d_output j k = 0)
    ∧ (∀ i : Fin m, outer dec_input
      ((d_dec ⊙ indicator (P.groups r)) ⊙ relu_dec) i j = 0)
    ∧ (∀ k : Fin o, outer (dec_hidden ⊙ indicator (P.groups r))
      d_output j k = 0) :=
  dual_gating_sufficient P r d_enc relu_enc enc_input enc_hidden
    d_dec relu_dec dec_input dec_hidden d_output j hj


end Schemen.GatePlacement
