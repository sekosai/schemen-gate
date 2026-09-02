# Claim boundaries

CDP combines a small algebraic mechanism with an authenticated runtime. The
mechanism and the runtime answer different questions and must be evaluated
separately.

## Supported claims

| Surface | Supported statement | Required conditions |
|---|---|---|
| Declared FFN tensor | A binary Hadamard gate makes every excluded coordinate exactly zero | The gate acts at the named tensor and no claim is made about upstream state |
| Intermediate FFN training | Aligned first-projection columns, hidden bias, down-projection rows, and optimizer state can be confined | Gate after the element-wise FFN activation, frozen undeclared state, support-aware optimizer, no bypass |
| Pre-MLP gate | Excluded FFN-input coordinates and aligned first-projection input gradients are zero | Gate immediately before the first projection; this is narrower than the intermediate placement |
| Whole-model authorization | Rejected scope makes zero model calls; an accepted call executes the unchanged model | Fail-closed decision before invocation and scope-bound runtime state |
| Authorized MoE routing | A learned router may select only inside a fixed execution-authorized expert set | Authority selects the candidate set before softmax/dispatch; prompt text is data, not authority |
| Private lanes | Tenant-stage training can leave a frozen backbone and inactive lanes unchanged | Complete private attachment selected by trusted authority; shared attention remains shared |

The cryptographic component binds authenticated scope to mask or resource
selection. The zero itself is an algebraic consequence of multiplication by a
binary mask; it is not a cryptographic hardness statement.

The resolved Regime has two security roles: its selection is authorization at
the ingress boundary, and its cryptographically bound identity authenticates
the execution context presented to downstream Gates. This second role is
conditional on preserving and verifying the same grant, model, operation, and
policy bindings; a bare integer or tensor does not authenticate itself.

Identity assurance terminates at the verifier's configured PKI roots and the
operator's issuance, private-key custody, rotation, revocation, and endpoint
controls. PKCS#12 packages credentials portably but does not prove hardware key
residency. A deliberately pinned self-signed root proves key possession and
continuity, not an independent third-party identity assertion.

## Claims this repository does not support

- complete multi-tenant isolation inside ordinary shared attention heads;
- end-to-end privacy, regulatory compliance, or absence of leakage through
  shared attention, embeddings, normalization, caches, logs, or bypasses;
- utility parity for a post-hoc narrowed Transformer;
- prompt text, a public token string, or a model-generated field as authority;
- storage compression when packed and separate expert bytes are equal;
- autoregressive generation quality from the token-routing classifier;
- bit identity as a general floating-point extraction guarantee; or
- proof that a deployment's Python hooks, tensor axes, IdP, or operator
  configuration match the Lean specification; or
- TPM, enclave, measured-boot, or confidential-computing protection merely
  because a credential was packaged as PKCS#12.

## Reading empirical results

Exact zeros in an artifact establish only the measured state or call boundary
under that runner. A zero-count wrong-key probe is a finite negative control,
not a universal privacy theorem. A non-significant paired test is not proof of
equality. One-seed results remain one-seed results even when every structural
assertion passes.

Start with `experiments/results/README.md` for canonical artifacts and
`docs/experiment-data-inventory.md` for claim-to-artifact custody.
