# Schemen Gate — Security Claims (code ↔ proof map)

Provenance: the 21 publication-facing Lean modules are included under
`research/cdp/proofs/`; this includes the cited GateSecurity and ModelSecurity
families. Claims requiring any theorem source not present in that directory are
excluded from the 1.0.2 proof claim. This file maps shipped `schemen_gate` code
to bundled proof, standard cryptographic assumptions, or observed tests and —
equally important — states what is *not* certified.

## Epistemic tiers

Every elementary claim in this project belongs to exactly one of three tiers.
A composed statement may depend on more than one tier only when each component
is named explicitly. Do not move a component between tiers without changing
its wording everywhere.

| Tier | Meaning | Examples |
|---|---|---|
| **Proven** | Bundled Lean theorem, kernel-checked, `sorry`-free | mask isolation, aligned update confinement |
| **Standard assumption** | Cited cryptographic assumption (PRF, EUF-CMA) | HMAC-SHA256 as PRF (represented by a bundled V2 axiom); Ed25519 EUF-CMA as an external assumption, not a bundled Lean axiom |
| **Observed** | Empirical hardening; measured, not proved | partition opacity of cotrained weights, superposition under capacity |

## Code ↔ theorem map

| Code path | Lean theorem(s) | Tier |
|---|---|---|
| `_crypto._csprng_permutation` (Fisher–Yates, HMAC-SHA256 counter mode, **rejection sampling**) | `rejection_sampling_count`, `rejection_unbiased` (GateSecurity §10) | Proven (uniformity); PRF assumption for indistinguishability |
| `_crypto.derive_partition` (equal slices, disjoint, exhaustive) | `ValidPartition` structure: `equal_size`, `disjoint`, `exhaustive`; `unique_membership` (GateSecurity §6) | Proven |
| `GateMask.apply` / `_mask.py` forward gating | `forward_isolation`, `gradient_isolation`, `gradient_confinement` (GateSecurity §1) | Proven |
| Gate-aware training with a conforming aligned update (including optimizer moments and weight decay) | `weight_update_confined`, `w2_update_confined`, `step_confined`, `end_to_end_isolation` (GateSecurity §2, §12, §13) | Proven for the modeled update contract |
| Wrong-mask inference behavior | `wrong_mask_reads_wrong_dims`, `wrong_key_valid_distribution` (GateSecurity §7, ModelSecurityV2 §4) | Proven |
| Exact combinatorial floor on partition recovery | `central_binom_lower`, `general_exponential_search`, `exceeds_aes256_security` (ModelSecurity §B–§E) | Proven |
| "Enumeration is optimal" (adversary must search) | `prf_brute_force_optimal` (ModelSecurityV2) — **axiom**, conditioned on opaque `Recovers` | Standard assumption |
| `_lockbox.py` grant signing, provenance, and verification | No bundled implementation-refinement theorem; negative tests cover trust roots, signatures, exact signed membership, expiry, revocation, key binding, and tamper | Observed implementation behavior under the selected EdDSA, ECDSA, or RSA signature assumption |
| `_pkcs12.Pkcs12KeyProvider` credential loading and signing | No bundled theorem; tests cover Ed25519, Ed448, ECDSA, RSA, key/certificate match, chain packaging, pinned trust anchors, certificate metadata, wrong-password refusal, signature verification, and wrong-root rejection | Observed implementation behavior under the selected signature assumption |
| `_tokens.py` AAD-bound token contracts | No bundled implementation-refinement theorem; canonical scope, release identity, expiry, tamper, and wrong-key tests | Observed implementation behavior under the AEAD/HMAC assumptions |
| `_cargo.py`, `_cargo_impl.py`, and `_rag.py` | No bundled theorem; adversarial tests cover exact manifest scope, finite operations, partition binding, immutable snapshots, material hashes, completion obligations, replay, and cross-partition denial | Observed implementation behavior under HMAC-SHA256 PRF/unforgeability assumptions |
| Cotraining opacity ("shared coplanes", inspection resistance) | No bundled confidentiality proof. Bounded historical observations are cataloged in `research/cdp/docs/experiment-data-inventory.md`; they are not promoted to a cryptographic claim. | Observed |

## Terminology note

Masks in this codebase are **binary 0/1 partition masks**. "Hadamard
product" throughout means the *elementwise (Schur) product* `h ⊙ M` — not
the ±1 Hadamard matrix construction. The Lean `hmul` definition is
elementwise multiplication; the wording is consistent if read this way.

## Scope exclusions (not certified, do not claim)

1. **execution binary integrity.** Release artifacts carry a build-time GitHub
   commit stamp, authenticated Gate contracts bind that release identity, and
   tagged public artifacts receive GitHub build-provenance attestations. These
   controls identify source and artifact bytes. They do not prove that the
   bytes executing on an operator's host are the published artifact; measured
   boot, native code-signing enforcement, or platform attestation remains a
   deployment control.
2. **Live-RAM extraction.** A currently-licensed operator with OS privileges
   can observe resident plaintext. Universal DRM limit; mitigation (enclave
   residency) is out of scope.
3. **Quantitative confidentiality (IND-CPA advantage).** Not formalized;
   the corpus bounds runtime *exposure* (what plaintext ever exists), not
   distinguishing advantage.
4. **V4 statistical camouflage** (`camouflage_indistinguishable`,
   `gradient_probing_hard`): retired. Superseded by custody (V5). The
   residual benefit of cotraining opacity is an **Observed**-tier claim,
   never a security foundation.
5. `PartitionOblivious` (V4 marker): the operational claim that training
   pipelines don't encode the partition is a deployment requirement, not a
   theorem.
6. **Trust-anchor distribution.** Certificate-chain validation proves only a
   path to a verifier-configured root. Secure distribution, rotation, and
   revocation of that external trust store remain deployment obligations.
7. **Cargo key custody and replay durability.** The library separates scoped
   manifest-access keys from bus-only receipt keys. Access keys and manifest
   AAD bind tenant, subject, regime, model, operation, policy version, and the
   exact partition. Cargo operations are restricted to `load`, `retrieve`, or
   the explicit `load_and_retrieve` union; the session verifies method
   authorization before invoking the storage adapter. Receipt keys and the
   verifier bind that same expected manifest; retrieval evidence additionally
   commits to the resolved query, request contract, ordered returned records,
   and returned vector material.
   The default bus rejects completion conditions it cannot independently
   evaluate. The in-memory bus rejects process-local manifest replay;
   cross-process replay prevention requires a durable, shared manifest-ID store
   at the serving layer.
8. **VON plaintext partitioning.** `gate_levels` is a deterministic research
   partition and does not consume a secret. It fails closed unless explicitly
   enabled as an unsafe proof-of-concept and carries no confidentiality claim.
9. **Hardware key residency.** PKCS#12 packages a private key, leaf certificate,
   and chain. The reference provider deserializes the key into process memory;
   it does not establish TPM, HSM, Secure Enclave, CNG, measured-boot, TEE, or
   confidential-computing residency. Those require a native `KeyProvider` and
   separate platform evidence.
10. **CA and endpoint operations.** Gate verifies the declared certificate path
    and its supported signed constraints against verifier-supplied root
    fingerprints. Gate performs no certificate-directed network egress; an
    operator-controlled fetcher supplies CRL/OCSP bytes, whose signature,
    freshness, identity, and scope Gate validates. Identity proofing, issuance,
    root distribution, key protection, renewal, revocation publication,
    transport egress, and host compromise remain within the operator's
    classical IT boundary.
11. **Lossless folding and model training.** `_regime0_fold.py` is an encoding
    codec, not a Gate or storage-authority boundary. `GatedRAGAdapter` confines
    its stable surface to retrieval and rejects model-training cache policies;
    it does not accept arbitrary optimizers or claim parameter-state
    confinement for them.

## Implemented security regression surface

Exact claim IDs and executable test node IDs are indexed in
[`CLAIM_TEST_MATRIX.md`](CLAIM_TEST_MATRIX.md).

The audited implementation additionally enforces external CA pinning for
lockboxes and model attestations, X25519-to-certificate key binding, validated
recipient chains, signed loading by default, authenticated and scoped CRL/OCSP
answers, controlled revocation retrieval, explicit revocation policy, versioned
canonical token AAD, complete-model ONNX hashing with confined external-data
paths, expiry-aware rights authorization, serialized Cargo session transitions,
atomic Cargo store batches, exact-manifest receipt verification,
material-output receipt hashing, verifier-owned completion state,
finite Cargo operation enforcement, exact signed-lockbox grant membership,
partition-scoped vector-store updates, pre-decode inference-frame bounds,
secret-redacted workload identity representations, immutable upstream
revisions for executable research inputs, detached public mask and metadata
copies, exact live Cargo partition-to-Regime binding, strict payload-kind and
embedding validation, immutable store ingress and egress, load-completion
obligations, internally consistent receipts, bounded canonical VON and vector
frames, hostile vector-store result validation, rejection of malformed
PostgreSQL vector rows instead of synthetic substitution, source-space
Gate-before-bridge projection, bounded duplicate-free Cargo JSON parsing,
strict lockbox YAML parsing, bounded PKCS#12 input, authenticated recipient
issuer paths before revocation, and delegated OCSP Key Usage plus OCSP No Check
enforcement. Explicit signed document IDs remain bound
to accepted store results, and the PostgreSQL adapter serializes complete
operations on its shared connection so transactions cannot interleave. The
lossless folding utility no longer exposes a gate-bearing API, and the RAG
adapter rejects training cache policies rather than invoking an unverifiable
optimizer. The history-free release verifier also rejects recoverable prior Git
objects and reflogs before transfer. The authenticated release surface is also
release-identity bound: mask and adapter AAD, rights HMACs, signed lockboxes,
capability tokens, Cargo manifests and receipts, and operation-gate public
attestations all carry the semantic version and exact build-stamped source
commit. Distribution inspection and clean-wheel tests reject a missing or
mismatched commit stamp, and tagged public artifacts receive GitHub build
provenance. The research launch surface is source-visible, revision-pinned,
syntax/lint
checked, remotely measured against image-baked source and launcher digests, and
free of bundled executable companion wheels. These are implementation
hardening controls. They do not expand the bundled theorem inventory above.

Unbundled historical theorem or service implementations are not part of the
Schemen Gate 1.0.2 release claim.
