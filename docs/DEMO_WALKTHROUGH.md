# Live demo walkthrough: the same model, different authority

The live demo is at <https://demo.sekos.ai/cdp>. It is a teaching fixture,
not a production deployment: it exists to let you watch the Gate, rather than
the caller, decide which parts of a model may run, and to hand you signed
evidence you can check yourself. This page explains what the demo does behind
each click and which Schemen Gate calls do the work.

## What you are looking at

- One co-trained MNIST digit model. Its expanded hidden activation has 2,560
  coordinates, derived into ten disjoint Regimes of 256 coordinates each with
  `GateMask.derive`, one Regime per digit.
- Ten one-versus-rest detector heads, one per digit. A head can only see its
  own Regime's coordinates; every other coordinate is an exact zero after the
  Gate.
- A workload identity: a pinned, self-signed certificate for a workload named
  `flagship-workload`. The verifier trusts it because its fingerprint is pinned
  in the deployment, not because a CA vouched for it.
- The workload arrives holding grants for digits 0, 1, and 2. Digit 9 is
  published for all callers as an explicit public policy with no identity
  claim. Digits 3 to 8 are closed.

## The ninety-second path

1. Pick the tile for digit 7. The ground-truth label is teaching metadata; it
   never enters the request.
2. Click **Recognize**. The answer is `UNRESOLVED`: lanes 0, 1, 2, and 9 ran,
   each confined to its own block of coordinates, and every other lane is
   black because it never executed. No error was raised. The recognizer for 7
   simply did not run.
3. Click the digit-7 tile. That is the Grant action: a signed, recipient-bound
   grant for digit 7 is minted and added to the workload's bundle.
4. Click **Recognize** again with the same pixels. The answer is `7`, lane 7
   lights, and the shared rows are bit-identical to the earlier run. The
   comparison card shows the two signed records side by side with drift
   exactly 0.0.

## What a Grant does

The page calls the demo's playground authority:

```text
POST /v1/cdp-playground-authority/grant/<digit>
{"recipient_cert_pem": <workload certificate>, "recipient_id": "flagship-workload"}
```

The authority is deliberately anonymous so that visitors can act as the
administrator; its health endpoint reports `unsafe_demo_authority: true`. In a
real deployment this role belongs to the operator's issuing authority. For each
grant it runs the library's normal issuance path:

| Step | Schemen Gate call |
|---|---|
| Build a one-level lockbox naming exactly one Regime and one capability `digits/recognize/<d>-v1` with operation `recognize`, bound to the model manifest hash | `create_lockbox`, `HierarchyDef`, `RegimeCapability` |
| Bind the exact ONNX artifact digest | `Lockbox.model_artifact_hash` |
| Seal the Regime key and mask token to the workload's certificate | `seal_grant` with `recipient_cert_pem` |
| Sign the lockbox with the authority key and attach its pinned root | `sign_lockbox` |

The response is an encoded package the browser keeps client-side and presents
with every inference request.

## What Recognize does

The page posts the pixels, the operation, and the grant packages:

```text
POST /v1/cdp-v2/infer
{"operation": "recognize", "input": {"kind": "pixels-u8-v1", "values": [...]}, "grant_bundle": [<packages>]}
```

The service serializes every request and records the model-call counter before
and after. The steps, in order, and where each denial stops:

| Stage | What is checked | Schemen Gate call | Denial code on failure |
|---|---|---|---|
| request parsed | exact request document, known input kind, operation is `recognize` (the only operation this service accepts, so a `write` request never reaches a grant) | | `REQUEST_INVALID` |
| package decoded | authority signature against the pinned root fingerprints, expected Gate release, recipient identity | `verify_authority` path inside the package decoder | `GRANT_NOT_SIGNED_MEMBER` |
| model bound | lockbox chain hash equals the model manifest hash; lockbox artifact hash equals the ONNX digest | | `MODEL_ARTIFACT_MISMATCH` |
| Gate release bound | lockbox release identity equals the service's release | `release_identity_matches` | `GATE_RELEASE_MISMATCH` |
| grant verified | exact signed membership of the grant in its lockbox, recipient certificate binding, trust root; revocation is explicitly skipped for fixture credentials | `verify_grant_provenance` | `GRANT_NOT_SIGNED_MEMBER`, `GRANT_RECIPIENT_MISMATCH`, `TRUST_ROOT_MISMATCH` |
| access resolved | the grant's access level names one Regime and one capability | `resolve_access` | `ACCESS_FINGERPRINT_UNKNOWN`, `CAPABILITY_BUNDLE_INVALID` |
| operation and resource bound | capability label equals the requested operation; resource is `digits/recognize/<d>-v1` for that Regime | | `SCOPE_OPERATION_MISMATCH`, `CAPABILITY_RESOURCE_MISMATCH` |
| mask redeemed | unwrap the Regime key with the workload's private key, redeem the AAD-bound mask token, rebuild the mask, check geometry (2,560 coordinates, ten Regimes) and the mask digest against the manifest | `unseal_grant`, `redeem_mask_token`, `GateMask.from_numpy` | `MASK_CONTRACT_MISMATCH` |
| public policy applied | digit 9 is added from the explicit, release-bound public policy; it carries no identity | `GateMask` public capability | `PUBLIC_POLICY_INVALID` |
| effective set non-empty | at least one capability resolved | | `NO_ACTIVE_CAPABILITIES` |
| model executed | one batched forward of only the granted detector heads; the resolved masks are applied to the tapped post-GELU expanded activation before the down projection | `GateMask.apply` | |
| call contract checked | the counter moved by exactly one for an allow and exactly zero for a deny; otherwise the record is downgraded to a denial | | `EVIDENCE_FAILURE` |

Every response is a signed evidence record (schema `cdp/inference-evidence-v1`).
An allow returns HTTP 200; every denial returns HTTP 403 with the same signed
record shape and `model_call_delta` 0.

## The denial buttons

Each button sends a real request; nothing is simulated in the browser.

| Button | What is presented | Expected record |
|---|---|---|
| Use wrong certificate | a grant the playground authority issued to a certificate this verifier does not trust | `DENY`, `GRANT_RECIPIENT_MISMATCH`, `model_call_delta` 0 |
| Tamper with a grant | an issued package with one byte flipped after signing | `DENY`, `GRANT_NOT_SIGNED_MEMBER` |
| Use wrong operation | a valid grant presented with operation `write` | `DENY`, `REQUEST_INVALID` at request parsing, before any grant is opened |
| Present unsigned private grant | a fabricated object with no authority signature | `DENY`, `GRANT_NOT_SIGNED_MEMBER` |
| Run as anonymous public caller | no grants and no identity, digit-9 fixture | `ALLOW` with `trust_mode` `public_for_all`; only lane 9 executes |

## Verify the evidence yourself

The evidence signer's public key is published in the session document:

```text
GET /v1/cdp-v2/session      -> evidence_public_key (Ed25519, base64)
POST /v1/cdp-v2/verify      -> {"verified": true|false} for a record you paste back
```

Offline verification needs no service: remove the `signature` field, serialize
the remaining record as canonical JSON (sorted keys, compact separators, ASCII),
and verify the Ed25519 signature with the pinned public key. The record binds
the model manifest digest, the ONNX file digest, the capability set digest and
each resolved capability's lockbox, grant, mask, and subject fingerprints, the
Gate release identity (version and exact source commit), the request and input
digests, the executed and not-executed detector ids, the per-detector logits,
and the call counter before and after.

## Check the Gate identity

The session and every record carry `gate_release.source_commit`: the exact
Schemen Gate source commit the service is running. Compare it with the release
you install:

```bash
git rev-parse v1.0.2^{commit}
```

```bash
python -c "import schemen_gate; print(schemen_gate.current_release_identity().to_dict())"
```

If the commits differ, the service is running a different build of the same
version number, and its records are reproducible only from that build. Version
equality is not source identity; the record tells you which one you have.

## Trust modes shown

- **Pinned self-signed** (live): the workload certificate and the authority
  root are pinned fingerprints. Revocation checking is skipped for these
  fixture credentials, and the page says so.
- **Public for all** (digit 9): an explicit `GateMask.public` policy bound to
  the model manifest and Gate release. It is an authority decision, not a
  missing check, and it asserts no identity.
- **CA-trusted**: supported by the protocol and the library's X.509 profile,
  not wired in this fixture deployment.

## What the demo proves, and what it does not

Supported by the evidence: certificate-verified grants; exact lockbox
membership; authority-resolved Regimes; the released binary Gate zeroing
excluded coordinates at the declared tensor; only granted detector heads in the
observed batch; zero model forwards on every observed denial; bit-identical
shared-detector logits across capability changes; independently verifiable
signed records.

Not proved: complete model or tenant privacy; partition of shared inputs,
process memory, logs, serving state, or side channels; that the operator's
issuance or key custody is wise; hardware-backed keys; universal bypass closure
from local call counting; a universal security theorem from finite denial
tests. The model is deliberately small and demonstrates the enforcement
contract, not state-of-the-art accuracy. The Lean theorems, the Python library,
the trained artifact, and the deployment are distinct evidence layers, and the
demo's own artifact manifest records its training and selection metrics.
