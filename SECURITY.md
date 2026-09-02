# Security Policy

## Supported version

Security fixes are applied to the latest 1.x release line.

## Reporting a vulnerability

When the canonical repository is public and private vulnerability reporting is
enabled, use **Security → Report a vulnerability**. That GitHub feature is a
repository setting; it is unrelated to Gate policy, Cargo Mode, or an
operator's execution-environment security mode.

Until that feature is enabled, email
[ryan@sekos.ai](mailto:ryan@sekos.ai) with the subject
`Schemen Gate security`. Do not disclose suspected vulnerabilities in a public
issue. Repository maintainers can create a draft Security Advisory after
private contact.

Include the affected version, entry point, impact, minimal reproduction, and
whether the report involves key custody, trust anchors, replay, path scope, or
runtime authority. Never include live credentials or production key material.

## Security boundary

Schemen Gate authenticates declared contracts and authority. It does not prove
source truth, policy wisdom, secure deployment configuration, or containment of
arbitrary code. See `docs/SECURITY_CLAIMS.md` for exact claim tiers and exclusions.
The verifier owns its CA fingerprints and revocation policy; a certificate or
caller-supplied chain cannot add itself to that trust set.
The executable security disposition is maintained in permanent regression tests
and `docs/CLAIM_TEST_MATRIX.md`. Internal iterative review records are retained
privately; they are not part of the public package or its security claim.
The trust-boundary review rules, decision inventory, and enforced package
complexity ceiling are documented in `docs/SECURITY_ENGINEERING.md`.
