# Security policy

This repository is a research and reproduction bundle. It is not a production
identity provider, key-management service, lockbox, policy engine, or model
server. The paper's guarantees apply only at the explicitly declared FFN or
whole-model gate and under its trusted-runtime assumptions.

## Reporting a vulnerability

Follow the repository-wide reporting policy in
[`../../SECURITY.md`](../../SECURITY.md). Use GitHub's private
vulnerability-reporting flow when it is enabled. If that flow is unavailable,
email [ryan@sekos.ai](mailto:ryan@sekos.ai). Do not include credentials,
customer data, unpublished model weights, or active exploit details in a
public issue.

Include the affected revision, runner or proof module, expected boundary,
reproduction steps, and whether the issue could cause an unauthorized model
call, state update, cache reuse, or gate bypass.

## In scope

- fail-open authorization or model invocation before authorization;
- wrong-scope access to a gated activation, parameter slice, artifact, or
  private lane;
- optimizer or residual paths that bypass the declared boundary;
- artifact/provenance verification that accepts mismatched bytes; and
- contradictions between executable behavior and a security claim.

General model-quality disagreements and unsupported extensions of the paper's
scope belong in ordinary issues rather than private security reports.
