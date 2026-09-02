# Operator execution boundary

Schemen Gate v1 is a library, not a serving system. It deliberately ships no
HTTP server, bearer-token router, model loader, cloud SDK, deployment
credential, or provider policy. An operator embeds the library into the process
that owns the protected action and must ensure every path to that action crosses
the same verified decision.

## What ships here

- NumPy-based masks and deterministic partition derivation;
- scoped capabilities, exact-operation gates, lockboxes, Cargo, and optional
  storage and identity adapters;
- executable local and Modal examples;
- source-visible research preflights that test rejection-before-callback and
  explicitly make no production authority claim; and
- the paper, proofs, receipts, and release verification tooling.

No companion repository or executable server package is required to
install, test, inspect, or reproduce this release.

## What the operator supplies

The embedding application owns authentication ingress, policy resolution,
durable replay control, model or resource custody, network exposure, request
limits, logging, telemetry, and bypass closure. It must place Gate verification
immediately before the protected callback, resource release, storage action, or
declared activation and test every alternate route to the same asset.

The small `research/cdp/experiments/execution_preflight.py` fixture is useful only
for falsifying callback ordering in experiments. It is not an identity
provider, credential verifier, network service, or production authorization
implementation.

## Repository admission rule

Code belongs in this repository only when it:

1. can execute without a private control plane or companion product;
2. does not expose a production network endpoint or launch untrusted code;
3. does not require deployment credentials;
4. has a deterministic, testable Gate contract; and
5. remains useful to third-party integrations through the public Python API.
