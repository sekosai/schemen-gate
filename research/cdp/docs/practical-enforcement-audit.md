# Practical enforcement audit

This document separates CDP's algebraic Gate from the identity, custody, and
operator controls required to enforce it. It describes the public Schemen Gate
1.0.0 boundary only.

## Audit verdict

The repository ships a self-contained Gate library, negative tests, proofs,
papers, and research runners. It does not ship an IdP, policy service, model
server, network endpoint, durable use ledger, or deployment control plane. A
deployment may claim the local Gate properties only at paths where it supplies
and tests those missing controls.

The central distinction is:

- external identity infrastructure authenticates the calling principal or
  machine against operator-selected roots;
- policy authorizes that identity to a precise resource and operation;
- signed Gate contracts bind the resolved subject, Regime, model, operation,
  partition, policy version, and lifetime;
- the operator applies the verified decision at the protected computation or
  release boundary; and
- durable state records replay, revocation, and audit events across replicas.

## Strict request anatomy

For each invocation, a conforming production path performs these steps in
order:

1. Validate the caller's credential against the configured issuer, audience,
   signature, validity, and revocation policy.
2. Resolve the canonical subject and tenant. Caller-provided tenant or Regime
   identifiers are requests, never authority.
3. Ask policy whether that subject may perform the operation on the exact
   model, corpus, attachment, tool, or FFN capability.
4. Issue or resolve a signed grant with an immutable model digest, operation,
   resource scope, policy version, expiry, and replay identifier.
5. Verify the grant and exact signed membership against independently
   configured roots.
6. Atomically reserve finite-use or token budgets before execution. Fail closed
   when current policy, revocation, or metering state is unavailable.
7. Apply the Gate before the protected callback or at every path crossing the
   declared activation surface.
8. Namespace caches, optimizer state, logs, and other mutable state wherever
   the deployment claims isolation.
9. Append a non-secret evidence record linked to the subject, grant, resource,
   operation, decision, and deployed version.

## Whole-model gates are structured capabilities

A whole-model gate is not an all-ones FFN mask. It authorizes an unchanged
model as one atomic resource and rejects before any model call. It can bind:

- subject and tenant;
- model identity and immutable version;
- operation and purpose;
- expiry and policy version;
- delegation chain;
- input and output ceilings;
- cumulative invocation or token budget; and
- audit identity.

Whole-model gates can be chained. Authorization to retrieve a corpus does not
imply authorization to invoke a generator or tool. Each edge needs its own
resource-and-operation scope.

## AAD and exact-use enforcement

Authenticated data makes a limit tamper-evident; it does not make the limit
stateful. A token declaring `max_uses=10` can be replayed unless an online,
atomic ledger records consumption.

Exact cumulative enforcement therefore needs a durable ledger keyed by grant
ID and epoch. Reserve the maximum authorized use before execution, settle the
actual use afterward, and deny ambiguous or concurrent over-redemption. The
library's in-process replay sets are not a distributed ledger.

## Principal-scoped derivation and revocation

Deriving authority for each complete subject and resource scope limits blast
radius. Revocation still cannot claw back copied plaintext key bytes. Immediate
revocation requires the operator to mark the grant inactive, reject its old
epoch on every mediated operation, and issue replacement authority only when
policy permits.

## Key and trust-root compromise

- Theft of a properly sealed artifact does not by itself reveal every
  recipient-wrapped key.
- Theft of one scoped key compromises the operations that accept that key until
  they reject its grant or epoch.
- Exfiltration of a root, unrestricted derivation key, signing key, or live
  authorized plaintext is a boundary compromise.

The operator owns non-exportable custody, separation of duties, recovery,
rotation, revocation publication, and immutable auditing. The portable PKCS#12
provider loads its private key into process memory and does not prove hardware
residency.

## Shipped implementation inventory

The public Gate includes:

- deterministic binary masks and keyed equal-width partition derivation;
- canonical scoped tokens and exact finite-operation contracts;
- signed lockboxes, exact grant-membership checks, recipient wrapping, and
  PKCS#12/X.509 verification against operator-selected roots;
- Cargo manifests and receipts with exact partition, operation, payload, query,
  returned-material, and completion-obligation binding;
- in-memory and PostgreSQL vector-store adapters with partition checks;
- bounded vector-frame and VON parsers; and
- source-visible research preflights that test callback ordering but make no
  production authority claim.

## Strict-profile acceptance checks

A deployment may claim the strict practical profile only when all of these
checks pass:

- direct subject, tenant, Regime, model, partition, and operation overrides are
  rejected;
- the complete canonical scope reaches the final Gate;
- wrong root, subject, tenant, model, operation, Regime, partition, grant,
  epoch, expiry, or policy version yields zero protected calls;
- identity, policy, revocation, audit, and metering outages fail closed;
- replay and concurrent-use tests cannot exceed the configured budget;
- a revoked principal stops working while unrelated principals continue;
- every alternate route to the protected resource crosses the same decision;
- storage and cache results cannot cross their bound partition;
- model, FFN, optimizer, cache, and logging paths match the claimed isolation
  surface; and
- root recovery and rollback are exercised as operational drills.
