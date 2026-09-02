# Public-for-all is an explicit Gate policy

Security is not maximized by making every operation maximally restrictive.
Organizations intentionally publish models, documentation, inference paths,
datasets, and capabilities. A principled authority system must represent that
decision without confusing it with a broken or missing check.

Schemen Gate uses three distinct states:

1. **Scoped:** a verified identity receives only the signed Regime and
   operation it was granted.
2. **Public-for-all:** policy deliberately authorizes every caller for the
   declared resource or activation support.
3. **Unresolved or invalid:** no applicable policy was established; deny.

Public-for-all is therefore not “fail open.” It is a positive policy decision.
For binary activation, `GateMask.public(n_dims)` returns the all-ones mask and
records `access_policy: public-for-all`. The Gate is still present and its
placement remains observable, but it intentionally excludes no coordinate.
`GateMask.full` remains a backward-compatible alias.

```text
scoped grant verified     -> apply the granted support
public-for-all selected   -> apply the explicit all-ones support
missing/invalid decision  -> do not execute
```

This distinction makes audit evidence meaningful. `GateMask.public()` itself is
unsigned, caller-constructible data: it represents the decision but does not
authenticate who made it. An authority-signed policy or operator-integration
receipt can state that a resource was public by policy rather than forcing an
auditor to infer whether an absent denial was intended. It also lets one
deployment mix public and restricted Regimes without weakening the restricted
path.

## What public-for-all does not mean

- It does not certify isolation; every declared coordinate is active.
- It does not authorize a caller to relabel a restricted resource as public.
- It does not turn verification errors, missing runtime dependencies, or
  unavailable policy stores into public access.
- It does not remove licensing, privacy, export, safety, or contractual duties
  that exist outside the Gate policy.

Only the policy authority or trusted operator integration may choose public-for-all for a
protected route. Untrusted request data cannot select it.
