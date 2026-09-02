# Changelog

This project follows a research-release model: signed tags identify immutable
paper, proof, code, and artifact bundles. Result artifacts are never silently
replaced; corrections receive a new artifact and an explicit manifest note.

## 1.0.2 - Release candidate

- Added reproducible Lean and GitHub CI configuration.
- Added public setup, local examples, and a safe Modal onboarding canary.
- Refreshed runnable reproduction pins to Torch 2.13.0, datasets 5.0.1,
  cryptography 50.0.0, and Modal 1.5.4. Historical result receipts remain
  unchanged and any rerun produces a new artifact.
- Added community health, citation, security, and open-source release guidance.
- Corrected proof-suite axiom inventory wording and build compatibility.
- Clarified the external AuthN, boundary AuthZ, and downstream Regime AuthN
  chain; documented PKCS#12, self-signed root pinning, and hardware-custody
  limits in the paper and claim boundaries.
