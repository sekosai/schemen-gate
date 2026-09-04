# Schemen Gate roadmap

This roadmap covers the open-source Gate library. It describes areas for
contribution, not delivery dates or guarantees of future functionality.
For shipped behavior, use the [README](README.md), [changelog](CHANGELOG.md),
and documentation at the release you are evaluating.

## Areas for contribution

- **Clearer onboarding.** Improve installation guidance, examples, and
  explanations of the certificate-to-grant-to-Gate path.
- **Reproducible integrations.** Contribute small examples with an explicit
  protected operation, verifier-owned trust, and both success and denial cases.
- **Compatibility.** Report reproducible issues against supported dependencies
  while preserving the documented [1.x API contract](docs/API_STABILITY.md).
- **Security and correctness.** Strengthen adversarial regression coverage and
  keep claims tied to the implementation, tests, and stated assumptions.
- **Measurement.** Contribute repeatable benchmarks with exact versions,
  workload definitions, and limits on what the results establish.

## Propose a change

Open an [issue](https://github.com/sekosai/schemen-gate/issues) describing the
use case, affected Gate boundary, and a small reproducible example. For a new
integration, include the expected authorized behavior and the requests it must
reject. Discuss substantial API or authority changes before implementation.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and review requirements.
Report suspected vulnerabilities through [SECURITY.md](SECURITY.md).
