# Release verification and maintenance

Use this guide to evaluate a published Gate revision and report reproducible
issues. Repository documentation describes verification procedures; it is not
an attestation of current hosting-account settings or deployment state.

## Verify the version you use

- Match the package version, full source commit, and canonical repository to
  the release you intend to evaluate.
- Check source identity, artifact contents, and available attestations using
  the [installation guide](INSTALLATION.md), [provenance contract](PROVENANCE.md),
  and [release-attestation contract](RELEASE_ATTESTATION.md).
- Inspect [GitHub CI](https://github.com/sekosai/schemen-gate/actions) for that
  exact revision. A passing run for another commit is separate evidence.
- Run the documented quickstarts in your environment and retain both success
  and denial results. Package and artifact availability must be verified at
  their published source before installation.

## Evaluate an upgrade

Read the [changelog](../CHANGELOG.md) and [API stability contract](API_STABILITY.md).
Test the upgrade with your supported dependency versions and integration.
Recheck the relevant [deployment obligations](PRODUCTION_DEPLOYMENT.md),
including authority, revocation, replay, alternate execution routes, and
release-identity expectations.

## Report an issue

Include the exact version and source commit, dependency versions, a minimal
reproduction, and the expected and observed behavior. Remove credentials and
private workload data from public reports. Follow
[SECURITY.md](../SECURITY.md) for suspected vulnerabilities and
[CONTRIBUTING.md](../CONTRIBUTING.md) for patches.

Public contribution areas are listed in the [roadmap](../ROADMAP.md).
