# Evaluating and adopting Schemen Gate

Start with one protected operation and a result you can test. The open-source
Gate library can be installed and evaluated using the repository's examples.
See the [product boundary](../README.md#gate-and-runtime) for the distinction
between this library and the separate serving product.

## Run the local example

Follow the [certificate-to-Gate quickstart](../README.md#two-minutes-to-working-ai-pki).
It exercises a valid authority path and wrong-root and wrong-recipient denials
using ephemeral local credentials. The example uses an explicit offline
revocation policy; production requirements are documented separately.

## Define your integration boundary

Choose one operation, such as releasing an encrypted shard, retrieving scoped
context, or admitting a model capability. Identify:

- the subject and independently configured trust root;
- the exact resource, operation, and scope authorized by the grant;
- the point where authority is checked before protected execution; and
- the evidence needed to distinguish success from denial.

The [API guide](USAGE.md), [operator boundary](OPERATOR_BOUNDARY.md), and
[claim-to-test matrix](CLAIM_TEST_MATRIX.md) connect these choices to supported
interfaces and executable examples.

## Test the denials

Exercise wrong signer, wrong subject, wrong scope, expiry, and tampered
contracts. Check that rejected requests do not reach the protected operation.
For your actual deployment, also test alternate routes, replay handling,
revocation behavior, and evidence retention as required by the
[production deployment contract](PRODUCTION_DEPLOYMENT.md).

Record the exact Gate commit, dependencies, configuration, and workload with
results. The [security claims map](SECURITY_CLAIMS.md) separates local proof,
cryptographic assumptions, implementation evidence, and deployment obligations.
A passing library example does not establish a deployment's complete boundary.

## Get help or contribute

For reproducible integration questions, open an
[issue](https://github.com/sekosai/schemen-gate/issues) with the version, minimal
example, expected behavior, and observed result. Follow
[CONTRIBUTING.md](../CONTRIBUTING.md) for changes and
[SECURITY.md](../SECURITY.md) for suspected vulnerabilities. Integration
inquiries can be sent to [Sekos AI](mailto:ryan@sekos.ai).
