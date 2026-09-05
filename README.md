# Schemen Gate

**AI PKI: bind workload identity to the AI resource or operation it may use.**

Schemen Gate carries verifier-trusted identity and signed, scoped authority to
an AI execution boundary. It supplies the cryptographic contracts and Gates
used to admit a model operation, release an encrypted shard or retrieval
result, or select a declared activation partition—and retain evidence of the
decision.

```text
trusted identity → signed grant → authorized Regime → Gate → execution evidence
```

A **Regime** is the execution scope resolved from verified authority. It can
select a model capability, attachment, data partition, or declared activation
support. A caller-supplied Regime number or mask does not authenticate itself.

## Gate and Runtime

**Gate is the open-source enforcement library. Schemen Runtime is a separate,
closed-source serving product.** Runtime consumes Gate contracts at authenticated
inference endpoints and supplies model loading, request enforcement, and
governed execution. The two projects have distinct release identities and
deployment responsibilities.

Use Gate to embed these controls in your own application. Use the separate
Runtime when you need Schemen's closed-source serving implementation; contact
[Sekos AI](mailto:ryan@sekos.ai) for Runtime evaluation. Runtime is not bundled
in the Gate wheel, and no companion service is needed to install Gate or run
its examples.

## Two minutes to working AI PKI

From a clean clone:

```bash
git clone https://github.com/sekosai/schemen-gate.git
cd schemen-gate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[lockbox]'
python examples/ai_pki_quickstart.py
```

Expected output includes:

```text
PASS: certificate -> signed grant -> resolved Regime -> Gate
PASS: wrong trust root denied before Gate
PASS: wrong recipient certificate denied before Gate
```

The example creates ephemeral local credentials, verifies an independently
configured authority root and exact signed grant membership, checks the
recipient certificate, unwraps its Regime key, and applies the authenticated
mask. It also exercises wrong-root and wrong-recipient denials. It uses an
explicit offline-fixture revocation policy; production requirements are in the
[deployment contract](docs/PRODUCTION_DEPLOYMENT.md).

For the NumPy-only algebra, run `python examples/quickstart.py`. For portable
credential loading and signing, run `python examples/pkcs12_identity.py`.

## See the authority change

Open the [live digit-model demo](https://demo.sekos.ai/cdp). Select the digit-7
input and recognize it before and after granting capability 7. The same pixels
become recognizable when the granted detector may execute; the shared detector
rows stay unchanged.

The demo also exposes malformed-authority cases and signed evidence with model
call counts. Its anonymous public-caller case demonstrates deliberate public
access. Each record identifies the deployed Gate source commit: compare that
identity with the release you are evaluating. The demo is a bounded co-trained
MNIST fixture, not evidence of complete model privacy.

[Follow the demo walkthrough](docs/DEMO_WALKTHROUGH.md).

## Choose a starting point

| Your task | Gate surface | Start here |
|---|---|---|
| Authorize an existing model or protected operation | Scoped capabilities and exact-operation gates | [API guide](docs/USAGE.md#other-primitives), [operator boundary](docs/OPERATOR_BOUNDARY.md) |
| Release encrypted model shards | Signed lockboxes and recipient-specific grants | [Executable shard fixture](examples/cotrained_shard_lockbox/README.md) |
| Release context or vectors | Cargo manifests, material-bound receipts, and gated retrieval | [Cargo contract](docs/CARGO_MODE.md), [API guide](docs/USAGE.md#other-primitives) |
| Gate a NumPy or PyTorch activation | Immutable `GateMask` | [First Gate](docs/USAGE.md#first-working-gate), [API reference](docs/USAGE.md#gatemask-reference) |
| Train with declared private support | Gate placement, frozen shared state, and aligned updates | [Training and model boundaries](docs/TRAINING.md) |

The core activation operation is deliberately small:

```python
gated = hidden * authorized_binary_mask
```

Its value depends on the authority that selects the mask, its placement, and
the state governed by it. Applying a mask after ordinary training does not
retroactively create tenant-private knowledge.

## Identity and security boundary

The authority path connects **external AuthN** (verify identity against an
independent root), **boundary AuthZ** (verify the exact signed scope), and
**downstream Regime AuthN** (carry the resolved execution identity to subsequent
Gates that verify the same bindings).

- **The verifier owns trust.** Gate has no built-in CA store or preferred
  issuer. The operator supplies approved root fingerprints; a credential or
  bundle cannot nominate its own root as trusted.
- **The operator controls revocation egress.** Gate does not ship a network fetcher.
  Production revocation checks require an operator-supplied fetcher that enforces
  DNS, redirect, TLS, timeout, and response-size policy; see the
  [identity guide](docs/USAGE.md#use-a-pkcs12-machine-identity).
- **The contract names the scope.** Bind the subject, model, operation, Regime,
  lifetime, policy context, and release identity. A grant must be an exact
  member of its authority-signed lockbox.
- **Denial precedes protected execution.** The embedding application must close
  alternate routes to the same resource and provide the required durable
  replay and usage controls.
- **Key custody is explicit.** `Pkcs12KeyProvider` loads the signing key into process memory.
  Hardware-backed custody requires an appropriate native `KeyProvider` and
  separate platform evidence.
- **Public access is an authority choice.** `GateMask.public(n_dims)` represents
  an all-ones policy. Trusted policy must select it; the mask alone does not
  authenticate that choice. See [public-for-all policy](docs/PUBLIC_FOR_ALL.md).

The [security claims map](docs/SECURITY_CLAIMS.md) distinguishes local Lean
theorems, standard cryptographic assumptions, observed implementation behavior,
and deployment obligations. Local activation zeros and modeled aligned updates
do not establish complete-model privacy, statistical independence, host
integrity, or universal bypass closure.

Before real traffic, follow the [production deployment contract](docs/PRODUCTION_DEPLOYMENT.md)
and exercise its denials at the actual serving boundary. The
[X.509 profile](docs/X509_PROFILE.md) defines supported certificate semantics;
the [claim-to-test matrix](docs/CLAIM_TEST_MATRIX.md) locates executable evidence.

## Install version 1.0.2

Version 1.0.2 is the production-ready release candidate for the documented
library boundary. The source quickstart above is available now. Package-index
publication and downloadable release artifacts are separate release steps;
verify their availability and provenance before installing by package name.

The core requires Python 3.10 or newer and NumPy. Signed grants use the
`lockbox` extra; other optional capabilities include `crypto`, `torch`, `onnx`,
`rag`, and `spiffe`. See the [dependency map](docs/DEPENDENCIES.md).

For a reviewed wheel such as `schemen_gate-1.0.2-py3-none-any.whl`, follow the
[installation and verification guide](docs/INSTALLATION.md). It covers the
locked build, exact source identity, archive verification, and GitHub artifact
attestations. Match the complete source commit as well as the version number.

The wheel contains the Gate library. The source distribution also includes
operational documentation, examples, tests, and release tools. Papers, proofs,
and research receipts remain in this Git repository at the corresponding
revision. Historical receipts retain the versions under which they were made.

## Training is a lifecycle choice

Choose the protocol that matches the state you intend to govern:

- Co-training with per-sample masks permits updates to a shared encoder.
- Strict private FFN training requires a frozen shared backbone, correctly
  placed activation Gates, and support-aligned updates including optimizer
  moments and weight decay.
- Private adapters or experts preserve per-Regime capacity at additional
  storage cost.

The [training guide](docs/TRAINING.md) retains the runnable patterns, dimension
rules, public-adaptation caveats, and attention/residual/cache boundaries.
`GateMask.apply` does not install model hooks or supply a masked optimizer.

## Papers and evidence

The [research bundle](research/cdp/README.md) contains the papers, Lean sources,
experiment code, launchers, and retained receipts. Begin with:

- [Binary Activation core paper](research/cdp/output/pdf/binary-activation-core.pdf)
  and [source](research/cdp/paper/split/binary-activation-core.tex).
- [Transformer boundary paper](research/cdp/output/pdf/binary-activation-transformers.pdf)
  and [source](research/cdp/paper/split/binary-activation-transformers.tex).
- [Full manuscript](research/cdp/paper/cdp.pdf) and [source](research/cdp/paper/cdp.tex).
- [Schemen-gated Transformer regime-lane paper](research/cdp/gated-transformer-regime-lanes/SCHEMEN_GATED_TRANSFORMER_REGIME_LANES_PAPER.md)
  and its [results, failures, and corrections](research/cdp/gated-transformer-regime-lanes/RESULTS_AND_CORRECTIONS.md).

The proof and experiment claims apply to their named constructions. A proof of
the local Gate algebra is not a proof of arbitrary Python execution or a
particular deployment. The regime-lane study is a separate bounded empirical
result: it does not amend the claims of the original CDP manuscript or establish
universal or production-grade Transformer isolation.

## Run a remote example

The optional Modal CPU canary runs the certificate-to-Gate example and its
denials against an exact tracked source export:

```bash
./scripts/modal.sh setup
./scripts/modal.sh canary
```

Modal handles signup and credentials through its own flow. This canary checks
remote packaging and execution; it does not establish production bypass
closure. See [examples](examples/README.md) for setup and the separate,
explicitly confirmed persistent-deployment command. GPU research runs have
their own [recertification procedure](docs/MODAL_RECERTIFICATION.md).

## Development

```bash
python -m pip install -e '.[crypto,lockbox,onnx,rag,spiffe,torch,dev]'
python -m pytest -q
python scripts/bootstrap_build_env.py
python scripts/release_check.py
```

The release check covers lint, formatting, strict typing, tests, research
validation, Lean, tracked-source builds, archive verification, and clean-wheel
quickstarts. The deterministic suite does not download an embedding model.
The optional real-encoder test and bootstrap-history rules are documented in
[CONTRIBUTING.md](CONTRIBUTING.md) and the [installation guide](docs/INSTALLATION.md).

## Documentation and support

| Topic | Documentation |
|---|---|
| Request flow and product boundary | [Architecture](docs/ARCHITECTURE.md), [operator responsibilities](docs/OPERATOR_BOUNDARY.md) |
| APIs and compatibility | [Usage guide](docs/USAGE.md), [1.x stability contract](docs/API_STABILITY.md) |
| Security and maintenance | [Security policy](SECURITY.md), [engineering controls](docs/SECURITY_ENGINEERING.md), [fail-closed defaults](docs/FAIL_CLOSED_DEFAULTS.md) |
| Release custody | [Provenance](docs/PROVENANCE.md), [attestations](docs/RELEASE_ATTESTATION.md) |
| Direction and contribution | [Roadmap](ROADMAP.md), [contributing](CONTRIBUTING.md), [adoption guide](docs/LAUNCH_AND_ADOPTION_PLAN.md) |

Report suspected vulnerabilities through [SECURITY.md](SECURITY.md). For
integration inquiries, contact [Sekos AI](mailto:ryan@sekos.ai).

## License and patent notice

The Gate library, examples, tests, build and deployment scripts, and Lean proof
source use [Apache License 2.0](LICENSE). Authored papers, explanatory research
prose, figures, and designated results use CC BY 4.0 under the
[research path map](research/cdp/LICENSES.md).

A U.S. provisional patent application was filed before release for subject
matter related to portions of Schemen Gate. This notice adds no separate
restriction. Apache-2.0's patent license and termination terms apply within
their defined scope; CC BY 4.0 does not grant patent rights. See the
[licensing policy](docs/LICENSING_DECISION.md) for the adopted terms and scope.
