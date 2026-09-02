# Schemen Gate

Schemen Gate is an Apache-2.0 AI PKI library for carrying verifier-trusted
identity and signed, scoped authority to an AI execution boundary. It provides
binary Gates, cryptographic capabilities, lockboxes, Cargo manifests, and
release-bound evidence without bundling a model server or control plane.

Start with the
[certificate-to-Gate quickstart](https://github.com/sekosai/schemen-gate/blob/v1.0.2/examples/ai_pki_quickstart.py),
then read the
[architecture](https://github.com/sekosai/schemen-gate/blob/v1.0.2/docs/ARCHITECTURE.md),
[X.509 profile](https://github.com/sekosai/schemen-gate/blob/v1.0.2/docs/X509_PROFILE.md),
and
[production deployment contract](https://github.com/sekosai/schemen-gate/blob/v1.0.2/docs/PRODUCTION_DEPLOYMENT.md).

```bash
python -m pip install 'schemen-gate[lockbox]==1.0.2'
```

The complete Git repository also contains the Cryptographic Dimension
Partitioning papers, Lean proofs, experiment launchers, and research receipts.
Those research files are intentionally not included in the Python wheel or
source distribution and use their
[path-specific license map](https://github.com/sekosai/schemen-gate/blob/v1.0.2/research/cdp/LICENSES.md).
Browse the
[publication bundle](https://github.com/sekosai/schemen-gate/tree/v1.0.2/research/cdp)
at the exact source revision used by your installation.

Report security issues through
[GitHub private vulnerability reporting](https://github.com/sekosai/schemen-gate/security/advisories/new)
or email [ryan@sekos.ai](mailto:ryan@sekos.ai).
