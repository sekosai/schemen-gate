# Dependency surface

The default `schemen-gate` install depends only on `numpy>=1.24`.

- `crypto`: `cryptography>=50.0` for AES-GCM, HKDF, EdDSA/ECDSA/RSA,
  X25519, and X.509.
- `lockbox`: `cryptography` plus `pyyaml>=6.0` for lockbox serialization,
  PKCS#12 credential loading, X.509 path verification, and authority signing.
- `torch`: `torch>=2.13` conversion helpers; no torch import occurs in the core path.
- `onnx`: `onnx>=1.14` model metadata integrations.
- `spiffe`: `spiffe>=0.3.0` Workload API identity retrieval.
- `rag`: `psycopg[binary]>=3.1` and `scikit-learn>=1.3` for optional stores and
architecture suggestions.
- `dev`: test, lint, type-check, coverage, and release-verification tools.
  `pytest-cov>=6.0` produces the CI coverage report. `pypdf>=6.0`
  is used only by the public-hygiene release gate to extract bounded text and
  metadata from tracked PDFs, and `twine==7.0.0` performs the standard package
  metadata and long-description check. Neither is imported by the Gate library.

The PKCS#12 adapter is portable software key loading. It does not call an OS
keystore or assert hardware residency. Platform-native TPM, enclave, HSM, CNG,
or cloud-KMS providers remain optional `KeyProvider` integrations.

All dependencies are open-source replaceable adapters. No private package is
required. Execution frameworks, cloud SDKs, FastAPI, model weights, and deployment
systems are intentionally excluded.

## Licenses of the dependency surface

Every runtime dependency is permissively licensed: `numpy` (BSD-3-Clause and
compatible), `cryptography` (Apache-2.0 or BSD-3-Clause), `pyyaml` (MIT),
`cffi` (MIT), `pycparser` (BSD-3-Clause), `torch` (BSD-3-Clause family with
Apache-2.0 and MIT components), `onnx` (Apache-2.0), `spiffe` (Apache-2.0),
`scikit-learn` (BSD-3-Clause), and `grpcio` (Apache-2.0). The optional `rag`
extra depends on `psycopg` and `psycopg-binary`, which are LGPL-3.0; they are
imported at runtime and are not vendored or modified, and nothing in this
repository redistributes them. No dependency in the default install or in any
extra carries a copyleft obligation on this library's own code. Development
tooling licenses (pytest, ruff, mypy, twine, pypdf) do not ship with the
package.

The `cryptography` floor is 50.0 because Gate uses its timezone-aware
certificate, CRL, and OCSP validity properties. Keeping that floor explicit
keeps the tested X.509 behavior on one maintained API generation.
