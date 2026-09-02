# Examples

Run every command below from the repository root. The examples are deliberately
small: each names the boundary it exercises, exits nonzero when an assertion
fails, and avoids turning a toy result into a broader security claim.

## Local examples

| Example | Install | Run | What it demonstrates | What it does not claim |
|---|---|---|---|---|
| `quickstart.py` | `python -m pip install -e .` | `python examples/quickstart.py` | Exact binary masking of one NumPy activation | Identity, authorization, or a closed serving path |
| `ai_pki_quickstart.py` | `python -m pip install -e '.[lockbox]'` | `python examples/ai_pki_quickstart.py` | Certificate verification, signed grant resolution, Regime selection, Gate application, and wrong-root/wrong-recipient denial | Hardware-backed key custody or an ungated-path-free deployment |
| `pkcs12_identity.py` | `python -m pip install -e '.[lockbox]'` | `python examples/pkcs12_identity.py` | Portable PKCS#12 loading and signing to an independently pinned root | TPM, enclave, HSM, or other non-exportable key residence |
| `cotrained_shard_lockbox/demo.py` | `python -m pip install -e '.[lockbox]'` | `python examples/cotrained_shard_lockbox/demo.py` | A bounded co-training and encrypted-shard fixture with positive and negative assertions | Transformer behavior, arbitrary knowledge privacy, or a production runtime |
| `benchmark_vector_bridge.py` | `python -m pip install -e .` | `python examples/benchmark_vector_bridge.py --dimension 2048` | A local microbenchmark comparing implicit and explicit identity projections | A stable cross-machine benchmark or security evidence |

The benchmark prints machine-local timing and allocation measurements as JSON.
Record the command, hardware, Python version, NumPy version, and exact Gate
commit before comparing runs; its output is not part of the paper evidence.

## Modal CPU canary

The repository helper installs an exact Modal CLI version in `.venv-modal`,
delegates signup and token storage to Modal's own browser flow, and then runs
the certificate-to-Gate example remotely on CPU:

```bash
./scripts/modal.sh setup
./scripts/modal.sh canary
```

The canary admits only an exact tracked source export from a clean Git commit
and checks the stamped Gate source identity remotely. It does not use a GPU and
does not establish a production deployment. `deploy-canary` is a separate,
explicitly confirmed command because it creates a persistent endpoint.

## Research examples

The paper-specific algebra and callback examples live in
[`research/cdp/examples/`](../research/cdp/examples/README.md). Experiment
launchers and historical result custody are documented separately under
[`research/cdp/experiments/`](../research/cdp/experiments/README.md).
