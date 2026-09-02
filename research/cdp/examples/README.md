# Examples

These examples deliberately separate the three boundaries used by the paper.

Run these commands from the Schemen Gate repository root; every path is
caller-directory explicit.

1. Prepare a small local environment:

   ```bash
   ./research/cdp/scripts/setup.sh examples
   ```

2. Run the algebraic gate example:

   ```bash
   research/cdp/.venv/bin/python research/cdp/examples/gate_numpy.py
   ```

   This demonstrates complete, disjoint masks and exact zeros at the declared
   tensor. It is not an end-to-end privacy or shared-attention result.

3. Authenticate and run one cheap Modal CPU canary:

   ```bash
   ./scripts/modal.sh setup
   ./scripts/modal.sh canary
   ```

   `setup` delegates signup and token creation to Modal's official browser flow.
   The helper never reads or prints credentials. `canary` uses no GPU and caps
   the app at one CPU container. See Modal's
   [account-setup guide](https://modal.com/docs/guide/modal-user-account-setup)
   for the underlying flow.

4. Optionally deploy the proxy-authenticated canary endpoint:

   ```bash
   ./scripts/modal.sh deploy-canary
   ```

   This is a persistent Modal deployment and therefore requires confirmation.
   It scales to zero when idle. Use the deployment URL shown by Modal and its
   authenticated client tooling; the endpoint requires Modal proxy auth.

The research GPU launchers live under `experiments/`. Always run a documented
`--smoke` or `--canary` path and inspect its persisted artifact before a larger
matrix. Several canonical launchers cap remote batches at three jobs.
