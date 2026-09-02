# Modal Re-certification Runbook

This runbook re-certifies the shipped Schemen Gate 1.0.2 Modal evidence without
changing the Gate method. It binds every run to one clean source commit, one
sealed execution plan, current provider pricing, an explicit campaign approval
ceiling, construction-specific acceptance checks, and externally held result
artifacts.

The default campaign is **compatibility**. It reruns one current-library
analogue of every canonical Gate-backed Modal evidence family. The broader
**paper-matrix** campaign is optional: it repeats the larger numerical matrices
and costs materially more. Neither campaign upgrades an experiment's stated
claim boundary.

## Cost decision

The checked-in price snapshot is dated 2026-08-30 and lives in
[`research/cdp/experiments/modal-recertification.json`](../research/cdp/experiments/modal-recertification.json).
Before any later campaign, compare that snapshot with Modal's current
[pricing](https://modal.com/pricing), [GPU](https://modal.com/docs/guide/gpu),
[billing](https://modal.com/docs/guide/billing), and
[budget](https://modal.com/docs/guide/budgets) documentation and regenerate the
plan if a rate changed.

The rates used for this campaign are:

| Resource | Per second | Per hour |
|---|---:|---:|
| A100 40 GB request (`gpu="A100"`) | $0.000583 | $2.0988 |
| T4 | $0.000164 | $0.5904 |
| Physical CPU core | $0.0000131 | $0.04716 |
| Memory GiB | $0.00000222 | $0.007992 |

Modal documents that an `A100` request means a 40 GB A100 and may be upgraded
to an 80 GB A100 without changing the requested GPU's price. CPU and memory are
additive. Billing is per second with no minimum usage-time increment.

The estimates are gross usage before credits. They do not assume that a
Starter or promotional credit remains available. Image construction, cold
downloads, CPU and memory above the historical proxy, retries, region
multipliers, non-preemptible execution, storage, and failed jobs can increase
actual usage.

| Campaign | Historical accelerator proxy, canary + full | Expected gross usage | Campaign approval ceiling |
|---|---:|---:|---:|
| Compatibility (recommended) | $2.10 | $2.50-$4.00 | $8.00 |
| Paper matrix (optional) | $6.34 | $7.00-$11.00 | $15.00 |

The compatibility campaign uses $5 as an operator alert and $8 as its explicit
approval ceiling. The historical proxy is arithmetic over recorded accelerator seconds: the
common canary stage is about $0.155; the compatibility full stage is about
$1.941; and the paper-matrix full stage is about $6.186. The wider expected
ranges and ceilings preserve room for non-accelerator resources and normal
variance. The ceiling is an approval and operational stop threshold, not a
spending target. It is not a provider-enforced hard cap on a Starter account.

## Provider-budget review and spending controls

Use a dedicated Modal Environment such as `schemen-gate-recert-1-0-2` so its
Apps, Volumes, and results are isolated from unrelated work. Environment
isolation is required even when the account plan does not offer an
Environment-level budget.

Before execution, inspect the provider's current billing-cycle usage, remaining
credits, Workspace usage limit, and available budget controls:

- On Team or Enterprise, set an $8 compatibility or $15 paper-matrix budget on
  the dedicated Environment when possible. That creates a provider-enforced
  compute cap for the Environment.
- On Starter, Environment budgets are unavailable and the Workspace usage
  limit is global. Do not lower a shared Workspace limit if that could stop or
  constrain unrelated workloads, and do not claim that this campaign has a
  provider-enforced hard cap.
- If an existing provider cap is lower than the campaign ceiling, keep the
  lower cap and accept that the campaign may stop early. Never raise or lower a
  shared limit implicitly.

Modal budgets apply to a billing cycle. An Environment budget covers compute,
not every Workspace-level charge such as storage or reservations. The local
runner cannot prove a dashboard setting or enforce a cumulative dollar limit.
It rejects an approval above the checked-in ceiling, invokes launchers
sequentially, stops on the first failure, and never retries automatically. When
no campaign-specific provider hard cap is available, observe gross usage
before the canary, between stages, and after the full campaign.
Provider billing data may arrive after a job completes, so those observations
are monitoring evidence, not a real-time hard stop.

## Exact campaign inventory

The CPU canary exercises the certificate-to-grant-to-Gate path and its
wrong-root and wrong-recipient denials. Seven research launchers then cover the
canonical, current-library Modal evidence surface.

| Job | Resource | Canary arguments | Compatibility arguments | Paper-matrix arguments |
|---|---|---|---|---|
| AI-PKI CPU canary | CPU | none | not repeated | not repeated |
| Dense FFN cotenancy | A100 | `--smoke` | `--r-values 8 --seeds 42` | `--r-values 1,2,4,8,16 --seeds 42,123,256` |
| Private Transformer lanes | A100 | `--smoke` | `--designs adapter,expert --seeds 42` | `--designs adapter,expert --seeds 42,123,256` |
| Public-gate adaptation factorial | A100 | `--smoke` | `--seeds 42` | `--seeds 42,123,256,512,1024` |
| Cargo Transformer authorization | T4 | no separate smoke | full protocol | full protocol |
| execution orthogonal superposition | A100 | `--smoke` | `--ratios 8,128` | `--ratios 8,128` |
| Generative intermediate FFN | A100 | `--seeds 0 --r 8 --smoke` | `--seeds 11 --r 8` | `--seeds 11 --r 8` |
| DistilBERT service consolidation | T4 | `--smoke` | full protocol | full protocol |

The canary stage is seven top-level Modal commands: one CPU command, five A100
commands, and one T4 command. Cargo has no reduced `--smoke` path, so it runs
only in the full stage. The compatibility full stage is seven top-level
commands containing seven A100 and two T4 remote invocations. The paper-matrix
full stage contains 29 A100 and two T4 remote invocations. Launchers remain
bounded to at most three concurrent remote containers internally; the
orchestrator invokes launchers sequentially.

Every step also has a sealed client wall-clock limit. Compatibility limits are
10 minutes for dense and private lanes, 30 minutes for public factorial and
orthogonal superposition, 50 minutes for generative intermediate, and 5
minutes for Cargo and service consolidation; reduced canaries are limited to
5-15 minutes. The corresponding remote functions use bounded 5-50 minute
timeouts instead of multi-hour defaults. If a client limit expires or the
client is interrupted, the runner marks completion ambiguous, attempts an
explicit `modal app stop` in the dedicated Environment, records whether that
stop was confirmed, and never retries automatically.

The following eight Modal entrypoints are intentionally excluded:

- `modal_attention_lane_stress.py`
- `modal_fused_multiplexing_benchmark.py`
- `modal_generative_ffn.py`
- `modal_generative_full.py`
- `modal_kv_cache_pollution.py`
- `modal_matched_adaptation.py`
- `modal_matched_deposition.py`
- `modal_vllm_kv_reuse.py`

They are exploratory, negative, superseded, or follow-up studies and do not all
implement the complete current-library custody contract used here: clean-HEAD
source export, dependency-bundle recording, launcher digest recording, and
remote installed-byte verification. Their presence in the repository is not a
claim that they re-certify version 1.0.2. `modal_schemen_image.py` is a shared
image/provenance helper, not a runnable evidence job.

## What the runner seals

[`scripts/modal_recertify.py`](../scripts/modal_recertify.py) is the only
campaign orchestrator. Planning does not import a Modal launcher or allocate a
remote resource. A plan records and hashes:

- release version, exact clean Git commit, release contract, release manifest,
  and campaign configuration;
- each launcher path, launcher digest, exact arguments, resource class,
  app identity, historical runtime proxy, maximum wall time, invocation count,
  and expected result shape;
- the pricing snapshot, expected range, approval ceiling, and campaign stage;
  and
- a canonical SHA-256 plan seal.

Execution rejects a dirty or different checkout, a changed plan, the wrong
release, an unverified Modal token, an evidence directory inside the
repository, and an approval that exceeds the configured ceiling. The runner
passes fixed argument arrays to Modal; it does not evaluate a shell command,
detach work, retry a failed job, or run launchers concurrently.

## Staged procedure

Run every command from the repository root. Replace the example external paths
with absolute paths outside the Git checkout. Do not put a token, secret, or
credential in a command, plan, ledger, or artifact path.

### 1. Verify the exact release candidate

The candidate must be one clean commit. The runner exports that exact local
source, records its SHA and source-tree digest, and requires the remote
installed bytes to match. The identical commit may be pushed or published
later; remote Git reachability is not an execution prerequisite and is not
evidence that the installed remote bytes matched.

```bash
git status --short
python3 scripts/release_check.py
```

Stop if the status is not empty or the release check does not pass.

### 2. Create the isolated provider boundary

```bash
./scripts/modal.sh setup
./scripts/modal.sh status
.venv-modal/bin/modal environment create schemen-gate-recert-1-0-2
```

If the Environment already exists, inspect it rather than creating a second
one. Review the provider budget state as described above. Configure a dedicated
Environment budget when the plan supports it; otherwise record that no
campaign-specific provider hard cap is available. This is provider state; the
repository script does not pretend to set, verify, or replace it.

### 3. Seal and inspect the plan

The compatibility campaign is the release re-certification default:

```bash
python3 scripts/modal_recertify.py plan \
  --campaign compatibility \
  --output /absolute/external/path/schemen-gate-1.0.2-compatibility-plan.json

python3 scripts/modal_recertify.py check \
  --plan /absolute/external/path/schemen-gate-1.0.2-compatibility-plan.json
```

For the optional expanded campaign, use `--campaign paper-matrix` and the $15
ceiling. Read the printed commit, plan SHA-256, job counts, expected gross
range, and approval ceiling before approving a run. A source or configuration
change invalidates the plan; generate a new one rather than editing the JSON.

### 4. Run and inspect canaries

Copy the exact printed plan seal into `--approve-plan-sha256`. The following
example uses the compatibility campaign's $8 approval ceiling:

```bash
python3 scripts/modal_recertify.py execute \
  --plan /absolute/external/path/schemen-gate-1.0.2-compatibility-plan.json \
  --stage canary \
  --modal-bin "$PWD/.venv-modal/bin/modal" \
  --environment schemen-gate-recert-1-0-2 \
  --evidence-root /absolute/external/path/schemen-gate-1.0.2-recertification \
  --approve-plan-sha256 PLAN_SHA256_FROM_PLAN_COMMAND \
  --approve-max-usd 8
```

The runner stops on the first failed process, ambiguous artifact, provenance
mismatch, or construction-specific acceptance failure. Inspect the reported
canary ledger and every retained artifact before authorizing the full stage. A
smoke result proves installation, source transport, denial ordering, result
transport, and the construction's reduced control flow; it is not a utility or
production-security result.

### 5. Run the bounded full campaign

The full stage requires the successful canary ledger from the same plan and
commit:

```bash
python3 scripts/modal_recertify.py execute \
  --plan /absolute/external/path/schemen-gate-1.0.2-compatibility-plan.json \
  --stage full \
  --modal-bin "$PWD/.venv-modal/bin/modal" \
  --environment schemen-gate-recert-1-0-2 \
  --evidence-root /absolute/external/path/schemen-gate-1.0.2-recertification \
  --canary-ledger /absolute/external/path/to/reported-canary-ledger.json \
  --approve-plan-sha256 PLAN_SHA256_FROM_PLAN_COMMAND \
  --approve-max-usd 8
```

Do not automatically retry an interrupted or ambiguous billed call. Preserve
its ledger, inspect Modal's call state and locally recovered artifacts, then
make a new explicit retry decision. A second full run creates new evidence; it
must never overwrite or silently stand in for the first attempt.

## Artifact custody and confirmation

Each research launcher writes one timestamped combined JSON artifact under
`research/cdp/experiments/results/` only after its remote invocations return.
The orchestrator snapshots that directory before every job, requires exactly
one new artifact with the configured prefix, validates it, copies it to a new
owner-only temporary file in the external evidence root, verifies the copy,
atomically publishes that destination, removes the source, and then requires
the Git checkout to be clean again. This avoids one job's untracked result
making the next canonical launch fail its clean-HEAD check.

Validation is not based on process exit status alone. The runner checks the
experiment identity, expected record count, local dependency bundle, remote
installed-byte verification, source commit, launcher digest, wrong-authority
rejections, zero unauthorized model calls, and the construction-specific
acceptance fields. The external ledger is written atomically after every job
and records artifact names and SHA-256 digests without copying Modal tokens,
command output, or environment-variable values.

After a successful full campaign:

1. Confirm every planned job is marked successful and every expected artifact
   has a digest in the ledger.
2. Re-run `python3 scripts/modal_recertify.py check --plan ...` from the same
   clean commit.
3. Compare measured resource usage with the gross estimate and campaign
   ceiling; state whether a provider-enforced hard cap existed, and record any
   variance without rewriting the sealed plan.
4. Review the new results against
   [`research/cdp/docs/CLAIM_BOUNDARIES.md`](../research/cdp/docs/CLAIM_BOUNDARIES.md)
   and the existing claim-to-artifact inventory. A passing re-certification
   confirms the existing construction at the recorded environment; it does not
   broaden the paper.
5. Import reviewed artifacts into a later repository commit only through a
   separate evidence-admission change. Never mix execution-produced files into
   the source commit they claim to test.

## Failure meanings

- **Source or plan rejection:** no evidence run should begin; correct custody
  and produce a new sealed plan.
- **Provider cap reached, or observed usage approaches the campaign ceiling:**
  the campaign is incomplete, not failed evidence and not a reason to raise the
  limit automatically.
- **Canary failure:** do not run the full stage.
- **Acceptance-field failure:** preserve the artifact as failed evidence; do
  not relabel it canonical.
- **Missing or ambiguous result:** preserve the ledger and inspect any
  provider-side call identifier or state shown in the dashboard; do not infer
  success from logs or retry without a new decision.
- **Numerical result differs while controls pass:** report the observed result
  and reassess the corresponding empirical claim. Do not tune the threshold or
  method after seeing the outcome.
