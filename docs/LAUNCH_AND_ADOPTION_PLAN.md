# AI PKI launch, adoption, and business plan

Audience: maintainers, contributors, technical evaluators, and organizations
considering Schemen Gate. This is a public adoption plan, not a promise of
features beyond the checked-in implementation and evidence.

## Category

**AI PKI** is the umbrella category: public-key infrastructure extended to
authenticate AI actors, bind signed and scoped authority to AI operations, and
produce independently verifiable evidence at execution boundaries. Schemen
Gate is an open implementation of the enforcement layer for that category.

Use **AI provenance PKI** when the emphasis is release identity, signed
authority, attestations, and receipts. Use **inference trust infrastructure**
when the emphasis is runtime enforcement, deployment, and integration. These
are descriptions of AI PKI, not separate categories.

AI PKI does not replace classical PKI. Its guarantees remain conditional on
the verifier's configured roots, CA and key hygiene, deployment integrity, and
complete Gate placement. Schemen Gate does not invent a CA or imply hardware
binding merely because it supports PKCS#12.

## The need in one sentence

AI PKI carries a classically authenticated identity beyond the service boundary
into a cryptographically verified authorization decision about which AI regime
and operation may execute.

## Why it matters

Classical access control often ends just before model execution, retrieval
release, attachment use, weight-shard release, or a state-changing operation.
The Gate makes that downstream boundary explicit and testable:

1. authenticate a subject through the organization's certificate policy;
2. verify a signed, scoped grant;
3. select the authorized regime and operation;
4. apply the Gate at the declared execution boundary; and
5. emit evidence that can be checked independently.

The security statement is conditional. The cryptographic path is only as
trustworthy as CA hygiene, key custody, deployment integrity, and complete Gate
placement. PKCS#12 support makes enterprise credentials easy to consume; it is
a credential container, not proof of hardware-backed key custody.

## What is useful and falsifiable

- **AuthN and AuthZ stay connected.** The verified certificate subject is the
  subject of the grant that selects downstream authority.
- **The decision reaches execution.** The unit of control can be an operation,
  retrieval release, model attachment, encrypted shard, or declared activation
  regime.
- **The core is inspectable.** Source, tests, formal statements, paper sources,
  generated PDFs, examples, and research receipts ship together.
- **Claims are separated.** Local algebra, cryptographic assumptions, measured
  behavior, and deployment assumptions are labeled independently.
- **Onboarding is small.** A local example proves core behavior; one CPU Modal
  canary proves packaging and remote execution without pretending to be a
  security theorem.

## Adoption wedge

Lead with a problem engineers can test:

> A valid API call should not automatically imply authority to execute every
> model path.

The first success is one protected operation, one independently pinned
authority, one scoped grant, and one passing denial test. Expand only after each
boundary has an owner, an adversarial test, and an observable receipt.

## Language discipline

- Lead with **AI PKI** as the category.
- Use **AI provenance PKI** for signed identity, authority, releases,
  attestations, and receipts.
- Use **inference trust infrastructure** for runtime enforcement, integrations,
  and operational deployment.
- Call this the launch, adoption, and business plan, not a generic acronym.
- Do not imply that PKCS#12 is hardware binding, that Gate replaces CA hygiene,
  or that a local theorem proves an end-to-end deployment secure.

## Audiences and first proofs

- **Security and identity teams:** bring an existing CA or pinned self-signed
  root; prove wrong signer, subject, scope, and expiry fail closed.
- **ML platform teams:** gate one operation or attachment without retraining;
  evaluate model-internal placement only where its assumptions fit.
- **Researchers:** rebuild the papers, inspect Lean dependencies, and replay one
  small experiment before a GPU matrix.
- **Maintainers:** install cleanly, run the quickstart and tests, and inspect the
  dependency surface.
- **Enterprise architects:** map certificate lifecycle, authority issuance,
  runtime placement, and evidence retention to existing controls.

## Release assets

- history-free repository, signed tag, protected `main`, and immutable release;
- wheel and source archive with checksums and provenance;
- five-minute local quickstart and fail-closed denial examples;
- PKCS#12 example with an independently pinned root;
- `./scripts/modal.sh setup` and `./scripts/modal.sh canary`;
- architecture and security-claim maps;
- papers, source, build instructions, Lean inventory, and research receipts;
- contribution, security-reporting, governance, and roadmap documents.

## Launch sequence

### T-14 to T-7: private proof

- Have two people who did not build the project follow the README from clean
  machines and record every point of confusion.
- Run denial cases first: wrong root, wrong subject, expiry, wrong regime,
  tampered payload, and missing Gate placement.
- Resolve licensing, patent, rights-holder, dependency, and dataset gates.
- Prepare a one-commit candidate with no remote until publication is explicitly
  authorized.

### T-6 to T-1: launch packet

- Freeze the exact commit and artifact hashes.
- Record a short terminal demo: quickstart, one denial, PKCS#12 signing, and CPU
  Modal canary.
- Prepare release notes, Substack, X, LinkedIn, social preview, and FAQ from one
  claim sheet.
- Run a private vulnerability-response drill.

### T0: source first

After explicit approval, publish source and packages, verify them from public
endpoints, and only then publish announcements. Every post points to the same
README quickstart and claim boundaries.

### T+1 to T+30: adoption, not impressions

- Answer installation issues within one business day during launch week.
- Turn repeated questions into FAQ entries and executable examples.
- Publish weekly deep dives on authority, denials, placement limits, and
  reproducibility.
- Invite integrations through the issue template; promise an adapter only when
  it has a maintainer and acceptance test.

## Channel copy

### Substack

Suggested title: **AI PKI: Authentication Should Reach the Model**

Outline: the API-to-execution gap; a concrete denial; the
certificate-to-grant-to-regime path; what cryptography establishes; what still
depends on CA hygiene and placement; the five-minute demo; an invitation to
falsify the claims.

Opening draft:

> AI systems need PKI that does not stop at the API boundary. The consequential
> part happens after it: selecting a model path, releasing context, loading an
> attachment, or changing state. Schemen Gate carries authenticated identity
> into a downstream decision that is cryptographic, scoped, and independently
> testable. That is AI PKI at the execution boundary.

### X thread

1. AI PKI should carry verified identity and scoped authority to the operation
   that actually executes. Schemen Gate carries AuthN into downstream AuthZ.
2. The path is small: verified certificate -> scoped signed grant -> authorized
   regime/operation -> Gate -> evidence.
3. PKCS#12 works today, including pinned self-signed roots. It packages a
   credential; it does not prove a TPM or Secure Enclave held the key.
4. The local claim is falsifiable: inactive declared coordinates are zero after
   the binary Gate. Broader claims require broader evidence.
5. The repository ships code, denial tests, papers, Lean sources, receipts, and
   a five-minute quickstart.
6. Run locally, then use one CPU Modal canary. No GPU is needed to understand
   the boundary.
7. Break a claim, improve an integration, or show where placement is incomplete.
8. Start here: `<public repository URL after authorization>`

### LinkedIn

> AI PKI is the missing trust layer between enterprise identity and AI
> execution. Authentication is not authorization, and authorization at an API
> gateway is not necessarily authorization at the model boundary. Schemen Gate
> connects those layers: an authenticated certificate subject receives a
> signed, scoped grant selecting the permitted regime and operation. The
> release includes the implementation, denial tests, papers, formal proof
> source, receipts, a PKCS#12 example, and a one-command CPU Modal canary. The
> guarantee is bounded by CA and key hygiene, correct Gate placement, and
> deployment integrity. We are publishing the mechanism so practitioners can
> inspect and falsify it.
> `<public repository URL after authorization>`

Never claim “unhackable,” unconditional security, or that a local theorem proves
AI safety. Use the repository's exact claim boundaries.

## Business around open AI PKI infrastructure

The free Gate must be useful without a sales conversation. Revenue can come
from reducing operational burden around it:

- enterprise CA, KMS/HSM, confidential-compute, and runtime integrations;
- managed authority lifecycle, revocation, policy distribution, and evidence;
- deployment hardening, compliance mappings, and architecture reviews;
- support SLAs, incident response, training, and implementation partnerships;
- hosted control-plane and observability products using the same open formats.

This preserves the flywheel: the open implementation becomes a credible
standard; operational depth becomes the paid value. Avoid proprietary forks of
the basic protocol or incompatible grant formats.

Two stories deserve first-class launch treatment:

- **Public-for-all is principled policy.** An explicit all-ones Gate proves a
  resource is open by authority decision; it is not the same as a missing,
  failed, or bypassed check.
- **Cargo Mode is bilateral authority.** Both parties accept a signed finite
  manifest. The open library proves the obligations it can establish locally;
  an operator integration can connect external completion to a mutually agreed
  evidence provider and emit a receipt bound to the exact result.

## Real adoption metrics

Track clean installs reaching a passing denial test, independent importing
repositories, repeat contributors, AI PKI integrations, independent
reproductions and adversarial findings, production evaluations, and support
questions eliminated by better docs.

Thirty-day target: five independent successful installs, two external
integration discussions, one independent reproduction, and one substantive
security review. Do not optimize stars before those exist.

## Launch authority gate

No post, repository, package, tag, DOI, demo endpoint, or press outreach goes
live until the rights holder approves the exact commit, license decision,
artifact hashes, release notes, and coordinated publication time.
