# Credential broker

An optional service for encrypted provider-credential custody and constrained
HTTP requests. Agents receive permission to use a connection; they cannot
retrieve its credential or select another tenant through request fields.

This directory is an independently installed Apache-2.0 package,
`schemen-credential-broker`, version **0.3.1**, requiring Python **3.11+** on
POSIX systems. It is excluded from the `schemen-gate` wheel and source
distribution. The base connection broker does not require Gate or a private
service. The optional `gate` extra enables
[one-off Google Calendar calls](CALENDAR_ONE_OFF.md): exact operation AAD,
one-use authority, per-call encrypted custody, and signed destruction receipts.
Reusable connections retain their tenant/subject/connection/route permission model.

The optional `delegation` extra adds an explicit
[signed approval and callback flow](DELEGATED_AUTHORIZATION.md): Ed25519 request,
delegation, approval, callback, client authentication, DPoP redemption and result
bindings; encrypted durable state; and a bridge to one-off Calendar custody.
It is an opt-in application module with a runnable local demo, not enabled by
the existing `serve` command.

## Install from this checkout

From the repository root:

```sh
cd services/credential-broker
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-deps .
```

This source installation does not imply that a matching package-index release
exists. The dependency lock pins published wheel hashes, excludes source builds, and
includes the test runner. Install the
service in its own environment rather than adding HTTP dependencies to a core
library environment.

## Initialize and serve

```sh
.venv/bin/schemen-broker init --secrets-dir local-secrets
.venv/bin/schemen-broker serve \
  --config local-secrets/config.json \
  --key-file local-secrets/master.key \
  --database local-data/vault.sqlite \
  --port 8787
```

Initialization creates owner-only key, token and configuration files. It
refuses to overwrite an existing directory and never prints secret values.
The initial tenant is `local`, with separate `admin` and `agent` subjects.
Configuration contains hashes of high-entropy broker tokens. The default
provider policy permits only `GET https://api.github.com/user`.

The service binds to `127.0.0.1`; its CLI requires HTTPS provider destinations.
`GET /healthz` returns process health. Key custody, host isolation, TLS ingress,
and network policy remain operator responsibilities. See [SECURITY.md](SECURITY.md).

## Store, use, rotate and revoke

Place a provider credential in an owner-only file using your normal secret
management mechanism. Keep that file and the administrator token inaccessible
to the agent. Credentials must contain 16–8192 printable, non-whitespace ASCII
characters. No live provider credential is included in this repository.

```sh
.venv/bin/schemen-broker store \
  --token-file local-secrets/admin.token \
  --connection github-me --provider github --subject agent \
  --credential-file /path/to/protected-provider-token

.venv/bin/schemen-broker request \
  --token-file local-secrets/agent.token \
  --connection github-me --method GET --path /user

.venv/bin/schemen-broker revoke \
  --token-file local-secrets/admin.token --connection github-me
```

Repeat `store` for the same connection with a replacement credential to rotate
it. Optional `--expires-at` sets a Unix expiry timestamp. Permitted write routes
accept a JSON body from `--json-file`. Secrets are read from protected files,
not command-line arguments. Provider terminal control characters are escaped.

## HTTP contract

Every HTTP request except exactly `GET /healthz` requires
`Authorization: Bearer <broker-token>`. Trailing-slash redirects and WebSocket
upgrades are disabled.
Tenant and subject come from the server's identity map, never the request body.

| Endpoint | Permission | Behavior |
| --- | --- | --- |
| `PUT /v1/connections/{id}` | Tenant administrator | Provision or replace a credential and subject ACL |
| `DELETE /v1/connections/{id}` | Tenant administrator | Revoke a connection |
| `POST /v1/connections/{id}/request` | Allowed connection subject | Execute a permitted provider request |
| `GET /healthz` | Local caller | Process health |

Provisioning accepts `provider`, `subjects`, `credential` and optional
`expires_at`. There is no credential-retrieval endpoint.

Example broker request:

```json
{"method":"GET","path":"/user","query":{}}
```

An optional `json` member supplies the provider body. Caller-supplied destination
URLs, authentication headers, cookies, proxy settings, tenant fields and other
unknown top-level fields are refused. Paths must be canonical; percent-encoded
resource paths are intentionally unsupported. Requests are limited to 64 KiB;
provider responses to 1 MiB. Request bodies must have one JSON content type
and no content encoding. Redirects and compressed provider responses are
refused. Upstream cookies and headers are not forwarded. Common direct/encoded
credential echoes cause an error instead of returning the body. Provider
network exchanges have an absolute 20-second deadline, including connection,
headers and body, plus 10-second inactivity timeouts. The broker holds admission
until response delivery completes, with a separate 10-second send deadline.
CLI failures never print exception details or tracebacks that could contain secrets.

## Provider configuration

Only the operator can register providers in the owner-only `config.json`.
Tenant administrators choose among those entries. Example provider entry:

```json
{
  "origin":"https://api.example.com",
  "auth_header":"X-API-Key",
  "auth_prefix":"",
  "routes":[
    {"method":"GET","path":"/v1/items/","prefix":true},
    {"method":"PATCH","path":"/v1/items/item-7"}
  ]
}
```

Supported credential formats are `Authorization: Bearer ...` and `X-API-Key`.
Prefix routes must end with `/` and authorize every descendant path. Define
different provider entries for different connection permission sets.

Each encrypted connection binds the **complete provider configuration
fingerprint**, including destination, authentication format and routes. After
a configuration change and restart, existing connections return HTTP 409 until
explicitly reprovisioned. Records from an earlier unbound implementation also
require reprovisioning. Identity configuration changes take effect on restart.

## Key custody and recovery

The `KeyProvider` interface must return 32 bytes. `FileKeyProvider` enforces
owner-only regular-file access. The service authenticates a database key-check
record before accepting an existing vault and before each transaction, and
never silently adopts a changed key file while running.

For offline master-key rotation, stop workers, prepare a new owner-only file
containing 32 random bytes, then run:

```sh
.venv/bin/schemen-broker rekey \
  --database local-data/vault.sqlite \
  --old-key-file local-secrets/master.key \
  --new-key-file /path/to/new-master.key
```

Restart with the new key file. Records, key-check and rotation audit change in
one transaction; corruption causes rollback. Stale workers fail closed. Retain
old keys according to the retention policy for old encrypted backups. Use
SQLite's backup API for consistent live backups and keep key backups separate.
Losing the relevant master key makes the encrypted credentials unrecoverable.

## Development and verification

```sh
python -m pip install -r requirements.lock -r requirements-gate.lock
python -m pip install -e ".[test,gate]"
python -m pytest -q
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m mypy src/credential_broker
python -m build --no-isolation
python scripts/verify_wheel.py
```

Install the repository's lint/build tooling separately when running the last
five commands locally. Dedicated CI runs the service on Python 3.11, 3.12 and
3.14, independently of core tests and releases. Tests use synthetic credentials,
real local HTTP sockets, an untrusted TLS fixture, and temporary owner-only
files. They require permission to bind loopback sockets.

The [security contract](SECURITY.md) maps claims to adversarial tests. The
[hardening review](HARDENING.md) records reproduced weaknesses and their fixes. Test
results are generated by CI for the exact commit; private review transcripts,
machine-specific receipts, key files, and built wheels are not tracked here.
See the root [contribution guide](../../CONTRIBUTING.md) and
[private vulnerability-reporting policy](../../SECURITY.md).

For resource operators, see the [resource-owner adoption guide](RESOURCE_OWNER_GUIDE.md)
and [Google Calendar live acceptance procedure](GOOGLE_CALENDAR_ACCEPTANCE.md).

[Display kit: swimlanes, delegation and resource boundaries](docs/display/README.md).

![Authenticated delegated authorization swimlanes](docs/display/01-authenticated-swimlanes.png)
