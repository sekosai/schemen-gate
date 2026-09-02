# API stability contract

Schemen Gate follows Semantic Versioning for the installed Python package. The
version identifies software behavior; authenticated token, lockbox, receipt,
and release-identity schemas retain their own explicit schema or format
versions.

## Public Python API

Every name exported by `schemen_gate.__all__` is a supported public API in the
1.x line. Underscore-prefixed modules and names are implementation details even
when tests import them to exercise a boundary directly.

For the remainder of 1.x, the project will not silently:

- remove or rename a public export;
- make a required argument incompatible with a previously valid call;
- change an authenticated field's meaning while retaining its schema version;
- widen authority, downgrade a denial, or make a fail-closed check optional;
  or
- change a serialized contract without an explicit version and migration path.

New optional arguments and new exports may be added in a minor release.
Ordinary removals require deprecation in a prior minor release and a major
version. A security correction may reject input that was ambiguous, malformed,
non-finite, out of bounds, or previously admitted only because of a fail-open
bug; that tightening is not treated as supported-input breakage.

## Optional capabilities

An optional public API has the same stability contract once its documented
extra is installed. Importing the base package does not install or initialize
those dependencies. Missing extras fail at the point the optional capability is
used; they do not change the core `GateMask` behavior.

The dependency and installation map is in [`DEPENDENCIES.md`](DEPENDENCIES.md).
Platform-native key custody remains an adapter boundary: the included PKCS#12
provider is portable software-key loading and does not claim hardware-backed
residence.

## Schemas and stored artifacts

Semantic Versioning does not replace protocol versioning. Callers should bind
and verify the Gate release identity and the contract's own schema/version
fields. A reader may support more than one historical schema explicitly, but it
must not guess a schema from shape or silently reinterpret unknown fields.

Historical paper receipts remain immutable evidence of the environment that
created them. A newer library release does not rewrite those receipts or imply
that every old experimental artifact is a supported runtime input.

## Repository-only surfaces

Research runners, paper build files, release-maintainer scripts, benchmarks,
and checked-in result records are reviewable and versioned with the Git tag,
but they are not installed Python APIs. Their command contracts are documented
at their entry points. Changes to an empirical method require a new result
artifact rather than rewriting the retained result.

No feature is implicitly "experimental" merely because it is less commonly
used. If a future provisional API is added, it must be placed in an explicitly
named experimental namespace, labeled in its documentation, and excluded from
the stable export list until promoted.
