#!/usr/bin/env bash
set -euo pipefail

CDP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CDP_AUDIT_OUTPUT="$(mktemp)"
trap 'rm -f "${CDP_AUDIT_OUTPUT}"' EXIT

cd "${CDP_REPO_ROOT}"
lake env lean scripts/lean_claim_audit.lean 2>&1 | tee "${CDP_AUDIT_OUTPUT}"

CDP_AUDITED_COUNT="$(grep -c "depends on axioms" "${CDP_AUDIT_OUTPUT}")"
if [[ "${CDP_AUDITED_COUNT}" -ne 9 ]]; then
  echo "Expected 9 audited headline theorems; saw ${CDP_AUDITED_COUNT}." >&2
  exit 1
fi

if grep -Eq \
  'Recovers|prf_brute_force_optimal|IsDistributionMatched|camouflage_indistinguishable|gradient_probing_hard|sorryAx|Lean\.ofReduceBool' \
  "${CDP_AUDIT_OUTPUT}"; then
  echo "A project-specific or unsafe axiom reached a headline theorem." >&2
  exit 1
fi

echo "Headline Lean theorem axiom audit passed."
