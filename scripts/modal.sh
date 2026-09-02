#!/usr/bin/env bash
set -euo pipefail

GATE_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_MODAL_COMMAND="${1:-help}"
GATE_PYTHON_BIN="${SCHEMEN_GATE_PYTHON:-python3}"
GATE_MODAL_VENV_PATH="${SCHEMEN_GATE_MODAL_VENV:-${GATE_REPO_ROOT}/.venv-modal}"
GATE_MODAL_VERSION="1.5.4"
GATE_MODAL_BIN="${GATE_MODAL_VENV_PATH}/bin/modal"
GATE_MODAL_PYTHON="${GATE_MODAL_VENV_PATH}/bin/python"
if (( $# > 0 )); then
  shift
fi

install_modal() {
  if [[ ! -x "${GATE_MODAL_PYTHON}" ]]; then
    if ! command -v "${GATE_PYTHON_BIN}" >/dev/null 2>&1; then
      echo "Python was not found: ${GATE_PYTHON_BIN}" >&2
      exit 1
    fi
    "${GATE_PYTHON_BIN}" -m venv "${GATE_MODAL_VENV_PATH}"
  fi
  GATE_INSTALLED_MODAL_VERSION="$(
    "${GATE_MODAL_PYTHON}" -c \
      'from importlib.metadata import version; print(version("modal"))' \
      2>/dev/null || true
  )"
  if [[ "${GATE_INSTALLED_MODAL_VERSION}" != "${GATE_MODAL_VERSION}" ]]; then
    "${GATE_MODAL_PYTHON}" -m pip --disable-pip-version-check install \
      "modal==${GATE_MODAL_VERSION}"
  fi
  GATE_INSTALLED_MODAL_VERSION="$(
    "${GATE_MODAL_PYTHON}" -c \
      'from importlib.metadata import version; print(version("modal"))'
  )"
  if [[ "${GATE_INSTALLED_MODAL_VERSION}" != "${GATE_MODAL_VERSION}" ]]; then
    echo "Expected Modal ${GATE_MODAL_VERSION}, found ${GATE_INSTALLED_MODAL_VERSION}." >&2
    exit 1
  fi
  if [[ ! -x "${GATE_MODAL_BIN}" ]]; then
    echo "Pinned Modal installation did not provide its CLI." >&2
    exit 1
  fi
}

require_auth() {
  install_modal
  if ! "${GATE_MODAL_BIN}" token info >/dev/null 2>&1; then
    echo "No verified Modal token was found; opening Modal's official setup flow."
    "${GATE_MODAL_BIN}" setup
  fi
  if ! "${GATE_MODAL_BIN}" token info >/dev/null 2>&1; then
    echo "Modal setup completed but authentication could not be verified." >&2
    exit 1
  fi
  echo "Modal authentication verified."
}

case "${GATE_MODAL_COMMAND}" in
  setup)
    require_auth
    ;;
  status)
    install_modal
    "${GATE_MODAL_BIN}" --version
    if "${GATE_MODAL_BIN}" token info >/dev/null 2>&1; then
      echo "Modal authentication verified."
    else
      echo "No verified Modal authentication found." >&2
      exit 1
    fi
    ;;
  recert-plan)
    cd "${GATE_REPO_ROOT}"
    exec "${GATE_PYTHON_BIN}" scripts/modal_recertify.py plan "$@"
    ;;
  recert-check)
    cd "${GATE_REPO_ROOT}"
    exec "${GATE_PYTHON_BIN}" scripts/modal_recertify.py check "$@"
    ;;
  recert-execute)
    if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
      cd "${GATE_REPO_ROOT}"
      exec "${GATE_PYTHON_BIN}" scripts/modal_recertify.py execute "$@"
    fi
    require_auth
    cd "${GATE_REPO_ROOT}"
    exec "${GATE_MODAL_PYTHON}" scripts/modal_recertify.py execute \
      --modal-bin "${GATE_MODAL_BIN}" "$@"
    ;;
  canary)
    require_auth
    cd "${GATE_REPO_ROOT}"
    exec "${GATE_MODAL_BIN}" run examples/modal_quickstart.py
    ;;
  deploy-canary)
    require_auth
    echo "This creates a persistent, proxy-authenticated CPU web endpoint."
    read -r -p "Deploy schemen-gate-quickstart? [y/N] " GATE_MODAL_CONFIRMATION
    if [[ ! "${GATE_MODAL_CONFIRMATION}" =~ ^[Yy]$ ]]; then
      echo "Deployment cancelled."
      exit 0
    fi
    cd "${GATE_REPO_ROOT}"
    exec "${GATE_MODAL_BIN}" deploy examples/modal_quickstart.py
    ;;
  help|-h|--help)
    cat <<'EOF'
Usage: ./scripts/modal.sh COMMAND

Commands:
  setup          Install the pinned CLI and run Modal's browser signup/token flow
  status         Show CLI and active-token status without printing token secrets
  canary         Run one scale-to-zero CPU Gate canary (no GPU)
  deploy-canary  Confirm, then deploy a proxy-authenticated CPU Gate endpoint
  recert-plan    Seal a local-only campaign plan; never contacts Modal
  recert-check   Check a sealed plan against the current clean commit
  recert-execute Execute one explicitly approved, budget-declared stage

The script enforces Modal 1.5.4 and never asks for, prints, or stores token
values itself. Modal's CLI stores credentials in its standard user
configuration outside this repository.

Run `./scripts/modal.sh recert-plan --help` for the planning interface and
`./scripts/modal.sh recert-execute --help` for the exact approval, provider
budget, external evidence, and canary-before-full requirements.
EOF
    ;;
  *)
    echo "Unknown command: ${GATE_MODAL_COMMAND}" >&2
    echo "Run ./scripts/modal.sh help" >&2
    exit 2
    ;;
esac
