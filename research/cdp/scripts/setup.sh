#!/usr/bin/env bash
set -euo pipefail

CDP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_REPO_ROOT="$(cd "${CDP_REPO_ROOT}/../.." && pwd)"
CDP_SETUP_MODE="${1:-test}"
CDP_PYTHON_BIN="${CDP_PYTHON:-python3}"
CDP_VENV_PATH="${CDP_VENV:-${CDP_REPO_ROOT}/.venv}"

case "${CDP_SETUP_MODE}" in
  examples|test|full|check-paths) ;;
  *)
    echo "Usage: ./scripts/setup.sh [examples|test|full|check-paths]" >&2
    exit 2
    ;;
esac

if ! command -v "${CDP_PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python was not found: ${CDP_PYTHON_BIN}" >&2
  exit 1
fi

cd "${CDP_REPO_ROOT}"
if [[ "${CDP_SETUP_MODE}" == "check-paths" ]]; then
  CDP_EXPECTED_GATE_ROOT="$(cd ../.. && pwd)"
  if [[ "${CDP_EXPECTED_GATE_ROOT}" != "${GATE_REPO_ROOT}" ]]; then
    echo "Research setup does not resolve the enclosing Gate root." >&2
    exit 1
  fi
  if [[ ! -f "${GATE_REPO_ROOT}/pyproject.toml" ]]; then
    echo "Research setup resolved a Gate root without pyproject.toml." >&2
    exit 1
  fi
  echo "Research paths: ${CDP_REPO_ROOT} -> ${GATE_REPO_ROOT}"
  exit 0
fi
"${CDP_PYTHON_BIN}" -m venv "${CDP_VENV_PATH}"
CDP_VENV_PYTHON="${CDP_VENV_PATH}/bin/python"

"${CDP_VENV_PYTHON}" -m pip install -e \
  "${GATE_REPO_ROOT}[crypto,lockbox]"

if [[ "${CDP_SETUP_MODE}" == "test" ]]; then
  "${CDP_VENV_PYTHON}" -m pip install \
    -r "${CDP_REPO_ROOT}/experiments/requirements-test.txt"
elif [[ "${CDP_SETUP_MODE}" == "full" ]]; then
  "${CDP_VENV_PYTHON}" -m pip install \
    -r "${CDP_REPO_ROOT}/experiments/requirements.txt"
fi

"${CDP_VENV_PYTHON}" "${CDP_REPO_ROOT}/experiments/library_provenance.py" \
  >/dev/null

echo "Environment ready: ${CDP_VENV_PATH}"
echo "Activate with: source ${CDP_VENV_PATH}/bin/activate"

if [[ "${CDP_SETUP_MODE}" == "test" ]]; then
  (cd "${CDP_REPO_ROOT}" && "${CDP_VENV_PYTHON}" -m pytest -q)
fi
