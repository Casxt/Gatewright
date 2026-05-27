#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH}"
else
  export PYTHONPATH="${SCRIPT_DIR}/src"
fi

exec "${PYTHON_BIN}" -m gatewright.orchestrator.runner "$@"
