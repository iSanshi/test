#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GENESIS_DIR="${GENESIS_DIR:-/mnt/p5/genesis-world-v1.0.0}"

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${GENESIS_DIR}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi
"${PYTHON_BIN}" evaluate_contact_styles.py "$@"
