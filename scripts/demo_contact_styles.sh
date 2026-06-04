#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-leap_singulation_genesis_style_conditioned_v1}"
BACKEND="${BACKEND:-gpu}"
NUM_ENVS="${NUM_ENVS:-1}"
ACTION_NOISE="${ACTION_NOISE:-0.02}"
SLEEP="${SLEEP:-0.02}"
RESET_INTERVAL="${RESET_INTERVAL:-700}"
STEPS="${STEPS:-100000}"

if [[ $# -gt 0 ]]; then
  CKPT="$1"
  shift
else
  LOG_DIR="${ROOT_DIR}/logs/${RUN_NAME}"
  if [[ ! -d "${LOG_DIR}" ]]; then
    echo "Missing log dir: ${LOG_DIR}" >&2
    exit 1
  fi
  CKPT="$(
    find "${LOG_DIR}" -maxdepth 1 -name 'model_*.pt' -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )"
fi

if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT:-<empty>}" >&2
  exit 1
fi

STYLE_NAMES=("fingertip" "palm" "mcp/proximal" "pip/dip" "thumb-side")

for STYLE_ID in 0 1 2 3 4; do
  echo "Viewing style ${STYLE_ID}: ${STYLE_NAMES[${STYLE_ID}]}"
  "${ROOT_DIR}/scripts/eval.sh" \
    --backend "${BACKEND}" \
    --checkpoint "${CKPT}" \
    --num-envs "${NUM_ENVS}" \
    --steps "${STEPS}" \
    --viewer \
    --stochastic \
    --action-noise "${ACTION_NOISE}" \
    --reset-interval "${RESET_INTERVAL}" \
    --sleep "${SLEEP}" \
    --use-contact-style-condition \
    --fixed-contact-style "${STYLE_ID}" \
    "$@"
done
