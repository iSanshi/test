#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-leap_singulation_genesis_style_conditioned_v1}"
BACKEND="${BACKEND:-gpu}"
NUM_ENVS="${NUM_ENVS:-4}"
ACTION_NOISE="${ACTION_NOISE:-0.08}"
SLEEP="${SLEEP:-0.02}"
RESET_INTERVAL="${RESET_INTERVAL:-700}"
STEPS="${STEPS:-100000}"
LOG_DIR="${ROOT_DIR}/logs/${RUN_NAME}"

if [[ ! -d "${LOG_DIR}" ]]; then
  echo "Missing log dir: ${LOG_DIR}" >&2
  exit 1
fi

LATEST_CKPT="$(
  find "${LOG_DIR}" -maxdepth 1 -name 'model_*.pt' -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
)"

if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found in: ${LOG_DIR}" >&2
  exit 1
fi

echo "Viewing checkpoint: ${LATEST_CKPT}"
echo "Viewer settings: backend=${BACKEND}, num_envs=${NUM_ENVS}, action_noise=${ACTION_NOISE}, reset_interval=${RESET_INTERVAL}, sleep=${SLEEP}"
exec "${ROOT_DIR}/scripts/eval.sh" \
  --backend "${BACKEND}" \
  --checkpoint "${LATEST_CKPT}" \
  --num-envs "${NUM_ENVS}" \
  --steps "${STEPS}" \
  --viewer \
  --stochastic \
  --action-noise "${ACTION_NOISE}" \
  --reset-interval "${RESET_INTERVAL}" \
  --sleep "${SLEEP}" \
  --use-contact-style-condition \
  "$@"
