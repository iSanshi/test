#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${ROOT_DIR}"

exec ./scripts/train.sh \
  --backend gpu \
  --num-envs 512 \
  --max-iterations 20000 \
  --save-interval 200 \
  --exp-name leap_singulation_genesis_finger_diverse \
  --state-running-max-mode state \
  --reach-curiosity-scale 5.12 \
  --contact-coverage-scale 800 \
  --contact-diversity-scale 8 \
  --non-fingertip-target-penalty-scale 0 \
  --action-penalty-scale 0.05 \
  --wrench-prob 0.02 \
  --force-scale 0.6 \
  --torque-scale 0.02 \
  "$@"
