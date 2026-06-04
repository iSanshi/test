#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

exec ./scripts/train.sh \
  --backend gpu \
  --num-envs 256 \
  --max-iterations 8000 \
  --save-interval 200 \
  --exp-name leap_singulation_genesis_style_conditioned_v1 \
  --state-running-max-mode global \
  --reach-curiosity-scale 1.28 \
  --contact-coverage-scale 100 \
  --contact-diversity-scale 20 \
  --use-contact-style-condition \
  --fixed-contact-style -1 \
  --contact-style-bonus-scale 5 \
  --contact-style-progress-bonus-scale 80 \
  --wrong-style-contact-penalty-scale 0.5 \
  --use-link-contact-diversity-reward \
  --link-contact-diversity-alpha 0.7 \
  --finger-contact-diversity-alpha 0.3 \
  --non-fingertip-target-penalty-scale 0 \
  --failed-penalty-scale -200 \
  --non-target-arm-contact-penalty-scale 5 \
  --action-penalty-scale 0.02 \
  --wrench-prob 0.0 \
  --force-scale 0.0 \
  --torque-scale 0.0 \
  "$@"
