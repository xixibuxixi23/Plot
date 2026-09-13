#!/usr/bin/env bash
# Explicit flow continuation baseline. Run only when another training is intended.
set -euo pipefail
PLOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PLOT_ROOT"
exec .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_persist_full.py \
  --phase train --objective flow \
  --resume outputs/m1_multiview_persist_gtcam_s01_all_v1/checkpoint_latest.pt \
  --epochs 30 --fixed-loss-samples 128 \
  --output outputs/m1_flow_baseline_continue_v1 "$@"
