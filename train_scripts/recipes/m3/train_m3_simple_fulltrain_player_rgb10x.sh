#!/usr/bin/env bash
set -euo pipefail

# Reuse the exact pixels-only trial launcher (same warm-start, LR, data and
# evaluation). Final CLI options override its loss/selection defaults.
# Required: PLOT_DATASET_ROOT, INIT_CHECKPOINT, FINAL_STEP, OUTPUT_DIR, GPUs.
export WANDB_NAME=${WANDB_NAME:-m3-fulltrain-c9-flow-rgb10x-unique-8gpu-b4}
recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec bash "$recipe_dir/train_m3_simple_fulltrain_player_pixels_only.sh" \
  --flow-loss-weight 1 --player-flow-loss-weight 1 \
  --player-pixel-l1-weight 1 --player-pixel-edge-weight 0.25 \
  --pixel-frame-selection player_unique --log-player-noise-bins "$@"
