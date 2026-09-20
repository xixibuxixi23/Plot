#!/usr/bin/env bash
set -euo pipefail

# Same 16 train / 8 held-out c9 clips as the original pilot, now with an
# independent player flow term. The shared recipe retains old defaults.
recipe_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FINAL_STEP=${FINAL_STEP:-6500}
export WANDB_NAME=${WANDB_NAME:-m3-multiclip16-independent-player-flow-from5500}
exec bash "$recipe_dir/train_m3_simple_full_player_short.sh" \
  --player-flow-loss-weight 1 --latent-player-region-upweight 0 \
  --save-every 100 --visualize-every 100 --validate-every 100 "$@"
