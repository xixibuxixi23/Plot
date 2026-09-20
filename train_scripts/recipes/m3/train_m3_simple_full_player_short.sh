#!/usr/bin/env bash
set -euo pipefail

# Small full-network experiment: 1 observed frame + 8 generated frames.
# --steps is an absolute checkpoint step; parent=5000, final=5500 adds 500 steps.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
: "${PLOT_DATASET_ROOT:?Set the canonical repaired release root}"
: "${WARM_START:?Set an M3-Simple checkpoint}"
: "${OUTPUT_DIR:?Set a fresh output directory}"
: "${CUDA_VISIBLE_DEVICES:?Set the selected usable GPU(s)}"
subset_dir=${SUBSET_DIR:-$repo_root/derived/m3_player_short_20260919}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"

exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node="${#devices[@]}" \
  train_scripts/train_renderer.py \
  --dataset-root "$PLOT_DATASET_ROOT" --train-split train \
  --window-index "$subset_dir/train_c9.pt" \
  --val-window-index "$subset_dir/val_id_c9.pt" \
  --visualization-probe-manifest "$subset_dir/probes.json" \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --latent-normalization pixel-vae --warm-start "$WARM_START" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-staging-dir "${PLOT_CHECKPOINT_STAGING_DIR:-$OUTPUT_DIR/staging}" \
  --checkpoint-errors raise \
  --simple-m3 --qk-rms-norm --player-reference-token-grid 4 2 \
  --player-reference-position-encoding --actor-channels 32 \
  --context-frames 9 --cache-frames 32 --block-frames 8 \
  --target-views-per-window 1 --batch-size "${BATCH_SIZE:-1}" \
  --workers "${WORKERS:-2}" --prefetch-factor 2 --gradient-accumulation 1 \
  --steps "${FINAL_STEP:-5500}" --precision bf16 \
  --loss-mode combined --flow-loss-weight 1 \
  --latent-entity-region-upweight 0 \
  --latent-player-region-upweight 4 --player-mask-probability 1 \
  --pixel-loss-frames 2 --pixel-frame-selection player \
  --player-pixel-l1-weight 0.1 --player-pixel-edge-weight 0.025 \
  --entity-pixel-l1-weight 0 --entity-pixel-edge-weight 0 --health-pixel-l1-weight 0 \
  --lr "${BASE_LR:-1e-5}" --appearance-lr "${APPEARANCE_LR:-5e-5}" \
  --save-every 250 --validate-every 100 --val-batches 8 \
  --visualize-every 250 --visualization-denoising-steps 20 --log-every 10 \
  --seed 20260919 --wandb-entity "${WANDB_ENTITY:-ckx23-tsinghua-university}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-simple-full-player-c9-from5000}" \
  --wandb-mode "${WANDB_MODE:-online}" "$@"
