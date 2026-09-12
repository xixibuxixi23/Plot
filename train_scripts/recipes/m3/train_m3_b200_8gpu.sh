#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

: "${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT to the polis_v1_20260909_360p release}"
: "${PLOT_CHECKPOINT_STAGING_DIR:?Set this to node-local SSD or PFS, never OSS FUSE}"

nproc=${NPROC_PER_NODE:-8}
output_dir=${OUTPUT_DIR:-outputs/m3_formal}

python scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"

torchrun --standalone --nproc_per_node="$nproc" train_scripts/train_renderer.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --window-index derived/m3/train_c65.pt \
  --val-window-index derived/m3/val_id_c65.pt \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --backbone-checkpoint checkpoints/m3_backbone/model.safetensors \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR" \
  --context-frames 65 --cache-frames 32 --target-views-per-window 2 \
  --batch-size "${BATCH_SIZE:-1}" --steps "${STEPS:-10000}" \
  --latent-entity-region-upweight "${LATENT_ENTITY_REGION_UPWEIGHT:-0}" \
  --pixel-loss-frames "${PIXEL_LOSS_FRAMES:-1}" \
  --entity-pixel-l1-weight "${ENTITY_PIXEL_L1_WEIGHT:-0.1}" \
  --entity-pixel-edge-weight "${ENTITY_PIXEL_EDGE_WEIGHT:-0.05}" \
  --health-pixel-l1-weight "${HEALTH_PIXEL_L1_WEIGHT:-0.2}" \
  --save-every "${SAVE_EVERY:-1000}" \
  --validate-every "${VALIDATE_EVERY:-1000}" \
  --visualize-every "${VISUALIZE_EVERY:-1000}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-formal-b200}" \
  --wandb-mode "${WANDB_MODE:-online}"
