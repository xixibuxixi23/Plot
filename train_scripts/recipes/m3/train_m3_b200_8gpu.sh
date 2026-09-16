#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

: "${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT to the polis_v1_20260909_360p release}"
: "${PLOT_CHECKPOINT_STAGING_DIR:?Set this to node-local SSD or PFS, never OSS FUSE}"

nproc=${NPROC_PER_NODE:-8}
output_dir=${OUTPUT_DIR:-outputs/m3_formal}
train_index=${M3_WINDOW_INDEX:-derived/m3/validated/train_c65.pt}
val_index=${M3_VAL_WINDOW_INDEX:-derived/m3/validated/val_id_c65.pt}
chunk_cache_root=${M3_CHUNK_CACHE_ROOT:-}
health_focus_index=${M3_HEALTH_FOCUS_INDEX:-}

for required_path in \
  "$PLOT_DATASET_ROOT" "$train_index" "$val_index" \
  derived/common/block_vocabulary.json \
  checkpoints/pixel_vae/model.safetensors \
  checkpoints/m3_backbone/model.safetensors; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

chunk_cache_args=()
if [[ -n "$chunk_cache_root" ]]; then
  if [[ ! -d "$chunk_cache_root" ]]; then
    echo "M3_CHUNK_CACHE_ROOT is not a directory: $chunk_cache_root" >&2
    exit 1
  fi
  chunk_cache_args=(--chunk-cache-root "$chunk_cache_root")
fi

health_focus_args=()
if [[ -n "$health_focus_index" ]]; then
  if [[ ! -f "$health_focus_index" ]]; then
    echo "M3_HEALTH_FOCUS_INDEX does not exist: $health_focus_index" >&2
    exit 1
  fi
  health_focus_args=(
    --health-focus-index "$health_focus_index"
    --health-focus-oversample "${HEALTH_FOCUS_OVERSAMPLE:-5}"
  )
fi

python scripts/check_m3_environment.py \
  --device cuda:0 \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR"

torchrun --standalone --nproc_per_node="$nproc" train_scripts/train_renderer.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --window-index "$train_index" \
  --val-window-index "$val_index" \
  "${chunk_cache_args[@]}" \
  "${health_focus_args[@]}" \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --backbone-checkpoint checkpoints/m3_backbone/model.safetensors \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$PLOT_CHECKPOINT_STAGING_DIR" \
  --context-frames 65 --cache-frames 32 --target-views-per-window 2 \
  --batch-size "${BATCH_SIZE:-1}" --workers "${WORKERS:-4}" --steps "${STEPS:-10000}" \
  --loss-mode "${LOSS_MODE:-combined}" \
  --latent-entity-region-upweight "${LATENT_ENTITY_REGION_UPWEIGHT:-0}" \
  --pixel-loss-frames "${PIXEL_LOSS_FRAMES:-2}" \
  --entity-pixel-l1-weight "${ENTITY_PIXEL_L1_WEIGHT:-0.5}" \
  --entity-pixel-edge-weight "${ENTITY_PIXEL_EDGE_WEIGHT:-0.2}" \
  --health-pixel-l1-weight "${HEALTH_PIXEL_L1_WEIGHT:-1.0}" \
  --damaged-health-upweight "${DAMAGED_HEALTH_UPWEIGHT:-4}" \
  --save-every "${SAVE_EVERY:-1000}" \
  --validate-every "${VALIDATE_EVERY:-1000}" \
  --visualize-every "${VISUALIZE_EVERY:-1000}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-formal-b200}" \
  --wandb-mode "${WANDB_MODE:-online}"
