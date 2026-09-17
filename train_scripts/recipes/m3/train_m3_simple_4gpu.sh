#!/usr/bin/env bash
set -euo pipefail

# Clean M3 retraining recipe.  The network has exactly three condition routes:
# a fused scene encoder, target-state AdaLN, and one compact appearance memory.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

dataset_root=${PLOT_DATASET_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/polis_two_player_fixed_skins_complete_20260917_360p}
output_dir=${OUTPUT_DIR:-outputs/m3_simple_fixed_skins_40k_20260917}
staging_dir=${PLOT_CHECKPOINT_STAGING_DIR:-/tmp/plot_m3_simple_checkpoints}
train_index=${M3_WINDOW_INDEX:-$dataset_root/derived/m3/validated/train_c65.pt}
val_index=${M3_VAL_WINDOW_INDEX:-$dataset_root/derived/m3/validated/val_id_c65.pt}
chunk_cache_root=${M3_CHUNK_CACHE_ROOT:-}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
nproc=${NPROC_PER_NODE:-4}

for required_path in \
  "$dataset_root" "$train_index" "$val_index" \
  derived/common/block_vocabulary.json checkpoints/pixel_vae/model.safetensors; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

chunk_cache_args=()
if [[ -n "$chunk_cache_root" ]]; then
  chunk_cache_args=(--chunk-cache-root "$chunk_cache_root")
fi

mkdir -p "$output_dir" "$staging_dir"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7} \
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node="$nproc" \
  train_scripts/train_renderer.py \
  --dataset-root "$dataset_root" \
  --window-index "$train_index" \
  --val-window-index "$val_index" \
  "${chunk_cache_args[@]}" \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$staging_dir" \
  --checkpoint-errors raise \
  --simple-m3 \
  --player-reference-token-grid 4 2 \
  --player-reference-position-encoding \
  --actor-channels 32 \
  --context-frames 65 --cache-frames 32 --block-frames 8 \
  --target-views-per-window 1 \
  --batch-size "${BATCH_SIZE:-1}" \
  --workers "${WORKERS:-4}" \
  --steps "${STEPS:-40000}" \
  --precision bf16 --gradient-accumulation 1 \
  --loss-mode combined \
  --latent-entity-region-upweight 0 \
  --latent-player-region-upweight "${LATENT_PLAYER_REGION_UPWEIGHT:-8}" \
  --player-mask-probability 1 \
  --pixel-loss-frames "${PIXEL_LOSS_FRAMES:-4}" \
  --entity-pixel-l1-weight 0 \
  --entity-pixel-edge-weight 0 \
  --player-pixel-l1-weight "${PLAYER_PIXEL_L1_WEIGHT:-8}" \
  --player-pixel-edge-weight "${PLAYER_PIXEL_EDGE_WEIGHT:-2}" \
  --health-pixel-l1-weight 0 \
  --lr "${LR:-1e-4}" \
  --save-every "${SAVE_EVERY:-500}" \
  --validate-every "${VALIDATE_EVERY:-1000}" \
  --visualize-every "${VISUALIZE_EVERY:-1000}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --wandb-entity "${WANDB_ENTITY:-ckx23-tsinghua-university}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-simple-fixed-skins-40k}" \
  --wandb-mode "${WANDB_MODE:-online}"
