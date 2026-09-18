#!/usr/bin/env bash
set -euo pipefail

# Player-only adaptation for M3-Simple. The full checkpoint supplies the
# established scene denoiser; only the reference encoder and ROI projector move.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

dataset_root=${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT to the downloaded release}
warm_start=${WARM_START:?Set WARM_START to an M3-Simple checkpoint}
output_dir=${OUTPUT_DIR:-outputs/m3_simple_player_appearance}
staging_dir=${PLOT_CHECKPOINT_STAGING_DIR:?Set PLOT_CHECKPOINT_STAGING_DIR}
train_index=${M3_WINDOW_INDEX:-$dataset_root/derived/m3/validated/train_c65.pt}
val_index=${M3_VAL_WINDOW_INDEX:-$dataset_root/derived/m3/validated/val_id_c65.pt}
chunk_cache_root=${M3_CHUNK_CACHE_ROOT:-}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the confirmed usable GPUs}"
IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
nproc=${NPROC_PER_NODE:-${#selected_gpus[@]}}
if [[ "$nproc" -ne "${#selected_gpus[@]}" ]]; then
  echo "NPROC_PER_NODE=$nproc does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

for required_path in \
  "$dataset_root" "$train_index" "$val_index" "$warm_start" \
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

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$python_bin" -m torch.distributed.run --standalone --nproc-per-node="$nproc" \
  train_scripts/train_renderer.py \
  --dataset-root "$dataset_root" \
  --window-index "$train_index" \
  --val-window-index "$val_index" \
  "${chunk_cache_args[@]}" \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --latent-normalization pixel-vae \
  --warm-start "$warm_start" \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$staging_dir" \
  --checkpoint-errors raise \
  --simple-m3 \
  --qk-rms-norm \
  --player-reference-token-grid 4 2 \
  --player-reference-position-encoding \
  --actor-channels 32 \
  --freeze-base-for-player-appearance \
  --context-frames 65 --cache-frames 32 --block-frames 8 \
  --target-views-per-window 1 \
  --batch-size "${BATCH_SIZE:-2}" \
  --workers "${WORKERS:-4}" \
  --steps "${STEPS:-5000}" \
  --precision bf16 --gradient-accumulation 1 \
  --loss-mode combined \
  --flow-loss-weight 0 \
  --latent-entity-region-upweight 0 \
  --latent-player-region-upweight 0 \
  --pixel-loss-frames "${PIXEL_LOSS_FRAMES:-1}" \
  --pixel-frame-selection player \
  --entity-pixel-l1-weight 0 \
  --entity-pixel-edge-weight 0 \
  --player-pixel-l1-weight "${PLAYER_PIXEL_L1_WEIGHT:-1}" \
  --player-pixel-edge-weight "${PLAYER_PIXEL_EDGE_WEIGHT:-0.25}" \
  --health-pixel-l1-weight 0 \
  --lr "${LR:-3e-5}" \
  --save-every "${SAVE_EVERY:-500}" \
  --validate-every "${VALIDATE_EVERY:-500}" \
  --visualize-every "${VISUALIZE_EVERY:-500}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --wandb-entity "${WANDB_ENTITY:-ckx23-tsinghua-university}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-simple-player-appearance}" \
  --wandb-mode "${WANDB_MODE:-online}"
