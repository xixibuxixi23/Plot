#!/usr/bin/env bash
set -euo pipefail

# Refine an already-QK M3 checkpoint with intact AdamW moments and a lower LR.
# This is an ordinary strict resume, not the one-time no-QK -> QK migration.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

dataset_root=${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT to the existing release}
output_dir=${OUTPUT_DIR:?Set a NEW output directory for background refinement}
staging_dir=${PLOT_CHECKPOINT_STAGING_DIR:?Set PLOT_CHECKPOINT_STAGING_DIR}
resume=${RESUME:?Set RESUME to the latest complete checkpoint from d7dev01k}
train_index=${M3_WINDOW_INDEX:-$dataset_root/derived/m3/validated/train_c65.pt}
val_index=${M3_VAL_WINDOW_INDEX:-$dataset_root/derived/m3/validated/val_id_c65.pt}
chunk_cache_root=${M3_CHUNK_CACHE_ROOT:-}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the confirmed free GPUs}"
IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
selected_gpu_count=${#selected_gpus[@]}
nproc=${NPROC_PER_NODE:-$selected_gpu_count}
if [[ "$nproc" -ne 7 || "${BATCH_SIZE:-4}" -ne 4 ]]; then
  echo "This recipe requires the established 7-GPU, batch-4 setup" >&2
  exit 1
fi
if [[ "$nproc" -ne "$selected_gpu_count" ]]; then
  echo "NPROC_PER_NODE=$nproc does not match CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
  exit 1
fi

for required_path in \
  "$dataset_root" "$train_index" "$val_index" \
  derived/common/block_vocabulary.json checkpoints/pixel_vae/model.safetensors \
  "$resume"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

chunk_cache_args=()
if [[ -n "$chunk_cache_root" ]]; then
  chunk_cache_args=(--chunk-cache-root "$chunk_cache_root")
fi

if [[ -d "$output_dir" && -n "$(ls -A -- "$output_dir")" ]]; then
  echo "Output directory must be new or empty: $output_dir" >&2
  exit 1
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
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$staging_dir" \
  --checkpoint-errors raise \
  --simple-m3 \
  --qk-rms-norm \
  --resume "$resume" \
  --override-resume-lr \
  --seed 0 \
  --train-sampler-seed "${TRAIN_SAMPLER_SEED:-20260920}" \
  --player-reference-token-grid 4 2 \
  --player-reference-position-encoding \
  --actor-channels 32 \
  --context-frames 65 --cache-frames 32 --block-frames 8 \
  --target-views-per-window 1 \
  --batch-size "${BATCH_SIZE:-4}" \
  --workers "${WORKERS:-4}" \
  --steps "${STEPS:-40000}" \
  --precision bf16 --gradient-accumulation 1 \
  --loss-mode flow \
  --latent-entity-region-upweight 0 \
  --latent-player-region-upweight 0 \
  --lr "${LR:-1e-5}" \
  --save-every "${SAVE_EVERY:-500}" \
  --validate-every "${VALIDATE_EVERY:-1000}" \
  --visualize-every "${VISUALIZE_EVERY:-1000}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --wandb-entity "${WANDB_ENTITY:-ckx23-tsinghua-university}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-qk-background-refine-lr1e5-7gpu-b4}" \
  --wandb-mode "${WANDB_MODE:-online}"
