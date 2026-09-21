#!/usr/bin/env bash
set -euo pipefail

# Continue the completed QK background-refinement checkpoint with a lower LR
# and exact full-resolution player supervision through the frozen Pixel VAE
# decoder. Global flow matching remains the primary scene objective. No
# resized latent player mask is used by this recipe.

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

dataset_root=${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT to the existing release}
output_dir=${OUTPUT_DIR:?Set a NEW output directory for player refinement}
staging_dir=${PLOT_CHECKPOINT_STAGING_DIR:?Set PLOT_CHECKPOINT_STAGING_DIR}
resume=${RESUME:?Set RESUME to the complete step_0040000.pt from run okd5jze8}
train_index=${M3_WINDOW_INDEX:-$dataset_root/derived/m3/validated/train_c65.pt}
val_index=${M3_VAL_WINDOW_INDEX:-$dataset_root/derived/m3/validated/val_id_c65.pt}
chunk_cache_root=${M3_CHUNK_CACHE_ROOT:-}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to the confirmed free GPUs}"
IFS=',' read -r -a selected_gpus <<< "$CUDA_VISIBLE_DEVICES"
selected_gpu_count=${#selected_gpus[@]}
nproc=${NPROC_PER_NODE:-$selected_gpu_count}
batch_size=${BATCH_SIZE:-2}
gradient_accumulation=${GRADIENT_ACCUMULATION:-2}

if [[ "$nproc" -ne 7 || $((batch_size * gradient_accumulation)) -ne 4 ]]; then
  echo "This recipe requires 7 GPUs and BATCH_SIZE*GRADIENT_ACCUMULATION=4" >&2
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
  --train-sampler-seed "${TRAIN_SAMPLER_SEED:-20260921}" \
  --player-reference-token-grid 4 2 \
  --player-reference-position-encoding \
  --actor-channels 32 \
  --context-frames 65 --cache-frames 32 --block-frames 8 \
  --target-views-per-window 1 \
  --batch-size "$batch_size" \
  --workers "${WORKERS:-4}" \
  --steps "${STEPS:-44000}" \
  --precision bf16 --gradient-accumulation "$gradient_accumulation" \
  --loss-mode combined \
  --latent-entity-region-upweight 0 \
  --latent-player-region-upweight 0 \
  --pixel-loss-frames 1 \
  --pixel-frame-selection player_unique \
  --entity-pixel-l1-weight 0 \
  --entity-pixel-edge-weight 0 \
  --player-pixel-l1-weight "${PLAYER_PIXEL_L1_WEIGHT:-1.0}" \
  --player-pixel-edge-weight "${PLAYER_PIXEL_EDGE_WEIGHT:-0.25}" \
  --health-pixel-l1-weight 0 \
  --player-identity-loss-weight 0 \
  --counterfactual-player-difference-weight 0 \
  --lr "${LR:-3e-6}" \
  --save-every "${SAVE_EVERY:-500}" \
  --validate-every "${VALIDATE_EVERY:-1000}" \
  --visualize-every "${VISUALIZE_EVERY:-1000}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --wandb-entity "${WANDB_ENTITY:-ckx23-tsinghua-university}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-qk-player-pixel-refine-lr3e6-from40000-7gpu}" \
  --wandb-mode "${WANDB_MODE:-online}"
