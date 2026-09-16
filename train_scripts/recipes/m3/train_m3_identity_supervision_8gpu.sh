#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"

dataset_root=${PLOT_DATASET_ROOT:-/data/huangyh/hxh/polis_v1_20260909_360p}
warm_start=${WARM_START:-outputs/m3_h100_8gpu_entity_reference_stage1_from25000_to27000_20260914/step_0026000.pt}
identity_checkpoint=${IDENTITY_CHECKPOINT:-outputs/player_identity_real_finetune_h100_5gpu_20260914/step_0003300.pt}
output_dir=${OUTPUT_DIR:-outputs/m3_identity_supervision_canary_from26000_to26500}
checkpoint_staging_dir=${PLOT_CHECKPOINT_STAGING_DIR:-.checkpoint_staging}
nproc=${NPROC_PER_NODE:-8}
nnodes=${NNODES:-1}
node_rank=${NODE_RANK:-0}
master_addr=${MASTER_ADDR:-127.0.0.1}
master_port=${MASTER_PORT:-29500}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}

for required_path in "$python_bin" "$dataset_root" "$warm_start" "$identity_checkpoint"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required path does not exist: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$output_dir" "$checkpoint_staging_dir"

launch_args=(--nproc_per_node="$nproc")
if (( nnodes > 1 )); then
  launch_args+=(
    --nnodes="$nnodes"
    --node_rank="$node_rank"
    --master_addr="$master_addr"
    --master_port="$master_port"
  )
else
  launch_args+=(--standalone)
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
"$python_bin" -m torch.distributed.run "${launch_args[@]}" \
  train_scripts/train_renderer.py \
  --dataset-root "$dataset_root" \
  --window-index derived/m3/validated/train_c65.pt \
  --val-window-index derived/m3/validated/val_id_c65.pt \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --warm-start "$warm_start" \
  --view-aware-appearance \
  --entity-reference-attention \
  --freeze-base-for-reference \
  --reference-unfreeze-last-spatial-blocks "${REFERENCE_UNFREEZE_LAST_SPATIAL_BLOCKS:-2}" \
  --unfrozen-base-lr-scale "${UNFROZEN_BASE_LR_SCALE:-0.1}" \
  --player-identity-checkpoint "$identity_checkpoint" \
  --player-identity-loss-weight "${PLAYER_IDENTITY_LOSS_WEIGHT:-0.1}" \
  --player-identity-margin "${PLAYER_IDENTITY_MARGIN:-0.2}" \
  --player-identity-negative-weight "${PLAYER_IDENTITY_NEGATIVE_WEIGHT:-0.5}" \
  --output-dir "$output_dir" \
  --checkpoint-staging-dir "$checkpoint_staging_dir" \
  --checkpoint-errors raise \
  --context-frames 65 \
  --cache-frames 64 \
  --block-frames 8 \
  --target-views-per-window 2 \
  --batch-size "${BATCH_SIZE:-1}" \
  --workers "${WORKERS:-4}" \
  --steps "${STEPS:-26500}" \
  --save-every "${SAVE_EVERY:-500}" \
  --validate-every "${VALIDATE_EVERY:-500}" \
  --visualize-every "${VISUALIZE_EVERY:-500}" \
  --visualization-denoising-steps "${VISUALIZATION_DENOISING_STEPS:-20}" \
  --val-batches "${VAL_BATCHES:-16}" \
  --precision bf16 \
  --latent-entity-region-upweight 0 \
  --latent-player-region-upweight "${LATENT_PLAYER_REGION_UPWEIGHT:-4}" \
  --player-mask-probability "${PLAYER_MASK_PROBABILITY:-0.5}" \
  --pixel-loss-frames 1 \
  --entity-pixel-l1-weight 0 \
  --entity-pixel-edge-weight 0 \
  --player-pixel-l1-weight 0 \
  --player-pixel-edge-weight 0 \
  --health-pixel-l1-weight 0 \
  --lr "${LR:-1e-4}" \
  --log-every "${LOG_EVERY:-20}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-identity-supervision-canary-26000-26500}" \
  --wandb-mode "${WANDB_MODE:-online}"
