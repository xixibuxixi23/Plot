#!/usr/bin/env bash
set -euo pipefail

# Full-network player-detail training. Preserve higher-resolution reference tokens,
# condition multiple DiT depths, and optimize correct-reference player pixels.

PLOT_ROOT="${PLOT_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/Plot}"
BASE_ROOT="${BASE_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/polis_v1_20260909_360p}"
S11_ROOT="${S11_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/s11_mixed_train_v2_g100_v4_20260914}"
INDEX_PATH="${INDEX_PATH:-${PLOT_ROOT}/derived/s11/mixed_train_v2_complete_g100_c65.pt}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PLOT_ROOT}/outputs/m3_mixed_s11_referenceonly_formal_27000_37000_20260914/step_0034750.pt}"
RESUME="${RESUME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PLOT_ROOT}/outputs/m3_full_player_detail_34750_39750_20260915}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
FINAL_STEP="${FINAL_STEP:-39750}"
S11_STEP_PROBABILITY="${S11_STEP_PROBABILITY:-0.20}"
LEARNING_RATE="${LEARNING_RATE:-0.00002}"
REINJECT_BLOCKS="${REINJECT_BLOCKS-3 7 11}"

CHECKPOINT_PATH="${RESUME:-${BASE_CHECKPOINT}}"
[[ -f "${CHECKPOINT_PATH}" ]] || {
  echo "missing checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
}
[[ -f "${INDEX_PATH}" ]] || {
  echo "missing audited S11 index: ${INDEX_PATH}" >&2
  exit 3
}

mkdir -p "${OUTPUT_DIR}"
cd "${PLOT_ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${PLOT_ROOT}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

reinjection_args=()
if [[ -n "${REINJECT_BLOCKS}" ]]; then
  read -r -a reinjection_blocks <<< "${REINJECT_BLOCKS}"
  reinjection_args=(--unified-reference-reinject-blocks "${reinjection_blocks[@]}")
fi

checkpoint_args=(--warm-start "${BASE_CHECKPOINT}")
if [[ -n "${RESUME}" ]]; then
  checkpoint_args=(--resume "${RESUME}")
fi

exec "${PLOT_ROOT}/.venv/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=8 train_scripts/train_renderer.py \
  --dataset-root "${BASE_ROOT}" --train-split train \
  --window-index derived/m3/validated/train_c65.pt \
  --val-window-index derived/m3/validated/val_id_c65.pt \
  --counterfactual-dataset-root "${S11_ROOT}" \
  --counterfactual-window-index "${INDEX_PATH}" \
  --counterfactual-step-probability "${S11_STEP_PROBABILITY}" \
  --counterfactual-variants 4 \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  "${checkpoint_args[@]}" --output-dir "${OUTPUT_DIR}" \
  --unified-player-reference "${reinjection_args[@]}" \
  --player-reference-token-grid 16 8 \
  --player-reference-position-encoding \
  --counterfactual-random-timesteps \
  --context-frames 65 --cache-frames 64 --block-frames 8 \
  --target-views-per-window 1 --batch-size 1 --workers 1 \
  --steps "${FINAL_STEP}" --save-every 250 --validate-every 1000 \
  --visualize-every 0 --log-every 10 --precision bf16 --gradient-accumulation 1 \
  --latent-player-region-upweight 8 --player-mask-probability 0.5 \
  --counterfactual-player-mask-probability 1 \
  --player-pixel-l1-weight 1 --player-pixel-edge-weight 0.5 \
  --entity-pixel-l1-weight 0 --entity-pixel-edge-weight 0 \
  --health-pixel-l1-weight 0 --counterfactual-player-difference-weight 0 \
  --mask-prefix-player-probability 0 \
  --counterfactual-mask-prefix-player-probability 1 \
  --pixel-loss-frames 2 \
  --lr "${LEARNING_RATE}" --seed 3 \
  --wandb-mode offline \
  --checkpoint-staging-dir "${PLOT_ROOT}/.checkpoint_staging_full_player_detail" \
  --checkpoint-errors raise
