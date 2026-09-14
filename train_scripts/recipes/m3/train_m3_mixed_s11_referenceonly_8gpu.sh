#!/usr/bin/env bash
set -euo pipefail

# Wait for the complete S11 collection, build one immutable grouped index, and
# then continue the validated reference-only M3 run.  This recipe deliberately
# leaves the legacy four-view texture-projection path disabled.

PLOT_ROOT="${PLOT_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/Plot}"
S11_ROOT="${S11_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/s11_mixed_train_v2_g100_v4_20260914}"
BASE_ROOT="${BASE_ROOT:-/public/0_DATA/2_Avatar/zhizhou_share/rcz/textagent/data/releases/polis_v1_20260909_360p}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${PLOT_ROOT}/outputs/m3_mixed_mild_referenceonly_26900_27000_20260914/step_0027000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PLOT_ROOT}/outputs/m3_mixed_s11_referenceonly_formal_27000_37000_20260914}"
INDEX_PATH="${INDEX_PATH:-${PLOT_ROOT}/derived/s11/mixed_train_v2_complete_g100_c65.pt}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-400}"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-160}"
COLLECTION_LOG_DIR="${COLLECTION_LOG_DIR:-${S11_ROOT}/cluster_logs/train_5x32}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
S11_STEP_PROBABILITY="${S11_STEP_PROBABILITY:-0.05}"
FINAL_STEP="${FINAL_STEP:-37000}"

mkdir -p "${OUTPUT_DIR}"
cd "${PLOT_ROOT}"

count_usable() {
  find "${S11_ROOT}/train" -mindepth 2 -maxdepth 2 -name validation.json -print0 \
    | xargs -0 -r jq -r 'select(.usable == true) | 1' \
    | wc -l
}

while true; do
  completed_shards=0
  running_shards=0
  failed_shards=0
  for ((shard = 0; shard < EXPECTED_SHARDS; shard++)); do
    shard_name="$(printf '%03d' "${shard}")"
    pid_file="${COLLECTION_LOG_DIR}/shard_${shard_name}.pid"
    exit_file="${COLLECTION_LOG_DIR}/shard_${shard_name}.exit"
    if [[ -s "${pid_file}" ]] && kill -0 "$(<"${pid_file}")" 2>/dev/null; then
      # A stale exit marker may remain from an earlier failed launch.  The PID
      # file is written by the current worker, so a live PID takes precedence.
      running_shards=$((running_shards + 1))
      continue
    fi
    [[ -e "${exit_file}" ]] || continue
    completed_shards=$((completed_shards + 1))
    [[ "$(<"${exit_file}")" == "0" ]] || failed_shards=$((failed_shards + 1))
  done
  usable="$(count_usable)"
  printf '%s collection usable=%s/%s running_shards=%s completed_shards=%s/%s failed_shards=%s\n' \
    "$(date -u +%FT%TZ)" "${usable}" "${EXPECTED_EPISODES}" \
    "${running_shards}" "${completed_shards}" "${EXPECTED_SHARDS}" "${failed_shards}"
  if (( failed_shards > 0 )); then
    echo "S11 collection has failed shards; refusing to start formal training" >&2
    exit 2
  fi
  if (( completed_shards == EXPECTED_SHARDS )); then
    if (( usable != EXPECTED_EPISODES )); then
      echo "S11 collection ended with ${usable}/${EXPECTED_EPISODES} usable episodes" >&2
      exit 3
    fi
    break
  fi
  sleep 30
done

PYTHONPATH="${PLOT_ROOT}" "${PLOT_ROOT}/.venv/bin/python" \
  dataset_toolkits/build_s11_counterfactual_index.py "${S11_ROOT}" \
  --output "${INDEX_PATH}" --split train --offsets 0,24,48,72 \
  --targets 0,1,2,3 --variants-per-group 4

# Do not steal a GPU if another experiment appeared while collection ran.
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')" ]]; do
  printf '%s waiting for all experiment GPUs to become idle\n' "$(date -u +%FT%TZ)"
  sleep 30
done

[[ -f "${RESUME_CHECKPOINT}" ]] || {
  echo "missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2
  exit 4
}

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export PYTHONPATH="${PLOT_ROOT}"
export WANDB_DIR="${OUTPUT_DIR}/wandb"
mkdir -p "${WANDB_DIR}"

exec "${PLOT_ROOT}/.venv/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=8 train_scripts/train_renderer.py \
  --dataset-root "${BASE_ROOT}" --train-split train \
  --window-index derived/m3/validated/train_c65.pt \
  --counterfactual-dataset-root "${S11_ROOT}" \
  --counterfactual-window-index "${INDEX_PATH}" \
  --counterfactual-step-probability "${S11_STEP_PROBABILITY}" \
  --counterfactual-variants 4 \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --resume "${RESUME_CHECKPOINT}" --output-dir "${OUTPUT_DIR}" \
  --unified-player-reference --freeze-base-for-reference \
  --context-frames 65 --cache-frames 64 --block-frames 8 \
  --target-views-per-window 1 --batch-size 1 --workers 1 \
  --steps "${FINAL_STEP}" --save-every 250 --validate-every 1000 \
  --visualize-every 0 --log-every 10 --precision bf16 --gradient-accumulation 1 \
  --latent-player-region-upweight 8 --player-mask-probability 0.5 \
  --counterfactual-player-mask-probability 1 \
  --player-pixel-l1-weight 1 --player-pixel-edge-weight 0.25 \
  --entity-pixel-l1-weight 0 --entity-pixel-edge-weight 0 \
  --health-pixel-l1-weight 0 --counterfactual-player-difference-weight 0.1 \
  --mask-prefix-player-probability 0 \
  --counterfactual-mask-prefix-player-probability 1 \
  --pixel-loss-frames 2 \
  --player-identity-checkpoint checkpoints/m3/player_identity_real_finetune/step_0003300.pt \
  --player-identity-loss-weight 0.5 --player-identity-margin 0.3 \
  --player-identity-negative-weight 1 --lr 0.00001 --seed 3 \
  --wandb-mode offline \
  --checkpoint-staging-dir "${PLOT_ROOT}/.checkpoint_staging_mixed_s11_formal" \
  --checkpoint-errors raise
