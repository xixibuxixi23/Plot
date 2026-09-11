#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
: "${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT to the extracted 360p release}"
: "${PLOT_M3_CHECKPOINT:?Set PLOT_M3_CHECKPOINT to a trained M3 checkpoint}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train_scripts/train_policy.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --train-index derived/m4/train.jsonl \
  --val-index derived/m4/val_id.jsonl \
  --vocabulary derived/common/block_vocabulary.json \
  --text-catalog derived/m4/text_catalog.json \
  --text-cache checkpoints/m4_text/text_cache.safetensors \
  --m3-checkpoint "$PLOT_M3_CHECKPOINT" \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --output-dir "${OUTPUT_DIR:-outputs/m4}" \
  --batch-size "${BATCH_SIZE:-1}" --steps "${STEPS:-10000}"
