#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

: "${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT first}"
: "${PLOT_CHECKPOINT_STAGING_DIR:?Set PLOT_CHECKPOINT_STAGING_DIR first}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
export NPROC_PER_NODE=${NPROC_PER_NODE:-4}
tune_root=${PLOT_BATCH_TUNE_ROOT:-outputs/m3_simple_batch_tuning}
mkdir -p "$tune_root"

for candidate in 4 2 1; do
  candidate_output="$tune_root/batch_${candidate}"
  candidate_log="$tune_root/batch_${candidate}.log"
  echo "Testing M3-Simple batch=$candidate on $NPROC_PER_NODE GPUs" \
    | tee "$tune_root/batch_${candidate}.status"
  if env \
      BATCH_SIZE="$candidate" \
      OUTPUT_DIR="$candidate_output" \
      STEPS=1 \
      SAVE_EVERY=500 \
      VALIDATE_EVERY=1000 \
      VISUALIZE_EVERY=1000 \
      WANDB_MODE=disabled \
      bash train_scripts/recipes/m3/train_m3_simple_4gpu.sh \
      >"$candidate_log" 2>&1; then
    effective_batch=$((candidate * NPROC_PER_NODE))
    {
      printf 'export BATCH_SIZE=%q\n' "$candidate"
      printf 'export NPROC_PER_NODE=%q\n' "$NPROC_PER_NODE"
      printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$CUDA_VISIBLE_DEVICES"
      printf 'export EFFECTIVE_BATCH_SIZE=%q\n' "$effective_batch"
    } >"$tune_root/selected_batch.env"
    echo "Selected batch=$candidate; effective batch=$effective_batch"
    echo "Run: source $tune_root/selected_batch.env"
    exit 0
  fi
  echo "batch=$candidate failed; see $candidate_log" \
    | tee -a "$tune_root/batch_${candidate}.status"
done

echo "No candidate batch completed successfully; inspect $tune_root/*.log" >&2
exit 1
