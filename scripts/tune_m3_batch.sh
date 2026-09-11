#!/usr/bin/env bash
set -uo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
: "${PLOT_DATASET_ROOT:?Set PLOT_DATASET_ROOT first}"
: "${PLOT_CHECKPOINT_STAGING_DIR:?Set PLOT_CHECKPOINT_STAGING_DIR first}"

export NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}
tune_root=${PLOT_BATCH_TUNE_ROOT:-outputs/m3_batch_tuning}
mkdir -p "$tune_root"

for candidate in 4 2 1; do
  echo "Trying per-GPU batch=$candidate on $NPROC_PER_NODE GPUs" | tee "$tune_root/batch_${candidate}.status"
  if env \
      BATCH_SIZE="$candidate" STEPS=1 SAVE_EVERY=1 VALIDATE_EVERY=1 \
      VISUALIZE_EVERY=0 WANDB_MODE=disabled \
      OUTPUT_DIR="$tune_root/batch_${candidate}" \
      bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh \
      >"$tune_root/batch_${candidate}.log" 2>&1; then
    printf 'export BATCH_SIZE=%s\n' "$candidate" > "$tune_root/selected_batch.env"
    echo "Selected per-GPU batch=$candidate; source $tune_root/selected_batch.env"
    exit 0
  fi
  echo "batch=$candidate failed; see $tune_root/batch_${candidate}.log" >&2
done

echo "No candidate batch completed; inspect logs before formal training" >&2
exit 1
