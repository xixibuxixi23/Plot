#!/usr/bin/env bash
set -euo pipefail

# Short full-network ablation: only decoded player RGB/edge supervision.
# Warm-start keeps the checkpoint's absolute step but resets AdamW moments.
# Keep the old RGB/edge coefficients: the only loss change is removing flow.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$repo_root"
: "${PLOT_DATASET_ROOT:?Set the canonical repaired release root}"
: "${INIT_CHECKPOINT:?Set the checkpoint from the stopped parent run}"
: "${FINAL_STEP:?Set the parent checkpoint step plus the short trial length}"
: "${OUTPUT_DIR:?Set a fresh output directory}"
: "${CUDA_VISIBLE_DEVICES:?Set the selected usable GPUs}"
train_index=${M3_WINDOW_INDEX:-$repo_root/derived/m3_fulltrain_c9_20260919/train_c9.pt}
probes_dir=${PROBES_DIR:-$repo_root/derived/m3_player_short_20260919}
python_bin=${PYTHON_BIN:-$repo_root/.venv/bin/python}
IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"

# Capture a true same-checkpoint, same-seed pure-noise baseline before updating.
# Run serially so this evaluator is gone before the eight-rank job starts.
if [[ "${EVALUATE_BASELINE:-1}" == 1 ]]; then
  CUDA_VISIBLE_DEVICES="${devices[0]}" "$python_bin" scripts/evaluate_m3_short_probes.py \
    --checkpoint "$INIT_CHECKPOINT" --probe-manifest "$probes_dir/probes.json" \
    --output-dir "${BASELINE_DIR:-$OUTPUT_DIR/before}" \
    --context-frames 9 --seed 20260919 --denoising-steps 20
fi

exec "$python_bin" -m torch.distributed.run --standalone --nproc-per-node="${#devices[@]}" \
  train_scripts/train_renderer.py \
  --dataset-root "$PLOT_DATASET_ROOT" --train-split train \
  --window-index "$train_index" --val-window-index "$probes_dir/val_id_c9.pt" \
  --visualization-probe-manifest "$probes_dir/probes.json" \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --latent-normalization pixel-vae --warm-start "$INIT_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" --checkpoint-staging-dir "$OUTPUT_DIR/staging" \
  --checkpoint-errors raise \
  --simple-m3 --qk-rms-norm --player-reference-token-grid 4 2 \
  --player-reference-position-encoding --actor-channels 32 \
  --context-frames 9 --cache-frames 32 --block-frames 8 \
  --target-views-per-window 1 --batch-size "${BATCH_SIZE:-4}" \
  --workers "${WORKERS:-2}" --prefetch-factor 2 --gradient-accumulation 1 \
  --steps "$FINAL_STEP" --precision bf16 \
  --loss-mode combined --flow-loss-weight 0 --player-flow-loss-weight 0 \
  --latent-entity-region-upweight 0 --latent-player-region-upweight 0 \
  --pixel-loss-frames 2 --pixel-frame-selection player \
  --player-pixel-l1-weight 0.1 --player-pixel-edge-weight 0.025 \
  --entity-pixel-l1-weight 0 --entity-pixel-edge-weight 0 --health-pixel-l1-weight 0 \
  --player-identity-loss-weight 0 --counterfactual-player-difference-weight 0 \
  --lr 1e-5 --appearance-lr 5e-5 \
  --save-every 100 --validate-every 100 --val-batches 8 \
  --visualize-every 100 --visualization-denoising-steps 20 --log-every 10 \
  --seed 20260919 --wandb-entity "${WANDB_ENTITY:-ckx23-tsinghua-university}" \
  --wandb-project "${WANDB_PROJECT:-plot-m3}" \
  --wandb-name "${WANDB_NAME:-m3-fulltrain-c9-player-pixels-only-8gpu-b4}" \
  --wandb-mode "${WANDB_MODE:-online}" "$@"
