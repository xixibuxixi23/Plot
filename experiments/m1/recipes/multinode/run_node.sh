#!/usr/bin/env bash
# Run manually on EACH node; default action only prints a command.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:?Usage: run_node.sh CONFIG NODE_RANK [dry-run|check|smoke|train]}"
NODE_RANK="${2:?Node rank required (see hosts.tsv)}"
MODE="${3:-dry-run}"
source "$CONFIG"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-0}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-0}"
DISCARD_PARTIAL_EPOCH="${DISCARD_PARTIAL_EPOCH:-0}"
for value in "$EVAL_EVERY_STEPS" "$CHECKPOINT_EVERY_STEPS" "$DISCARD_PARTIAL_EPOCH" "$NNODES" "$GPUS_PER_NODE" "$BATCH_PER_GPU" "$MAX_STEPS" "$AUDIT_SAMPLES" "$FIXED_LOSS_SAMPLES" "$WORKERS" "$NODE_RANK" "$MASTER_PORT"; do
  [[ "$value" =~ ^[0-9]+$ ]] || { echo 'Integer configuration required' >&2; exit 2; }
done
(( NNODES > 0 && GPUS_PER_NODE > 0 && BATCH_PER_GPU > 0 && NODE_RANK < NNODES && MASTER_PORT > 0 && MASTER_PORT < 65536 )) || exit 2
(( DISCARD_PARTIAL_EPOCH == 0 || DISCARD_PARTIAL_EPOCH == 1 )) || exit 2
WORLD_SIZE_EXPECTED=$((NNODES * GPUS_PER_NODE))
(( AUDIT_SAMPLES >= WORLD_SIZE_EXPECTED )) || { echo 'AUDIT_SAMPLES must be >= total GPUs' >&2; exit 2; }
case "$MODE" in dry-run|check|smoke|train) ;; *) echo 'Unknown mode' >&2; exit 2;; esac
export EVAL_EVERY_STEPS CHECKPOINT_EVERY_STEPS DISCARD_PARTIAL_EPOCH
export CACHE_DIR PERSIST_ROOT RESUME_CHECKPOINT OUTPUT_DIR MAX_STEPS AUDIT_SAMPLES
export WORLD_SIZE_EXPECTED GPUS_PER_NODE PLOT_ROOT BATCH_PER_GPU FIXED_LOSS_SAMPLES WORKERS
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
PROBE="$HERE/preflight.py"
LAUNCH=("$PYTHON_BIN" -m torch.distributed.run --nnodes "$NNODES" --nproc-per-node "$GPUS_PER_NODE" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --max-restarts 0)
TRAIN=("$PLOT_ROOT/experiments/m1/train_multiview_persist_full.py" --phase train --objective flow --time-sampling original --cache "$CACHE_DIR" --persist "$PERSIST_ROOT" --resume "$RESUME_CHECKPOINT" --output "$OUTPUT_DIR" --max-steps "$MAX_STEPS" --batch-size "$BATCH_PER_GPU" --workers "$WORKERS" --audit-samples "$AUDIT_SAMPLES" --fixed-loss-samples "$FIXED_LOSS_SAMPLES" --eval-every-steps "$EVAL_EVERY_STEPS" --checkpoint-every-steps "$CHECKPOINT_EVERY_STEPS")
echo "node=$NODE_RANK total_gpus=$WORLD_SIZE_EXPECTED global_batch=$((WORLD_SIZE_EXPECTED * BATCH_PER_GPU))"
if [[ "$MODE" == dry-run ]]; then printf '%q ' "${LAUNCH[@]}" "${TRAIN[@]}"; printf '\n'; exit 0; fi
if [[ "$MODE" == check ]]; then exec "$PYTHON_BIN" "$PROBE" local; fi
[[ "$MASTER_ADDR" != REPLACE_WITH_PRIVATE_IP ]] || { echo 'Set an inter-node reachable private training IP' >&2; exit 2; }
cd "$PLOT_ROOT"
if [[ "$MODE" == smoke ]]; then exec "${LAUNCH[@]}" "$PROBE" collective; fi
exec "${LAUNCH[@]}" "$PROBE" train
