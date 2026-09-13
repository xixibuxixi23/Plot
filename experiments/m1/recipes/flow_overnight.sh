#!/usr/bin/env bash
set -euo pipefail
PLOT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PLOT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
exec .venv/bin/torchrun --standalone --nproc_per_node=8 \
  experiments/m1/train_multiview_persist_full.py --phase train --objective flow \
  --resume outputs/m1_multiview_persist_gtcam_s01_all_v1/checkpoint_latest.pt \
  --max-steps 1000000 --fixed-loss-samples 128 --audit-samples 32 \
  --output outputs/m1_flow_full_1000k_v1 "$@"
