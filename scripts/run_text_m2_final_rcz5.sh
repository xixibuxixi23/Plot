#!/usr/bin/env bash
set -euo pipefail

ROOT=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
RUN="$ROOT/runs/m4-state/text-m2-final-20260924-v1"
cd "$ROOT/Plot"
CUDA_VISIBLE_DEVICES=0 "$ROOT/envs/plot-env-py311-cu128/bin/python" \
  train_scripts/train_text_m2_group_relative.py \
  --cache "$ROOT/runs/m4-state/large-100k-20260922-v1/cache/language_builder" \
  --m2-index "$ROOT/runs/m2-player-rollout/index.json" \
  --m2-checkpoint "$ROOT/runs/m2-edit/context-countw4-500-20260924-v1/checkpoint.pt" \
  --m4-checkpoint "$ROOT/runs/m4-state/text-m2-group-relative-s01-e512-100k-20260924-v1/latest.pt" \
  --output "$RUN" \
  --steps 100000 \
  --episodes 512 \
  --episode-substring _S01_ \
  --windows-per-episode 64 \
  --group-size 16 \
  --lr 3e-6 \
  --wrong-weight 5 \
  --hard-correct-bonus 2 \
  --hard-wrong-penalty 5 \
  --distance-weight 1 \
  --aim-weight 0.25 \
  --kl-weight 0.05 \
  --bc-weight 0.05 \
  --pointer-weight 0.05 \
  --min-reward-std 0.01 \
  --save-every 5000
