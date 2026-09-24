#!/usr/bin/env bash
set -euo pipefail

ROOT=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$ROOT/Plot"
CUDA_VISIBLE_DEVICES=0 "$ROOT/envs/plot-env-py311-cu128/bin/python" \
  scripts/evaluate_text_pointer_set_accuracy.py \
  --cache "$ROOT/runs/m4-state/large-100k-20260922-v1/cache/language_builder" \
  --checkpoint "$ROOT/runs/m4-state/text-hier-pointer-20k-20260923-v1/best-val-loss.pt" \
  --batch-size 16 \
  --output "$ROOT/runs/m4-state/text-hier-pointer-20k-20260923-v1/eval/pointer-any-future-goal-val.json"
