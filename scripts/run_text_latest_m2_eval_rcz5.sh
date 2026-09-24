#!/usr/bin/env bash
set -euo pipefail

ROOT=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$ROOT/Plot"
CUDA_VISIBLE_DEVICES=0 "$ROOT/envs/plot-env-py311-cu128/bin/python" \
  scripts/evaluate_text_m2_closed_loop.py \
  --episode "$ROOT/data/polis_two_player_fixed_skins_complete_20260917_360p/val_id/polis_two_player_fixed_skins_s02_s07_s08_s09_s10_full_20260916__00002239_S07_seed741492374" \
  --initial-cache "$ROOT/runs/m4-state/large-100k-20260922-v1/cache/language_builder/adaeacfcdb6d04f1ca9f2207.npz" \
  --cache-summary "$ROOT/runs/m4-state/large-100k-20260922-v1/cache/language_builder/summary.json" \
  --m2-checkpoint "$ROOT/runs/m2-player-rollout/formal-128f-1500-v1/best-val-128f.pt" \
  --m4-checkpoint "$ROOT/runs/m4-state/text-hier-pointer-20k-20260923-v1/best-val-loss.pt" \
  --output "$ROOT/runs/m4-state/text-hier-pointer-20k-20260923-v1/eval/m2-128f-closed-and-gt-state-s07-2239-final20k.json" \
  --target 0 \
  --max-blocks 60 \
  --modes m4 m4_gt_state \
  --model-start 0
