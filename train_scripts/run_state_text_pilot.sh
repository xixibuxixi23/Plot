#!/usr/bin/env bash
set -euo pipefail
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$root/Plot"
python="$root/envs/plot-env-py311-cu128/bin/python"
run="$root/runs/m4-state/text-pilot-20260922-v1"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
"$python" -m pytest -q tests/test_state_policy.py
if [[ ! -f "$run/cache/summary.json" ]]; then
  "$python" dataset_toolkits/prepare_state_policy.py \
    --dataset-root "$root/data/polis_two_player_fixed_skins_complete_20260917_360p" \
    --vocabulary derived/common/block_vocabulary.json --output-dir "$run/cache" \
    --profile language_builder --train-episodes 256 --val-episodes 64 \
    --windows-per-episode 16 --shuffle-episodes --seed 42 --compact-cache --cache-workers 8 \
    --text-catalog "$root/downloads/plot-checkpoints-main/m4_text/text_catalog.json" \
    --text-cache "$root/downloads/plot-checkpoints-main/m4_text/text_cache.safetensors"
fi
"$python" train_scripts/train_state_policy.py --cache-dir "$run/cache" \
  --output-dir "$run/model" --profile language_builder --steps 3000 --batch-size 8 \
  --validate-every 500 --workers 4 --device cuda --precision bf16 --train-eval-limit 1024
echo TEXT_PILOT_COMPLETE
