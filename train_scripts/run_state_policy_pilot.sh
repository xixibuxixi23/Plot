#!/usr/bin/env bash
set -euo pipefail
node="${1:?node required: rcz3 through rcz6}"
case "$node" in
  rcz3) profile=villager_peaceful ;;
  rcz4) profile=zombie_melee ;;
  rcz5) profile=skeleton_swordsman ;;
  rcz6) profile=villager_defender ;;
  *) echo "Unexpected node label: $node" >&2; exit 2 ;;
esac
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$root/Plot"
python="$root/envs/plot-env-py311-cu128/bin/python"
run_id="${2:?unique run id required}"
cache="$root/runs/m4-state/$run_id/cache/$profile"
output="$root/runs/m4-state/$run_id/models/$profile"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
echo "NODE=$node PROFILE=$profile RUN=$run_id"
if [[ ! -f "$cache/summary.json" ]]; then
  "$python" dataset_toolkits/prepare_state_policy.py \
    --dataset-root "$root/data/polis_two_player_fixed_skins_complete_20260917_360p" \
    --vocabulary derived/common/block_vocabulary.json \
    --output-dir "$cache" --profile "$profile" \
    --train-episodes 8 --val-episodes 3 --windows-per-episode 8
fi
"$python" -m pytest -q tests/test_state_policy.py
"$python" train_scripts/train_state_policy.py --cache-dir "$cache" \
  --output-dir "$output" --profile "$profile" --steps 200 --batch-size 4 \
  --validate-every 50 --workers 0 --device cuda --precision bf16
echo "PILOT_COMPLETE=$profile"
