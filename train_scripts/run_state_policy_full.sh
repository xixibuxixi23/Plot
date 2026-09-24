#!/usr/bin/env bash
set -euo pipefail
node="${1:?node required}"
case "$node" in
  rcz3) profile=villager_peaceful; extra=(--attack-positive-weight 1) ;;
  rcz4) profile=zombie_melee; extra=(--balance-attack --attack-positive-weight 8) ;;
  rcz5) profile=skeleton_swordsman; extra=(--balance-attack --attack-positive-weight 8) ;;
  rcz6) profile=villager_defender; extra=(--balance-attack --attack-positive-weight 8) ;;
  *) echo "Unexpected node: $node" >&2; exit 2 ;;
esac
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$root/Plot"
python="$root/envs/plot-env-py311-cu128/bin/python"
run_id="${2:?unique run id required}"
cache="$root/runs/m4-state/$run_id/cache/$profile"
output="$root/runs/m4-state/$run_id/models/$profile"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0
echo "NODE=$node PROFILE=$profile RUN=$run_id"
"$python" -m pytest -q tests/test_state_policy.py
if [[ ! -f "$cache/summary.json" ]]; then
  "$python" dataset_toolkits/prepare_state_policy.py \
    --dataset-root "$root/data/polis_two_player_fixed_skins_complete_20260917_360p" \
    --vocabulary derived/common/block_vocabulary.json \
    --output-dir "$cache" --profile "$profile" \
    --all-data --seed 42 --cache-workers 8
fi
"$python" train_scripts/train_state_policy.py --cache-dir "$cache" \
  --output-dir "$output" --profile "$profile" --steps 10000 --batch-size 8 \
  --validate-every 1000 --workers 4 --device cuda --precision bf16 \
  --train-eval-limit 1024 "${extra[@]}"
echo "FULL_10K_COMPLETE=$profile"
