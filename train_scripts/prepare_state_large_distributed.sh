#!/usr/bin/env bash
set -euo pipefail
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$root/Plot"
profile="${1:?profile}"
shard_index="${2:?shard index}"
shard_count="${3:?shard count}"
shift 3
run="$root/runs/m4-state/large-100k-20260922-v1"
extra=()
if [[ "$profile" == language_builder ]]; then
  extra=(--text-catalog "$root/downloads/plot-checkpoints-main/m4_text/text_catalog.json"
         --text-cache "$root/downloads/plot-checkpoints-main/m4_text/text_cache.safetensors")
fi
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$root/envs/plot-env-py311-cu128/bin/python" dataset_toolkits/prepare_state_policy.py \
  --dataset-root "$root/data/polis_two_player_fixed_skins_complete_20260917_360p" \
  --output-dir "$run/cache/$profile" --vocabulary derived/common/block_vocabulary.json \
  --profile "$profile" --all-data --m3-state --cache-workers 16 --resume-existing \
  --cache-shards "$shard_count" --cache-shard-index "$shard_index" "${extra[@]}" "$@"
