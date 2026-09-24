#!/usr/bin/env bash
set -euo pipefail
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$root/Plot"
run="$root/runs/m4-state/large-100k-20260922-v1"
python="$root/envs/plot-env-py311-cu128/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
rank="${1:?node rank 0 or 1}"
if [[ "$rank" == 0 ]]; then
 "$python" dataset_toolkits/prepare_state_policy.py \
 --dataset-root "$root/data/polis_two_player_fixed_skins_complete_20260917_360p" \
 --output-dir "$run/trainer-check-cache" --vocabulary derived/common/block_vocabulary.json \
 --profile language_builder --train-episodes 8 --val-episodes 2 --windows-per-episode 16 \
 --compact-cache --m3-state --cache-workers 4 \
 --text-catalog "$root/downloads/plot-checkpoints-main/m4_text/text_catalog.json" \
 --text-cache "$root/downloads/plot-checkpoints-main/m4_text/text_cache.safetensors"
else
 deadline=$((SECONDS+1800))
 while [[ ! -f "$run/trainer-check-cache/summary.json" ]]; do
   ((SECONDS<deadline)) || exit 3
   sleep 5
 done
fi
launch=("$python" -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank="$rank"
 --master-addr=29.163.242.17 --master-port=29659 train_scripts/train_state_large.py
 --cache "$run/trainer-check-cache" --output "$run/trainer-check-model"
 --steps 2 --batch-size 4 --workers 2 --warmup 1 --validate-every 2 --checkpoint-every 1 --allow-pilot)
"${launch[@]}"
"${launch[@]}" --resume
echo TRAINER_CHECK_AND_RESUME_PASS
