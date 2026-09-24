#!/usr/bin/env bash
set -euo pipefail

node="${1:?rcz3 or rcz4 required}"
mode="${2:?smoke or train required}"
resume="${3:-${M1_RESUME:-}}"
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
plot="$root/Plot"
run="${M1_OUTPUT_DIR:-$root/runs/m1-unified/maskfill-50k}"
case "$node" in
  rcz3) node_rank=0 ;;
  rcz4) node_rank=1 ;;
  *) echo "Expected rcz3 or rcz4" >&2; exit 2 ;;
esac
case "$mode" in smoke|train) ;; *) echo "Expected smoke or train" >&2; exit 2 ;; esac

cd "$plot"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
python="$root/envs/plot-env-py311-cu128/bin/python"
launch=("$python" -m torch.distributed.run --nnodes=2 --nproc-per-node=8
        --node-rank="$node_rank" --master-addr=29.163.224.197 --master-port=29614
        --max-restarts=0)
common=(train_scripts/train_fill.py
        --dataset-root "$root/data/polis_two_player_fixed_skins_complete_20260917_360p"
        --vocabulary derived/common/block_vocabulary.json
        --episode-index derived/m1/s01_episode_index.json
        --samples-per-agent 4 --frontier-sampling
        --frontier-image-probability 0.3333333333
        --base-channels 32 --precision bf16 --wandb-mode offline)
if [[ -n "$resume" ]]; then
  common+=(--resume "$resume")
fi

if [[ "$mode" == smoke ]]; then
  exec "${launch[@]}" "${common[@]}" \
    --output-dir "$root/runs/m1-unified/maskfill-ddp-smoke" \
    --batch-size 8 --num-workers 1 --steps 3 --save-every 0 --eval-every 0 \
    --visualize-every 0 --log-every 1 --wandb-name m1-unified-maskfill-ddp-smoke
fi

exec "${launch[@]}" "${common[@]}" --output-dir "$run" \
  --batch-size 8 --num-workers 2 --steps 50000 --save-every 1000 \
  --eval-every 500 --visualize-every 2000 --log-every 20 \
  --wandb-name m1-unified-maskfill-50k
