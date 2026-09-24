#!/usr/bin/env bash
set -euo pipefail

node="${1:?rcz3 or rcz4 required}"
mode="${2:-train}"
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
repo="$root/Plot"
cache="$root/runs/m4-state/large-100k-20260922-v1/cache/zombie_melee"
run="$root/runs/m4-state/zombie-v3-combat-100k-20260922-v1"
initialize="$root/runs/m4-state/zombie-v2-20k-20260922-v1/best-val-loss.pt"

case "$node" in
  rcz3) node_rank=0 ;;
  rcz4) node_rank=1 ;;
  *) echo "expected rcz3 or rcz4" >&2; exit 2 ;;
esac

cd "$repo"
python="$root/envs/plot-env-py311-cu128/bin/python"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=bond1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

launch=("$python" -m torch.distributed.run
  --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank"
  --master-addr=29.163.224.197 --master-port=29648)
args=(train_scripts/train_state_zombie_v2.py
  --model-version v3 --cache "$cache" --output "$run"
  --steps 100000 --batch-size 4 --workers 4
  --hidden 640 --heads 10 --depth 8 --dropout 0.1
  --lr 6e-5 --weight-decay 0.05 --warmup 500
  --validate-every 2000 --checkpoint-every 1000
  --positive-weight 8 --move-key-weight 1 --attack-weight 3
  --locomotion-weight 1 --combat-range-weight 0.5 --combat-bearing-weight 0.25
  --attack-range 3.25
  --mouse-smoothing 0.2 --mouse-move-positive-weight 2
  --horizon-min-weight 0.35
  --early-stop-patience 4 --early-stop-min-step 12000)

[[ -f "$cache/summary.json" ]] || { echo "missing full zombie cache" >&2; exit 3; }
if [[ "$mode" == resume ]]; then
  args+=(--resume)
elif [[ "$mode" == train ]]; then
  [[ -f "$initialize" ]] || { echo "missing v2 initializer" >&2; exit 3; }
  args+=(--initialize "$initialize")
else
  echo "mode must be train or resume" >&2
  exit 2
fi

exec "${launch[@]}" "${args[@]}"
