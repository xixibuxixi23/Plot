#!/usr/bin/env bash
set -euo pipefail

node="${1:?rcz5 or rcz6 required}"
mode="${2:-train}"
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
repo="$root/Plot"
cache="$root/runs/m4-state/large-100k-20260922-v1/cache/language_builder"
run="$root/runs/m4-state/text-v2-100k-20260922-v1"
case "$node" in
  rcz5) node_rank=0 ;;
  rcz6) node_rank=1 ;;
  *) echo "expected rcz5 or rcz6" >&2; exit 2 ;;
esac

cd "$repo"
python="$root/envs/plot-env-py311-cu128/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
launch=("$python" -m torch.distributed.run
  --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank"
  --master-addr=29.163.242.17 --master-port=29655)
args=(train_scripts/train_state_text_v2.py
  --cache "$cache" --output "$run"
  --steps 100000 --batch-size 4 --workers 4
  --hidden 640 --heads 10 --depth 8 --dropout 0.1
  --lr 1e-4 --weight-decay 0.05 --warmup 1000
  --validate-every 5000 --checkpoint-every 1000
  --positive-weight 4 --place-positive-weight 4
  --move-key-weight 1 --attack-weight 2 --place-weight 2
  --mouse-smoothing 0.2 --mouse-move-positive-weight 2
  --horizon-min-weight 0.25
  --early-stop-patience 3 --early-stop-min-step 10000)
[[ -f "$cache/summary.json" ]] || { echo "missing complete text cache" >&2; exit 3; }
if [[ "$mode" == resume ]]; then
  args+=(--resume)
elif [[ "$mode" != train ]]; then
  echo "mode must be train or resume" >&2
  exit 2
fi
exec "${launch[@]}" "${args[@]}"
