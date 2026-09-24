#!/usr/bin/env bash
set -euo pipefail

node="${1:?rcz5 or rcz6 required}"
mode="${2:-train}"
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
repo="$root/Plot"
text_cache="$root/runs/m4-state/large-100k-20260922-v1/cache/language_builder"
zombie_cache="$root/runs/m4-state/large-100k-20260922-v1/cache/zombie_melee"
text_run="$root/runs/m4-state/text-v2-100k-20260922-v1"
run="$root/runs/m4-state/unified-v4-100k-20260922-v1"
initialize="$text_run/best-val-loss.pt"

case "$node" in
  rcz5) node_rank=0 ;;
  rcz6) node_rank=1 ;;
  *) echo "expected rcz5 or rcz6" >&2; exit 2 ;;
esac

if [[ "$mode" == wait ]]; then
  echo "waiting for $text_run/COMPLETE.json"
  while [[ ! -f "$text_run/COMPLETE.json" ]]; do sleep 60; done
  mode=train
fi

[[ -f "$text_cache/summary.json" ]] || { echo "missing full text cache" >&2; exit 3; }
[[ -f "$zombie_cache/summary.json" ]] || { echo "missing full zombie cache" >&2; exit 3; }
[[ -f "$initialize" ]] || { echo "missing text v2 initializer" >&2; exit 3; }

cd "$repo"
python="$root/envs/plot-env-py311-cu128/bin/python"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=WARN
launch=("$python" -m torch.distributed.run
  --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank"
  --master-addr=29.163.242.17 --master-port=29665)
args=(train_scripts/train_state_text_v2.py
  --cache "$text_cache" --zombie-cache "$zombie_cache" --output "$run"
  --steps 100000 --batch-size 4 --workers 4
  --hidden 640 --heads 10 --depth 8 --dropout 0.1
  --lr 6e-5 --weight-decay 0.05 --warmup 1000
  --validate-every 5000 --checkpoint-every 1000
  --positive-weight 6 --place-positive-weight 4
  --move-key-weight 1 --attack-weight 2.5 --place-weight 2
  --mouse-smoothing 0.2 --mouse-move-positive-weight 2
  --horizon-min-weight 0.3 --skip-baseline
  --early-stop-patience 4 --early-stop-min-step 20000)

if [[ "$mode" == resume ]]; then
  args+=(--resume)
elif [[ "$mode" == train ]]; then
  if [[ "$node_rank" == 0 ]]; then
    [[ ! -e "$run" ]] || { echo "output already exists; use resume" >&2; exit 4; }
  fi
  args+=(--initialize "$initialize")
else
  echo "mode must be train, wait, or resume" >&2
  exit 2
fi

exec "${launch[@]}" "${args[@]}"
