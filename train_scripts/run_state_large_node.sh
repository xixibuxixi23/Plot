#!/usr/bin/env bash
set -euo pipefail
node="${1:?rcz3 through rcz6 required}"
mode="${2:?smoke, train or resume}"
root=/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d
cd "$root/Plot"
run="$root/runs/m4-state/large-100k-20260922-v1"
case "$node" in
 rcz3) profile=zombie_melee; node_rank=0; master=29.163.224.197; port=29631 ;;
 rcz4) profile=zombie_melee; node_rank=1; master=29.163.224.197; port=29631 ;;
 rcz5) profile=language_builder; node_rank=0; master=29.163.242.17; port=29651 ;;
 rcz6) profile=language_builder; node_rank=1; master=29.163.242.17; port=29651 ;;
 *) exit 2 ;;
esac
# These addresses were read inside the exact taiji_client nodes, never used for SSH.
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_SOCKET_IFNAME=bond1 GLOO_SOCKET_IFNAME=bond1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
python="$root/envs/plot-env-py311-cu128/bin/python"
launch=("$python" -m torch.distributed.run --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank"
        --master-addr="$master" --master-port="$port")
if [[ "$mode" == smoke ]]; then
 "${launch[@]}" scripts/smoke_large_distributed.py --profile "$profile" \
    --output "$run/smoke-$profile.json" --steps 3 --batch-size 1
elif [[ "$mode" == train || "$mode" == resume ]]; then
 [[ -f "$run/smoke-$profile.json" ]] || { echo 'Missing successful 16-rank smoke'; exit 3; }
 deadline=$((SECONDS+86400))
 while [[ ! -f "$run/cache/$profile/summary.json" ]]; do
   if grep -a -q 'Traceback (most recent call last)' "$run/logs/prepare-$profile.log"; then
     echo 'Data preparation failed; refusing to train'; exit 4
   fi
   (( SECONDS < deadline )) || { echo 'Data preparation timeout'; exit 5; }
   sleep 30
 done
 "$python" scripts/verify_state_large_cache.py --cache "$run/cache/$profile" --profile "$profile"
 extra=()
 if [[ "$mode" == resume ]]; then extra=(--resume); fi
 "${launch[@]}" train_scripts/train_state_large.py --cache "$run/cache/$profile" \
    --output "$run/models/$profile" --steps 100000 --batch-size 4 --workers 4 \
    --hidden 768 --heads 12 --depth 12 --validate-every 5000 --checkpoint-every 1000 \
    --warmup 1000 --positive-weight 4 --key-weight 2 "${extra[@]}"
else
 exit 2
fi
