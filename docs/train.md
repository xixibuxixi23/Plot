# Training

Run from the repository root:

```bash
source .venv/bin/activate
export PLOT_DATASET_ROOT=/path/to/polis_v1_20260909_360p
```

For multi-GPU jobs, replace `python` with
`torchrun --standalone --nproc_per_node=8`. Batch sizes are per process.

## M1

```bash
python train_scripts/train_fill.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --vocabulary derived/common/block_vocabulary.json \
  --episode-index derived/m1/s01_episode_index.json \
  --output-dir outputs/m1 \
  --samples-per-agent 4 \
  --frontier-sampling \
  --frontier-image-probability 0.3333333333
```

This is the only M1 architecture. One optional-image `FillNetwork` handles
image-conditioned initialization, geometry-only frontier completion, and
image-assisted frontier completion. With four deterministic slots per
resident, the command above uses a 25/50/25 percent mixture. Known voxels are
conditioning only; loss and commits are restricted to `fill_mask`.

## M2

```bash
python dataset_toolkits/build_episode_index.py "$PLOT_DATASET_ROOT" \
  --output derived/m2/episode_index.json
python train_scripts/train_transition_full.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --index derived/m2/episode_index.json \
  --vocabulary derived/common/block_vocabulary.json \
  --cache-root /fast/cache/plot-m2 --output-dir outputs/m2
```

M2 sees known 8-step actions with bidirectional attention. The proposal handles
key-driven kinematics; the network predicts residual motion and sparse 13³
writes. HP is currently a condition and is not supervised from ambiguous raw
damage events.

## M3

```bash
export PLOT_CHECKPOINT_STAGING_DIR=/fast/local/or/pfs/plot-checkpoints
bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh
```

Each sample has one known plus 64 unknown frames. Flow loss covers all unknown
frames with a causal mask. Two player views are sampled and flattened into
`batch × player`. Evaluation generates eight frames per chunk, retains a
32-frame KV cache and rolls out 64 frames. Instance masks weight entity pixels
in the loss and are not inputs. Fixed combat, multiplayer, building and NPC
probes are logged to W&B every 1000 steps.

Use local SSD or PFS for `--checkpoint-staging-dir`. Direct safetensors writes
to OSS FUSE can fail with `Device or resource busy`; staging avoids that file
operation limitation.

## M4

```bash
export PLOT_M3_CHECKPOINT=/path/to/m3/checkpoint.pt
bash train_scripts/recipes/m4/train_m4_8gpu.sh
```

M4 freezes M3 and inserts read-only actor tokens in its final blocks. It reads
eight completed frames from the 65-frame causal context and predicts one
structured eight-action chunk. Balanced sampling retains all attack/edit
windows, limits common waits and balances builder, villager and combat profiles.

M1/M3/M4 accept `--resume`; M2 currently writes one replaceable checkpoint.
W&B is rank-zero only where DDP is supported. API keys belong in environment
variables and must not be stored in this repository.
