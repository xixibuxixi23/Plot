# M4 implementation and data selection

The default M4 is `plot/models/inserted_policy.py`. Its deployment contract is fixed:
eight completed frames and their incoming actions produce one eight-action
chunk. The simulator executes all eight actions before M4 is called again. It
does not run one-frame receding-horizon prediction.

## Architecture

M3 is loaded from a selected checkpoint and completely frozen. During training,
it performs a clean zero-noise-time causal forward over a window of up to 65
completed latent frames. `policy_indices` selects one consecutive eight-frame
block from that context, and one target
resident policy token is inserted after each of the last four M3 DiT blocks.
Each inserted block follows the original 2DAction policy design:

1. the policy token cross-attends to every detached spatial M3 patch in its own
   frame;
2. causal temporal attention lets its eight tokens read earlier completed
   frames;
3. it cross-attends to the frozen T5-v1.1-base encoding of current task text;
4. an MLP updates only the policy branch.

The selected spatial features therefore retain M3 history before the eight-frame
M4 block. M3 video tokens never read policy tokens. No M4 loss reaches M3. The final
policy token summarizes the completed chunk and predicts all eight next actions.
M4 does not need its own KV cache because each call consumes exactly the previous
completed chunk. M3 retains its causal KV cache for rendering and exposes the
last four cached spatial layers directly to M4, without re-encoding the chunk in
isolation.

There are three complete, independent parameter families: `language_builder`,
`villager`, and `combat`. Profiles route to language builder, peaceful villager,
zombie melee, skeleton swordsman, or villager defender behavior. Profile
embeddings specialize behavior within a family. There is no absolute resident
ID embedding, and policy tokens from different residents do not communicate.
They can still react to other residents through the visual and structured M3
conditions visible to the target resident.

Language builders receive two frozen T5 conditions. Shared task text is pooled
into the initial policy token; current task text is cross-attended in every
inserted block. Non-language families have text masks forced off. The current
formal release records shared task descriptions but no distinct per-resident
subtask, so its current text falls back to the shared text. Future data can put
`agent_task_texts` or `agent_subtasks` in the manifest without changing M4.

`plot/models/structured_action.py` matches the original structured head:

- independent BCE logits for ten supported compatible keys;
- one categorical choice over no hotbar change or slots 1--9;
- separate 17-bin categorical horizontal and vertical mouse deltas.

Zoom and inventory are excluded from the learned head. Peaceful villagers have
hard constraints against dig, place, drop and hotbar actions. Horizon losses are
weighted 8 through 1 so the immediately executed part of a chunk has the
strongest supervision.

## Formal data

The canonical release is:

```text
$PLOT_DATASET_ROOT
```

Build the targeted index and deterministic text catalog with:

```bash
.venv/bin/python dataset_toolkits/build_m4_index.py \
  $PLOT_DATASET_ROOT \
  --output-dir derived/m4 --workers 32 \
  --max-windows-per-agent 12 --max-wait-windows-per-agent 2

.venv/bin/python dataset_toolkits/build_m4_text_catalog.py \
  derived/m4/train.jsonl \
  derived/m4/val_id.jsonl \
  --output derived/m4/text_catalog.json
```

The adapter loads a 65-frame M3 context and marks the eight policy observations
`o[t-7:t+1]` inside it. Their incoming actions are `a[t-8:t]`, and the targets
are `a[t:t+8]`. Early episode decisions select positions 1--8 of the first
65-frame window; later decisions receive up to 64 earlier M3 frames. It excludes invalid HP, dead residents,
termination crossings and policy-mask gaps. It retains every legal attack/edit
window, sparsely retains intentional wait, and spreads ordinary windows across
each trajectory. S02/S04/S06 have no supported M4 family and remain outside M4
training. Weighted sampling balances profiles and prioritizes attack/edit.

The current indices contain 1,058,239 train and 61,196 val-ID windows. Train has
890,655 builder, 29,475 peaceful-villager and 138,109 combat windows, including
31,210 direct attack windows. Detailed counts are in
`derived/m4/summary.json`.

The checked-in catalog and local cached embeddings under `checkpoints/m4_text`
remove the need to load T5 during M4 training:

```bash
torchrun --standalone --nproc_per_node=8 train_scripts/train_policy.py \
  --dataset-root /path/to/polis_v1_20260909_360p \
  --train-index derived/m4/train.jsonl \
  --val-index derived/m4/val_id.jsonl \
  --vocabulary derived/common/block_vocabulary.json \
  --text-catalog checkpoints/m4_text/text_catalog.json \
  --text-cache checkpoints/m4_text/text_cache.safetensors \
  --m3-checkpoint /path/to/final_m3.pt \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --output-dir outputs/m4_inserted
```

The trainer supports DDP, validation, atomic family-only checkpoints, exact
resume, and balanced or uniform sampling. It encodes RGB with the frozen Pixel
VAE at training time. The old pooled 128-D model, dataset, cache builder and
trainer are grouped under `ablations/m4_sidecar/`; they are excluded from the
production API and retained only as an ablation.

## Closed loop

`plot/pipelines/closed_loop_pipeline.py` enforces the block order:

1. M4 reads the previous completed eight-frame latent/state block and emits the
   next complete action chunk for model-controlled residents. Human slots keep
   externally supplied actions.
2. M2 predicts all eight residents' updates and sparse writes jointly.
3. `TransitionCommitter` applies each transition in stable slot order.
4. M1 fills newly entered unknown 48-cubed regions before each M3 snapshot.
5. M3 generates and commits eight completed frames. The last four spatial-layer
   tensors from this cached commit become the next M4 visual input.

An external first frame cannot satisfy an eight-frame M4 input. The first action
chunk is therefore externally provided; from the second chunk onward M4 always
receives exactly the previous completed block.
