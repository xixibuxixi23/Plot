# M3 Renderer implementation

The architecture follows `method.md`. The initial implementation reuses a
namespaced copy of `3d2d-world-model/2daction/persist` under
`plot/models/renderer_backbone/`; it does not import the sibling project's global
`models` or `trainers` packages.

## Implemented interfaces

- `plot/models/renderer.py`: shared causal Pixel DiT, voxel class embedding with a
  separate UNFILLED entry, four-view appearance, relative resident geometry,
  HP, orientation, held item, resident type, incoming actions and event cues.
- `plot/models/renderer_codec.py`: frozen Pixel VAE; RGB is `[B,T,3,360,640]` in
  `[0,1]`, latent is `[B,T,16,36,64]`. Encoding uses the posterior mean.
- `plot/data/renderer_dataset.py`: accepted continuous TextAgent episodes
  become 65-frame clips. Two valid target residents are sampled from each
  window and flattened into the view batch; every view still receives the
  complete state and appearance of all residents.
- `plot/training/renderer_trainer.py`: causal diffusion forcing over one known frame
  and all 64 unknown frames, plus eight-frame rollout with persistent per-view
  KV caches.
- `plot/pipelines/renderer_pipeline.py`: capture the eight successive committed
  memory states under a fixed crop anchor before rendering a block.

The neural camera condition is `camera_world - player_world`. The raster adapter
uses the actual crop anchor solely to calculate the W2C transform; it accounts
for the half-open cube's geometric center at `anchor - 0.5`. Raw camera position
is never mistaken for the extrinsic translation `-R @ C`.

The loader reads observation-aligned `player_health` and `entity_weapon_name`,
and incoming `action_continuous[t]` conditions RGB at observation `t+1`. It does
not reuse the pre-action inventory field as the next observation's held item.
HP/profile/text/center conventions follow the method. The first clip frame has
a learned action-prefix indicator. World-relative angles are supplied in radians.

Raw uint16 instance masks provide both latent-resolution weights and exact
full-resolution entity masks; neither is a neural input. Four-view PNGs are
mandatory. Missing crop coverage has an explicit unknown embedding; it is not
silently converted to air or filled with future observations. Live rendering
requires M1 to fill all unknown cells first.

## Training

Use the existing PLOT CUDA environment. The renderer additionally requires
`einops`, `timm`, `loguru`, `safetensors`, the existing 2DAction-compatible
`utils3d`, and `nvdiffrast`. CPU contract tests use supplied depth samples; raw
voxel projection uses CUDA. The `renderer` extra declares the ordinary PyPI
dependencies; retain the working 2DAction geometry packages in the environment.

```bash
cd /path/to/Plot
torchrun --standalone --nproc_per_node=8 train_scripts/train_renderer.py \
  --dataset-root "$PLOT_DATASET_ROOT" \
  --window-index derived/m3/train_c65.pt \
  --val-window-index derived/m3/val_id_c65.pt \
  --vocabulary derived/common/block_vocabulary.json \
  --pixel-vae checkpoints/pixel_vae/model.safetensors \
  --backbone-checkpoint checkpoints/m3_backbone/model.safetensors \
  --output-dir outputs/m3_formal \
  --context-frames 65 --cache-frames 32 --target-views-per-window 2 \
  --batch-size 1 --steps 10000
```

The entry point encodes RGB with a frozen VAE and predicts all 64 unknown frames
in one causal forward. By default, each target view selects an entity-rich frame
and an HP-informative frame, reconstructs their clean latent estimates, and
decodes them through the frozen VAE with gradients retained to M3. The auxiliary
objective contains RGB L1 in
the exact full-resolution entity mask, spatial-gradient L1 around entity edges,
and RGB L1 over the Minecraft heart bar (`x=190:314, y=300:322` at 640x360).
The default objective is
`flow + 0.5*entity_L1 + 0.2*entity_edge + 1.0*health_L1`; the VAE stays frozen.
Within a batch, damaged heart bars receive five times the weight of full-health
bars. A separately generated health-focus index can repeat windows containing a
non-full-health target and guarantees that target is one of the selected views.
The old latent entity-region upweight is disabled by default so that it does not
double-count the new pixel objective. Configure these terms with
`--pixel-loss-frames`, `--entity-pixel-l1-weight`,
`--entity-pixel-edge-weight`, `--health-pixel-l1-weight`, and
`--latent-entity-region-upweight`. Use `scripts/build_m3_health_focus_index.py`,
then pass `--health-focus-index` and `--health-focus-oversample 5` for balanced
heart-state training. Checkpoints save model/optimizer/config state.
Each source batch row is one episode
window; its two randomly selected target views are flattened before VAE and M3
execution. Only accepted `train` manifests are read;
item dictionaries must agree across episodes. It rejects clips with missing HP
or a terminated transition before the last target, and does not train past a
target resident's death. Public files remain continuous and unmodified.

`--backbone-checkpoint` optionally transfers shape-compatible 2DAction DiT
weights and reports every missing/skipped key. This is an initialization, not
a full resume: PLOT's new resident encoder and block vocabulary embedding need
training. The VAE checkpoint must match the 2DAction default ViTVaeArgs.

The initial full configuration is 1024 width / 12 layers with a 32-frame
inference cache. Training uses exactly 65 frames: frame zero is clean and all
remaining frames receive independent flow timesteps and loss. These are pilot
settings, not frozen experimental choices.
`train_renderer.py` supports torchrun DDP, gradient accumulation, validation,
staged and verified checkpoints, and exact optimizer/model resume.

Every 1,000 steps, rank zero runs five fixed val-ID deployment rollouts and
uploads side-by-side GT / prediction / absolute-error MP4s to W&B. The probes
cover two-player construction (S01), four-player motion (S06), PvE combat
(S08), three-resident combat (S09), and mixed building/combat with NPCs (S10).
Their exact episode, start frame and target resident are frozen in
`visualizations/probes.json`; the model sees one known frame and generates the
next 64 as eight cached eight-frame chunks, so these are not teacher-forced
previews.
Visualization inherits `--precision`: the VAE, denoiser, and temporal KV cache
all use BF16 when training uses BF16, while quality metrics are accumulated in
FP32. Configure probes with `--visualize-every`, `--visualization-denoising-steps`
and the `--wandb-*` flags.

B200 environment and OSS-safe checkpoint instructions are in
[`m3_b200.md`](m3_b200.md). In particular, set `--checkpoint-staging-dir` to
node-local storage or PFS when `--output-dir` is on an OSS FUSE mount.

## Inference integration

1. Encode the externally supplied first RGB with `RendererCodec.encode` under the
   same BF16/FP32 autocast mode used for training. `ClosedLoopPipeline` defaults
   to BF16 and keeps its input noise, latent, VAE decode and KV cache in that mode.
2. `RendererRollout.start(first_latent, first_conditions)` initializes and
   prefills independent caches for the batch's target residents.
3. For each M2 transition, commit writes, run required M1 fill, then call
   `RendererMemoryBlock.append` to copy that observation's state. Keep the
   anchor fixed for all eight steps. Join `block.conditions()` with the eight
   observation-aligned resident conditions; neither masks nor anchors belong
   in the network condition dictionary.
4. `RendererRollout.generate(noise, conditions)` accepts exactly eight noisy
   latent frames. Denoising reads history without writing it. A separate clean
   t=0 pass commits the completed frames once. Decode the result with the codec.
   `generate_64` applies this operation eight times while retaining only the
   most recent 32 committed frames in KV cache.
5. Retain batch resident ordering across blocks. Start a new rollout when the
   episode/slot identity changes; changing the crop anchor alone retains cache.

`ClosedLoopPipeline` connects the implemented M4, M2 committer, M1 fill and this
renderer. Long-rollout quality and full-size training throughput still require
trained formal checkpoints; code-level closure is covered by contract tests.

## Validation

```bash
.venv/bin/python -m pytest -q tests
PYTHONPATH=../textagent/src .venv/bin/python -m pytest -q \
  ../textagent/tests/test_hostile_branches.py \
  ../textagent/tests/test_scenario_protocols.py
```

Tests check future-frame invariance with nonzero model outputs, training
gradients, world-translation invariance, independent committed history across
multiple eight-frame blocks, memory snapshot copies, data split/time alignment,
uint16-mask weighting, and CUDA voxel-projection gradients when CUDA is present.
