# M3 QK player-pixel refinement

Continue the completed `okd5jze8` checkpoint at global step 40,000 without
changing the renderer architecture. This stage keeps full-scene flow matching,
adds sparse full-resolution player RGB/edge supervision through the frozen
Pixel VAE decoder, and lowers the active AdamW learning rate from `1e-5` to
`3e-6` while preserving optimizer moments.

## Why this supervision

The Pixel VAE uses global encoder and decoder attention. An RGB player mask
resized to the `36x64` latent grid is only a geometric occupancy estimate, not
the decoder's true latent influence map. This recipe therefore leaves both
latent region upweights at zero. It decodes one player-visible future frame per
sample and applies the original-resolution instance mask after decoding, so the
frozen decoder Jacobian routes gradients to every responsible latent token.

The global flow objective stays enabled. The B-branch pixels-only ablation
damaged backgrounds, while its combined-loss trial was substantially safer.
The player RGB/edge coefficients follow that combined trial (`1.0` and `0.25`).

## Memory and batch semantics

The 65-frame parent used seven GPUs with per-rank batch 4 and accumulation 1.
Decoder-backprop adds memory, so this recipe defaults to per-rank batch 2 and
accumulation 2. The effective source-window batch remains 28 per optimizer
update. Only one future frame per sample is decoded.

## Launch contract

- Parent: complete `step_0040000.pt` from W&B run `okd5jze8`.
- Use strict `--resume`; do not warm-start or reset AdamW.
- Output and checkpoint-staging directories must be fresh.
- Run a five-update disabled-W&B GPU smoke from the original parent first.
- After smoke, start the real run from the original step-40,000 parent, not the
  smoke checkpoint.
- Default cumulative target is 44,000; override `STEPS` if needed.

Entry point:

```bash
bash train_scripts/recipes/m3/train_m3_qk_player_pixel_refine_7gpu.sh
```

Expected runtime configuration includes:

```text
loss_mode=combined
latent_entity_region_upweight=0
latent_player_region_upweight=0
pixel_loss_frames=1
pixel_frame_selection=player_unique
player_pixel_l1_weight=1.0
player_pixel_edge_weight=0.25
batch_size=2
gradient_accumulation=2
optimizer saved LR=1e-5
optimizer active LR=3e-6
```
