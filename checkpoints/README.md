# Checkpoint assets

This directory separates fixed upstream assets from checkpoints learned by
PLOT. Large tensor files are intentionally ignored by Git. Fixed bundle assets
are recorded in `MANIFEST.json`; operational resume selections are recorded in
`SELECTED.json`, both with exact size and SHA-256.

```text
pixel_vae/model.safetensors       required frozen RGB/latent codec for M3 and M4
m3_backbone/model.safetensors     recommended 2DAction-compatible M3 warm start
m4_text/text_catalog.json         canonical 476-string catalog
m4_text/text_cache.safetensors    frozen T5-v1.1-base embeddings for that catalog
m1/                               selected M1 resume checkpoint and evidence
m2/                               future selected formal M2 checkpoint
m3/                               selected M3 resume/identity checkpoints
m4/                               future selected formal M4 checkpoint
```

M1 and base M2 do not require external pretrained encoders. M3 can omit its
backbone warm start, but it cannot train with the current entry point without
the frozen Pixel VAE. M4 requires the selected formal M3 checkpoint and the same
Pixel VAE. It reads the cached text tensors directly and does not load T5 during
training.

The M3 warm start is an initialization rather than a resume: only tensors whose
names and shapes match the PLOT renderer core are transferred. New voxel,
resident, appearance, and event-conditioning parameters remain trainable.

## Selected operational resume points

Large files below are ignored by Git. `SELECTED.json` records their immutable
names, sizes, hashes, and lineage.

- M1 resumes from
  `m1/m1_flow_2nodes_b32_eval10k_v2/step_001476000.pt` on
  `zhizhou-avgen-2` after compatibility smoke tests.
- M3 identity supervision resumes from
  `m3/m3_entity_reference_stage1/step_0026000.pt` on `js-public` and also
  requires
  `m3/player_identity_real_finetune/step_0003300.pt`.
- M3 step 25,000 is a parent of step 26,000, not the selected current resume
point. The aborted identity run logged through step 26,420 but did not save a
checkpoint.

The selected weights and their run metadata are mirrored in the private
Hugging Face model repository `xixibuxixi/plot-checkpoints`. The immutable Hub
revision and upload verification status are recorded in `SELECTED.json`.
