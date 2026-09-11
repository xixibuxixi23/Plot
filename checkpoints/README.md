# Checkpoint assets

This directory separates fixed upstream assets from checkpoints learned by
PLOT. Large tensor files are intentionally ignored by Git and are recorded in
`MANIFEST.json` with their exact size and SHA-256.

```text
pixel_vae/model.safetensors       required frozen RGB/latent codec for M3 and M4
m3_backbone/model.safetensors     recommended 2DAction-compatible M3 warm start
m4_text/text_catalog.json         canonical 476-string catalog
m4_text/text_cache.safetensors    frozen T5-v1.1-base embeddings for that catalog
m1/                               future selected formal M1 checkpoint
m2/                               future selected formal M2 checkpoint
m3/                               future selected formal M3 checkpoint
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
