# Unified M1 mask-conditioned voxel completion

M1 has one production architecture and one checkpoint format. The same
`FillNetwork` accepts a partial 48-cubed voxel tile, `known_mask`, `fill_mask`,
and optional resident images.

## Operating modes

| mode | known voxels | images | supervised output |
| --- | --- | --- | --- |
| initialization | none | required | the full valid tile |
| geometry frontier | partial | absent | the newly exposed unknown fringe |
| image-assisted frontier | partial | present | the newly exposed unknown fringe |

Known voxels are conditioning and never contribute to fill loss. Predictions
are committed only where `known=False`. Image-free examples use a learned
null-image token, and camera auxiliary losses are disabled for those examples.

## Canonical implementation

- Dataset: `plot/data/fill_dataset.py`
- Model: `plot/models/fill.py`
- Trainer: `plot/training/fill_trainer.py`
- Training entry: `train_scripts/train_fill.py`
- Evaluation: `scripts/evaluate_m1_unified_modes.py`
- Tests: `tests/test_unified_fill.py`

Each resident contributes four deterministic slots: one image-conditioned
initialization sample, two geometry-only frontier samples, and one
image-assisted frontier sample. This yields a 25/50/25 percent training
mixture. Frontier states are selected only when the target tile exposes voxels
not covered by prior resident windows; image decoding is skipped for
geometry-only samples.

The completed reference run used rcz3 and rcz4 as 16 DDP ranks, per-GPU batch
8, global batch 128, BF16, and 50,000 optimizer updates. Its final checkpoint
is `runs/m1-unified/maskfill-50k-20260922-v1/checkpoint_final.pt` in the shared
training workspace. Checkpoints and run logs are intentionally outside Git.
