# Data setup

The canonical release is `polis_v1_20260909_360p`; its Hugging Face dataset is
`xixibuxixi/polis-v1`. Download or copy the release, then pass its extracted
directory through `--dataset-root` or `PLOT_DATASET_ROOT`.

The release stays continuous and model independent. It stores RGB at 640×360,
lossless uint16 instance masks, actions, cameras, 49³ voxel observations,
resident states and events. Training-time 48³ memory tiles, 13³ write
addresses, 65-frame M3 contexts and 8-frame M4 chunks are derived here.

Prepared metadata under `derived/` includes common vocabularies, portable M3
train/val indexes and balanced M4 train/val indexes. All episode paths are
relative to the extracted dataset root.

Regenerate M3 and M4 metadata with:

```bash
python dataset_toolkits/build_renderer_window_index.py "$PLOT_DATASET_ROOT" \
  --split train --context-frames 65 --stride 8 --output derived/m3/train_c65.pt
python dataset_toolkits/build_m4_index.py "$PLOT_DATASET_ROOT" \
  --output-dir derived/m4
```

M2 materializes caches lazily. Build its portable index first:

```bash
python dataset_toolkits/build_episode_index.py "$PLOT_DATASET_ROOT" \
  --output derived/m2/episode_index.json
```

Never edit the raw release to form fixed 8-frame groups. Chunking belongs in
the adapters so the same release remains reusable.
