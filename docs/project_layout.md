# Project layout

```text
Plot/
├── plot/                 reusable M1–M4 models, adapters and trainers
├── train_scripts/        maintained training entry points and recipes
├── dataset_toolkits/     deterministic index/cache builders
├── derived/              portable vocabularies and M3/M4 indexes
├── checkpoints/          fixed VAE, warm start and text features
├── requirements/         pinned B200/CUDA environment
├── scripts/              environment and bundle checks
├── tests/                CPU contract and unit tests
└── .venv/                optional relocatable Linux x86-64 environment
```

The package dependency direction is `models <- training <- train_scripts`.
Models do not import training scripts, experiments or data loaders. Generated
checkpoints and W&B runs go under ignored `outputs/` and `wandb/` directories.

The former `Polis/datasets` directory mixed derived metadata with experimental
caches. Here, reusable metadata is under `derived/`; raw data is external and
selected explicitly with `--dataset-root`.
