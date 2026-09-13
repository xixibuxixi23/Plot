# PLOT

PLOT is the canonical source repository for **Multi-Inhabitant World Models
with Writable 3D Memory**. It contains the maintained M1–M4 model code, dataset
adapters, training entry points, prepared indexes, fixed upstream weights and a
relocatable Linux training environment. Large datasets, run outputs and learned
checkpoints remain outside Git.

| Component | Entry point | Purpose |
|---|---|---|
| M1 | `experiments/m1/train_multiview_persist_full.py` | current two-view PERSIST flow baseline for 48³ memory |
| M2 | `train_scripts/train_transition_full.py` | predict kinematic residuals and sparse 13³ writes |
| M3 | `train_scripts/train_renderer.py` | causal 65-frame diffusion-forcing renderer |
| M4 | `train_scripts/train_policy.py` | predict an eight-action chunk from completed M3 frames |

`plot/` is the reusable package. `train_scripts/` contains canonical trainers.
`dataset_toolkits/` derives model-specific indexes without modifying the public
continuous release. See [project layout](docs/project_layout.md), [data
setup](docs/data.md), [training commands](docs/train.md), and the
[project management runbook](docs/project_management.md).

The maintained M1 flow baseline and its controlled alternatives live under
`experiments/m1/` while their reusable geometry and codec components remain in
`plot/`. The older geometry-fill entry points are retained for comparison.

Before starting a formal or long-running job, capture its immutable code and
data provenance with [`scripts/capture_run_manifest.py`](scripts/capture_run_manifest.py).
Active and historical runs are indexed in [`docs/runs/README.md`](docs/runs/README.md).
The current H200/H100 directory roles and migration exceptions are recorded in
[`docs/machine_workspaces.md`](docs/machine_workspaces.md).

## Move to another machine

1. Copy this directory and extract `polis_v1_20260909_360p` anywhere.
2. On compatible Linux x86-64/CUDA 12.8, run `source .venv/bin/activate`.
   Otherwise create Python 3.11 and install `requirements/m3-b200.txt`.
3. Validate the moved bundle without a GPU:

   ```bash
   python scripts/validate_bundle.py --dataset-root /path/to/polis_v1_20260909_360p
   ```

4. Start M3, for example:

   ```bash
   export PLOT_DATASET_ROOT=/path/to/polis_v1_20260909_360p
   export PLOT_CHECKPOINT_STAGING_DIR=/fast/local/or/pfs/plot-checkpoints
   bash train_scripts/recipes/m3/train_m3_b200_8gpu.sh
   ```

Prepared M3/M4 indexes contain relative episode paths, so moving the dataset
does not require rewriting them. Exact fixed-asset sizes and hashes are in
`checkpoints/MANIFEST.json`.

For a full B200 handoff, send the archive together with
[`HANDOFF_README.md`](HANDOFF_README.md). The included downloader extracts one
dataset shard at a time, avoiding a second 744 GB copy of train and val data.
