# Machine and workspace map

This page records directory roles, not credentials or network addresses. The
Git commit in the run registry is authoritative; a directory name or branch
prefix does not identify the hardware that created a commit.

## Current assignment (2026-09-13 UTC audit)

All three current cluster nodes see the same
`/public/0_DATA/2_Avatar/zhizhou_share/rcz` filesystem. Hardware was checked
with `nvidia-smi`: both assigned training hosts contain eight NVIDIA H200 GPUs.
The names below are host roles, not assumptions inferred from SSH aliases.

| Host / hardware | Assigned role | Development workspace | Training workspace | Rule |
|---|---|---|---|---|
| `zhizhou-avgen-2` / 8x H200 | M1 | shared `Plot` on `main` or a short-lived feature branch | shared `Plot-runs/m1_h200_flow_resume1476000_20260913`, detached and not launched | Resume only after strict-load, fixed-batch, and multi-GPU smoke checks of the selected step-1,476,000 checkpoint. Do not develop in legacy `Polis`. |
| `js-public` / 8x H200 | M3 | shared `Plot` on `main` or a short-lived feature branch | shared `Plot-runs/m3_h200_identity_resume26000_20260914`, detached at `172f278` and not launched | The next M3 stage starts from step 26,000 plus the step-3,300 player-identity encoder; do not fall back silently to step 25,000. |
| historical independent host / 8x H100 | checkpoint source only | historical `Plot`/`Plot-dev` workspaces | historical stopped outputs | Do not launch new training here. Preserve it until selected M3 checkpoints and evidence are verified on shared storage. |

No M1 or M3 training process was active at the 2026-09-13 UTC audit.
Preserve both output directories and do not report a prepared worktree as a
running job. The legacy `Polis` workspace and the independent H100 workspaces
predate the final layout and remain explicit exceptions until their outputs
and branches are archived.

## Required layout for the next run

```text
Plot-dev/                     # edit, test, commit, push, and open PRs
Plot-runs/
  <run-id>/                   # detached immutable worktree for one run
outputs/
  <run-id>/                   # logs, configs, manifests, and checkpoints
```

Create a run worktree only after the source has been committed and tested:

```bash
cd Plot-dev
git fetch origin
git worktree add --detach ../Plot-runs/<run-id> <commit>
cd ../Plot-runs/<run-id>
python scripts/capture_run_manifest.py \
  --output-dir /path/to/outputs/<run-id> \
  --model <m1-or-m3> \
  --run-id <run-id> \
  --dataset-repo xixibuxixi/polis-v1 \
  --dataset-revision <revision> \
  --command-file /path/to/launch-command.txt
```

Run smoke tests and the formal job from that detached worktree. If a fix is
needed, make it in `Plot-dev`, commit and push it on a new feature branch, then
start a new run worktree or document an intentional checkpoint resume.

## Synchronization boundary

- GitHub `main` is the only integrated source of record.
- Feature branches carry reviewable code changes; one person or agent owns a
  feature branch at a time.
- `git push` publishes commits on a feature branch. A pull request reviews and
  merges those commits into `main`. Other machines update only their
  development worktrees with `git pull --ff-only origin main`.
- Datasets and selected checkpoints use Hugging Face, not Git. Local frequent
  checkpoints and W&B artifacts remain outside the repository.
- Never infer run provenance from the current branch of a directory after a
  process has started. Record the launch commit in `run_manifest.json`.
