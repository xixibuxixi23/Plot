# H200 and H100 workspace map

This page records directory roles, not credentials or network addresses. The
Git commit in the run registry is authoritative; a directory name or branch
prefix does not identify the hardware that created a commit.

## Current transition state

| Machine | Development workspace | Training workspace | Rule |
|---|---|---|---|
| H200 | `Plot` on `main` or a short-lived feature branch | legacy `Polis` contains the stopped M1 run | Do not restart new development in `Polis`; the next M1 run must use a detached PLOT worktree. |
| H100 | `Plot-dev` on `main` or a short-lived feature branch | `Plot-runs/m3_h100_blockcausal_resume25000_20260914` is detached at `5e874fd` and prepared, but not launched | `Plot` retains the historical M3 feature branch; use the detached worktree only if the step-25,000 run is intentionally resumed. |

No M1 or M3 training process was active at the 2026-09-13 21:47 UTC audit.
Preserve both output directories and do not report a prepared worktree as a
running job. The old H200 `Polis` and H100 `Plot` workspaces predate the final
layout and remain explicit exceptions until their outputs and branches are
archived.

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
