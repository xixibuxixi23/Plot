# Machine and workspace map

This page records directory roles, not credentials or network addresses. The
Git commit in the run registry is authoritative; a directory name or branch
prefix does not identify the hardware that created a commit.

## Current assignment (2026-09-14 UTC)

The two M3 workers see the same
`/public/0_DATA/2_Avatar/zhizhou_share/rcz` filesystem. Hardware was checked
with `nvidia-smi`: each contains eight NVIDIA H200 GPUs. M1 runs on an
independent eight-GPU H100 host with `/data/huangyh/hxh` storage. The names
below are host roles, not assumptions inferred from SSH aliases.

| Host / hardware | Assigned role | Development workspace | Training workspace | Rule |
|---|---|---|---|---|
| independent `H100-02` / 8x H100 | M1 | `/data/huangyh/hxh/Plot-dev`; update from GitHub `main` only | `/data/huangyh/hxh/Plot-runs/m1_h100_flow_resume1476000_20260914`, detached at `4a18d3b` | Stage and checksum the exact 46,200-sample legacy S01 cache and selected step-1,476,000 checkpoint before smoke testing. The balanced compact release cannot recreate this cache exactly. |
| `zhizhou-avgen-2` / 8x H200 | M3 node 0 | shared `Plot` for development only | shared `Plot-runs/m3_h200_identity_resume26000_20260914`, detached at `4a18d3b` | Launch rank 0 with socket NCCL on the dedicated multi-node rendezvous port. |
| `js-public` / 8x H200 | M3 node 1 | shared `Plot` for development only | the same shared detached M3 worktree | Launch rank 1 against `zhizhou-avgen-2`; both nodes jointly form one 16-GPU job. |

M3 identity-supervision training is active on the two H200 workers from step
26,000 to the step-26,500 evaluation gate. M1 is staged separately on H100 and
must not be reported as active until asset verification and the resume smoke
test pass. The legacy `Polis` workspace remains read-only checkpoint/cache
provenance, not a development checkout.

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
