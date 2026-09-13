# PLOT project management

This repository is the only source of record for maintained PLOT code. The old
`Polis` working directory is a temporary legacy runtime for the M1 job that was
already active when this rule was introduced. Do not start new development
there.

## Four records with separate responsibilities

| Record | Canonical location | Contents |
|---|---|---|
| Source | `xixibuxixi23/Plot` | Code, tests, recipes, small configs and docs |
| Dataset | `xixibuxixi/polis-v1` | Immutable raw release and reproducible indexes |
| Learned weights | `xixibuxixi/plot-checkpoints` (planned; create after HF login) | Selected milestone, best and final weights |
| Run evidence | Local output plus W&B | Logs, videos, metrics and frequent resume checkpoints |

Never commit credentials, raw datasets, virtual environments, W&B state or
training outputs. A checkpoint is meaningful only when it points back to an
exact source commit and dataset revision.

## Branches and ownership

`main` is the latest integrated version that has passed relevant tests. Use one
short-lived branch for each coherent change:

```text
experiment/m1-<topic>
experiment/m2-<topic>
experiment/m3-<topic>
experiment/m4-<topic>
maintenance/<topic>
```

Commit and push at each recoverable milestone. Before integration, update the
branch from `origin/main`, resolve conflicts, and run the tests affected by the
change. Do not use a shared physical working directory from two terminals or
agents at the same time.

Name branches for the component and change, not for the machine that happens
to run them. Hardware belongs in the run manifest. The existing `b200/*`
branches are historical M3 milestones created before this convention; a branch
with that prefix is not evidence that the commit was made on a B200 host.

## Separate development from training

Development remains in the normal clone. Each long run gets a detached
worktree at an immutable commit:

```bash
git fetch origin
git worktree add --detach ../Plot-runs/m3-<run-id> <commit>
cd ../Plot-runs/m3-<run-id>
python scripts/capture_run_manifest.py \
  --output-dir /fast/outputs/<run-id> \
  --model m3 \
  --run-id <run-id> \
  --dataset-repo xixibuxixi/polis-v1 \
  --dataset-revision <revision> \
  --command-file /path/to/launch-command.txt
```

Do not edit or pull a training worktree. Continue development in the normal
clone. A code change used by a running experiment requires a new commit and a
new run, or an explicitly documented compatible resume from a checkpoint.
Use `Plot-dev` for a mutable development worktree and `Plot-runs/<run-id>` for
detached training worktrees when naming new machine-local directories. Existing
jobs that predate this rule remain explicit exceptions in the run registry;
do not rename or move a directory underneath a live process.

## Run lifecycle

1. Give the run a unique ID: `<model>_<purpose>_<machine>_<YYYYMMDD>_vN`.
2. Commit the source and ensure the worktree is clean.
3. Create a detached training worktree at that commit.
4. Generate `run_manifest.json` in the output directory.
5. Run smoke tests, then start the formal job without changing its worktree.
6. Add the job to `docs/runs/README.md` with owner, commit, parent checkpoint,
   machine label, output path and W&B URL.
7. On completion, record final step and metrics, upload selected weights, and
   move the entry to the completed section.

Frequent local checkpoints are operational data. Upload only milestones needed
for cross-machine resume and selected best/final checkpoints. Never upload a
file currently being overwritten: first make an immutable staged copy, verify
its step, compute SHA-256, and then upload it.

## Integrating the legacy Polis workspace

The legacy M1 run stopped at step 1,476,560 with a complete step-1,476,000
resume checkpoint. Its maintained M1 model, trainer, tests and documentation
were migrated into this repository at commit `4b6ce14`; the earlier immutable
step-1,127,000 checkpoint passes a strict CPU state-dict load. Do not restart
new development in `Polis`. Before the first PLOT-based resume, run a short
multi-GPU smoke test and compare one fixed evaluation batch. Once that passes,
rename the old directory to `Polis-legacy-readonly`; do not delete its outputs.

Do not merge the two unrelated Git histories with
`--allow-unrelated-histories`, and do not copy either `.git` directory. Migrate
reviewable source files into the PLOT history instead.
