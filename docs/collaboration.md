# Code collaboration

The transfer archive bootstraps a machine once. After that, update source code
through one GitHub repository and keep large machine-local assets in place.

## Directory ownership

Git manages source, tests, recipes and documentation. Git must not manage:

```text
.venv/
checkpoints/**/*.safetensors
derived/**/*.pt
derived/**/*.jsonl
outputs/
wandb/
```

These paths are protected by `.gitignore`. Raw episodes remain outside the
repository under `PLOT_DATASET_ROOT`.

## Connect an extracted bundle to GitHub

After the GitHub repository exists, run once inside `Plot`:

```bash
bash scripts/connect_git_remote.sh https://github.com/OWNER/REPOSITORY.git main
```

The script refuses a remote that tracks packaged environments, tensor assets or
run outputs. It makes the remote branch authoritative for source files while
preserving ignored local assets.

## Daily workflow

Each collaborator uses a branch:

```bash
git switch main
git pull --ff-only
git switch -c person/topic
# edit, test, commit
git push -u origin person/topic
```

Merge reviewed branches into `main`. Update a training machine between runs:

```bash
git switch main
git pull --ff-only
```

Record `git rev-parse HEAD` in every run report. Do not update a working tree
while training uses it. For simultaneous development and training, create a
stable worktree with `git worktree add ../Plot-run <commit-or-tag>` and link its
`.venv`, `checkpoints` and `derived` paths to the persistent assets.

Never copy an entire old source directory over a newer one. Never run
`git clean -fdx` on a training machine because it deletes ignored assets and
outputs.
