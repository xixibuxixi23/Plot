#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 <github-repository-url> [branch]" >&2
  exit 2
fi
repo_url=$1
branch=${2:-main}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

if [[ ! -d .git ]]; then git init -b "$branch"; fi
if git remote get-url origin >/dev/null 2>&1; then
  current=$(git remote get-url origin)
  if [[ "$current" != "$repo_url" ]]; then
    echo "origin already points to $current; refusing to replace it" >&2
    exit 1
  fi
else
  git remote add origin "$repo_url"
fi

git fetch origin "$branch"
remote_ref="origin/$branch"
required_ignores=(
  '.venv/' 'outputs/' 'wandb/'
  'checkpoints/**/*.safetensors' 'derived/**/*.pt' 'derived/**/*.jsonl'
  'derived/m2/episode_index.json'
)
remote_ignore=$(git show "$remote_ref:.gitignore")
for pattern in "${required_ignores[@]}"; do
  if ! grep -Fqx "$pattern" <<<"$remote_ignore"; then
    echo "remote .gitignore is missing required protection: $pattern" >&2
    exit 1
  fi
done
if git ls-tree -r --name-only "$remote_ref" | grep -Eq \
  '^(\.venv/|outputs/|wandb/|checkpoints/.*\.safetensors$|derived/.*\.(pt|jsonl)$|derived/m2/episode_index\.json$)'; then
  echo "remote tracks a machine-local environment, tensor asset, index, or output" >&2
  exit 1
fi

git reset --hard "$remote_ref"
git branch --set-upstream-to="$remote_ref" "$branch"
echo "connected to $repo_url ($branch); ignored local assets were preserved"
