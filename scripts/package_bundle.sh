#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
destination=${1:-"$(dirname "$repo_root")/plot_training_bundle.tar.zst"}

ZSTD_NBTHREADS="${ZSTD_NBTHREADS:-8}" ZSTD_CLEVEL="${ZSTD_CLEVEL:-3}" \
tar --zstd -cf "$destination" \
  --exclude='.git' --exclude='outputs' --exclude='wandb' \
  --exclude='__pycache__' --exclude='.pytest_cache' \
  -C "$(dirname "$repo_root")" "$(basename "$repo_root")"
echo "$destination"
