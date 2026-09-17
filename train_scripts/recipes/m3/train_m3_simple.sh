#!/usr/bin/env bash
set -euo pipefail

# Portable entry point. The legacy-named script now derives its world size
# from CUDA_VISIBLE_DEVICES and supports any number of confirmed free GPUs.
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
exec bash "$repo_root/train_scripts/recipes/m3/train_m3_simple_4gpu.sh" "$@"
