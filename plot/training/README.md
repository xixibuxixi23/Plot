# Training library

Reusable objectives, trainer state, monitoring and rollout helpers live here.
Executable argument parsing and launch behavior belong in `train_scripts/`.
This separation keeps training code testable without importing command-line
entry points.
