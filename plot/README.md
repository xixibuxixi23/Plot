# PLOT production package

Everything imported by training or inference lives under this namespace.

| Path | Responsibility |
|---|---|
| `models/` | Trainable M1--M4 networks and the isolated M3 backbone |
| `data/` | Continuous-episode adapters and collation |
| `training/` | Reusable losses, trainers, monitoring and rollout helpers |
| `pipelines/` | Inference orchestration and committed-state execution |
| package root | Shared schemas, coordinates, kinematics and sparse world memory |

The dependency direction is `data -> models -> training`, while pipelines may
compose all production layers. Production modules do not import executable
files from `train_scripts/`, `scripts/`, `experiments/`, or `ablations/`.
