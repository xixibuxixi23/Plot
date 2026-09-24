# Training entry points

| Model | Canonical entry point |
|---|---|
| M1 Unified Fill | `train_fill.py` |
| M2 Transition/Write | `train_transition_full.py` |
| M3 Renderer | `train_renderer.py` |
| M4 Inhabitant Policy | `train_policy.py` |

Only canonical production entries live at this level. Reproducible cluster
launchers are grouped by module under `recipes/`. Exploratory and superseded
trainers are intentionally excluded from this portable project.
