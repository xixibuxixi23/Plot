# Production models

M1 occupies `fill.py`/`geometry_*`, M2 is `transition.py`, M3 is `renderer.py`,
and M4 is `inserted_policy.py` plus `structured_action.py`. The vendored M3
backbone is isolated in `renderer_backbone/` with provenance recorded there.
Research architectures belong under `experiments/` or `ablations/` and are not
re-exported.
