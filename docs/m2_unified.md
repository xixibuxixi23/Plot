# Unified M2 (opt-in, 2026-09-25)

## Architecture

`TransitionArgs(unified_interactions=True, air_class=..., event_queries=8)`
uses one joint-action dynamics trunk for all residents (players and NPCs):

```text
13^3 local voxel tokens + resident state tokens
                       |
joint actions + kinematic proposal + camera/state + optional velocity
                       |
        Shared Joint-action Transformer (default: 6 layers)
                 /                       \
     Dense state heads              Unified event decoder (2 layers)
  pose/camera/held item        count, time, operation, address, payload
                 \                       /
                 Shared world state commit
```

- There is no independent `player_blocks` or `attack_decoder` in this mode.
- State and all interaction losses reach the **same** trunk.
- Geometry is a 13^3 local crop. There is no second 7^3 motion branch.
- RGB is unused by default, including when `previous_rgb` is present. Set
  `unified_use_rgb=True` / `--use-rgb` only for an explicit RGB ablation.
- The reusable dataset/cache materializer still reads recorded RGB from episode
  files; this change removes runtime model dependence, not video cache I/O.
- The shared candidate space is 2197 voxel addresses + A resident addresses +
  null. Self, inactive residents and unknown voxel targets are masked.
- Events share one ordered sequence: **remove=0, place=1, attack=2**. Count=0
  represents no interaction. Default 8 slots cover 8 steps at one event/actor/step.
- A target-aware shared payload MLP produces block-class logits and a
  non-positive attack HP delta (`-softplus`). Removal writes the vocabulary air
  class; placement cannot select air; attacks can select only other residents.
- Camera position/direction condition the shared state embedding. This is a
  learned target pointer, **not** a new hard raycast/occlusion model. Place
  addresses denote the destination cell, not the block surface being pointed at.

## Supervision and limitations

`transition_loss(..., objective='unified')` supervises pose, angles, camera,
inventory, and unified count/time/operation/address/block/damage. It never uses
an auxiliary HP prediction to update authoritative health. `full` dispatches
to the same unified objective; `no_hp`, `edit`, and `player` fail explicitly
instead of silently discarding attack supervision.

The existing dataset marks multi-write-per-actor-per-frame labels invalid. This
version preserves that contract; it does not recover missing multi-event labels.
Actor-windows containing invalid event frames or more events than query capacity
are excluded from ordered-event supervision and counted in metrics. Valid state
supervision remains. Ambiguous payloads do not become false payload labels.

Operation labels come from confirmed block payloads (air -> remove, otherwise
place) or optional recorded edit-kind labels; resident-address events are attacks.
The air class is looked up from the vocabulary (default raw air ID 126), not
assumed to have class index zero.

`decode_interactions()` emits the existing eight-step `TransitionCommitter`
arrays. Event slot collisions at the same actor/time keep the highest-confidence
slot, first on ties, so a slot is not applied twice. `decode_transition()` in
the shared closed-loop pipeline now dispatches to this decoder when opted in.
Specialized M4/player-rollout scripts that directly consume `attack_*` outputs
remain legacy-only until migrated; `decode_attacks()` rejects unified mode.

## Training entry point

`train_scripts/train_m2_unified.py` starts a new unified run. It is **not**
automatically launched by this change. Pass actual dataset/index/vocabulary and
item-map paths, a fresh cache root and a fresh output directory:

```bash
python train_scripts/train_m2_unified.py \
  --dataset-root DATASET_ROOT --index INDEX_JSON \
  --vocabulary VOCAB_JSON --items ITEMS_OR_CONFIG_JSON \
  --cache-root NEW_UNIFIED_CACHE --output-dir NEW_RUN \
  --event-queries 8 --event-context-frames 4 --device cuda
```

The items file is either a name-to-index map or a JSON object with an `items`
field. Cache identity includes dataset root, index/vocabulary hashes, and items
to avoid silently reusing caches encoded under another vocabulary. A training
index must include actual combat examples to learn attack behavior; an
edit-only subset cannot supervise a useful damage predictor.

`--resume` supports only an identical **unified** architecture/vocabulary and
restores its optimizer in a fresh output directory. It is not a converter for
split-branch checkpoints. Old motion weights cannot be merged into the shared
trunk merely by renaming keys; distillation/warm-start needs a separate study.

## Compatibility and tests

All new architecture flags default off. Legacy weights retain their original
module names and forward path. Existing checkpoint files, launchers and active
jobs are not modified. Unified mode rejects `player_branch`/`attack_branch`
instead of constructing unused second networks. Legacy per-frame diagnostic
heads remain for API compatibility but are frozen in the unified training entry.

CPU checks:

```bash
python -m pytest -q tests/test_transition.py tests/test_unified_transition.py \
  tests/test_closed_loop.py tests/test_kinematics_gradient.py
```

These verify gradients, typed decoding, world commit, null/self/unknown/padding
masks, RGB independence, prior context, checkpoint round trips and legacy tests.
They do not establish training quality or full M4 rollout integration.
