# M4-State NPC pilot

This experimental path implements the 2026-09-21 state-policy design without
changing the existing inserted-visual-policy baseline. It has four independent
checkpoints: peaceful villager (rcz3), melee zombie (rcz4), swordsman skeleton
(rcz5), and defender villager (rcz6). Labels refer to authorized node mappings,
not SSH host names. Existing GPU occupancy jobs are retained as requested.

## Network and information contract

- Current 48-cubed local block classes and known mask; unknown is not air.
- 16-cubed near and 48-cubed far 3D CNN branches, 64 tokens each.
- Eight completed states: relative xyz, angle sine/cosine, HP, camera-relative
  position/direction, committed event cues, item and resident type embeddings.
- Two resident-attention layers, four history layers, three action decoder
  layers; hidden 256, 8 heads, FFN ratio 4; eight learned horizon queries.
- Eight structured actions: key BCE, categorical hotbar and two 17-bin mouse
  heads. Peaceful constraints are enforced. No RGB, VAE, M3 weights, text,
  teacher targets, future actions/states, engine-only velocities or slot IDs.

This is a **local privileged-state** policy, not a first-person-visible policy.
Other residents outside +/-24 blocks are masked at each observed time. Geometry
stays world-axis aligned; metric coordinates are relative to the target's latest
position and include the sub-voxel grid offset. Integer crop centers only map
the raw observation to this metric geometry.

Training at t uses states t-7..t and incoming actions t-8..t-1; labels are
actions t..t+7. Early histories are left padded and masked. Episodes must have
valid live target state and valid NPC policy labels without termination crossing.
Data adapter uses the same transition-aligned held-item convention as M3;
item IDs are remapped by name because source metadata and old repository
vocabularies differ. Checkpoints contain their item and block vocabularies.

## Pilot scope

Each profile uses 8 training episodes and 3 val-ID episodes, up to 8 windows per
episode. The cache is approximately 64 train and 24 held-out windows per profile;
actual counts and attack/wait frequencies are recorded in summary.json.
Selection is deterministic and is not representative full-dataset evaluation.
Train 200 steps, batch 4, AdamW 2e-4, BF16, single GPU per node; 8-to-1 horizon
weights. Log unweighted train/val loss, each horizon's loss, key accuracy and
attack TP/FP/FN. The test is learnability and infrastructure, not task success.

No verified guard-origin/radius was supplied to this adapter. The goal token is
masked in this pilot, so defender return-to-guard behavior cannot be claimed.
Formal training needs a deployable goal contract and broader episode sampling.
Pilot geometry comes from current engine state; M1/M2 rollout robustness and
engine action replay remain separate required evaluations.

The first pilot exposed attack sparsity (6--18 attack steps among 512 training
action steps). A follow-up for the three combat profiles runs 1,000 steps from
the same initialization with `--balance-attack`: attack-containing and other
windows each receive half the training sampling probability. Held-out sampling
remains unchanged. These runs are under `models-balanced/` and their logs have
`-balanced` suffixes. Both training duration and sampling changed, so the result
is a diagnostic follow-up, not an isolated sampler ablation. The optional
balanced sampler currently supports single-device training only.

## Commands and outputs

From the authorized node shell in the shared repository:

```bash
bash train_scripts/run_state_policy_pilot.sh rcz3 pilot-20260921-v1
```

Use the corresponding rcz4/rcz5/rcz6 argument on its mapped node. The launcher
does not connect to or select a node, and never stops existing processes.
It prepares a profile-specific cache, runs contract tests, then trains.

Artifacts live below the shared root at
`runs/m4-state/pilot-20260921-v1/{cache,models,logs}`. Each model has config.json
(data provenance, vocabulary and source hashes), train.jsonl, validation.jsonl,
latest.pt, and COMPLETE.json on successful completion. Existing output folders
are never reused for a new training run. Source staging on Windows contains only
new small files, not a second repository or dataset copy.

## Runtime handoff

`StatePolicySession` in `plot/pipelines/state_policy_pipeline.py` accepts
committed CharRows, WriteEvents and executed actions, stores its own immutable
eight-state history, and captures a WorldMemory crop. After capture, the caller
can dispatch `session.act(frozen_inputs)` alongside M3. Same-profile sessions
can share one model. Checkpoint loading validates block vocabulary and remaps
deployment item IDs by name; missing item names fail explicitly.

The existing ClosedLoopPipeline still uses the visual policy. This pilot adds
the independent runtime adapter but does not replace that orchestrator or
claim measured M3/M4 concurrency. M1 and M2 retain their current recent-RGB
dependencies. End-to-end scheduler integration follows pilot validation.
