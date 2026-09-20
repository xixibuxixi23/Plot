# M3 full train: flow + 10x player RGB/edge + unique frames (2026-09-20)

User approved a new 200-update short experiment after the pixels-only trial.
No architecture changes; do not initialize from its degraded 7565 weights.

## Initialization and training

- Host: `root@vr.turbo-ai.com:20470`, CUDA_VISIBLE_DEVICES `0,1,2,3,4,5,6,7`.
- Initialize from `/root/rcz-runs/m3_fulltrain_8gpu_b4_player_flow_20260919/step_0007365.pt`.
  Warm-start all model weights, reset AdamW moments, keep absolute step counter.
- Stop at global **7565**, exactly **200 new updates**. Fresh output and W&B run.
- Loss: **`full_flow + player_flow + 1.0 * player_RGB_L1 + 0.25 * player_edge_L1`**.
  Both flow coefficients are 1; RGB/edge are 10x the original .1/.025.
  All other loss coefficients remain zero. This is not loss-value equalization.
- Same complete validated train pool, 2,294,147 c9 windows / 25,229 episodes.
  Eight GPUs, batch 4 each, accumulation 1: global batch 32, 6400 sampled windows.
- All 461,064,506 M3 parameters trainable; VAE frozen. LR base/joint old channels
  `1e-5`, appearance + appended joint input columns `5e-5`; BF16.
- Same context9 (one clean prefix + eight future), random uniform block noise,
  cache32, 20-step pure-noise probe generation. No inference change.

## Frame selection

New opt-in `--pixel-frame-selection player_unique`, legacy `mixed` and `player`
are unchanged. Sample by visible player mask area without replacement, excluding
the prefix. With at least two visible-player frames, both selected frames have
players. With one, draw it once then another unused future frame; its empty mask
gives zero player loss. With none, choose two distinct future frames; player
loss is zero and full-frame flow still trains the sample. No GT masks are fed
to the network. Two frames decoded per sample as before.

## Noise diagnostics

`--log-player-noise-bins` adds detached diagnostic sums and counts for both
player latent flow MSE and decoded player RGB L1. Bins are low `[0,1/3)`,
mid `[1/3,2/3)`, high `[2/3,1]`. Frames with empty player masks are excluded.
The diagnostic mean is equal-weight over valid frames (the actual pixel loss
remains mask-area-weighted across the decoded batch).

Training diagnostic totals accumulate since the previous log, across every
microbatch and rank; SUM-reduce numerators and counts before division. The
ordinary `train/*loss` values still report the latest optimizer step. Validation
diagnostics aggregate all evaluated batches/ranks. An empty bin has count 0 and
no mean, never a fabricated zero error. `player_pixel_supervised_fraction`
reports how many decoded frames actually have a nonempty player mask.

## Evaluation / interpretation

- Fixed same 4 train + 4 val-ID probes, seed 20260919 + probe index, 20 Euler steps.
- Reuse the exact 7365 pure-noise baseline from the previous trial (same model,
  data, inference and seeds): `m3_fulltrain_8gpu_b4_player_pixels_only_20260919/before_7365`.
- Save/validate/visualize at 7400, 7500 and final 7565, W&B online.
- These are few qualitative probes, not full-validation estimates. New decoded
  validation losses use a different frame-selection rule, so comparing their
  magnitude directly with old validation logs confounds selection. Fixed probe
  generation remains comparable.
- This practical trial changes loss weights and frame selection together; it
  does not separately identify their causal effects. Noise diagnostics do not
  change gradients or sampling.

## Paths / recipe

- Recipe: `train_scripts/recipes/m3/train_m3_simple_fulltrain_player_rgb10x.sh`.
  Reuses the pixels-only launcher with explicit final CLI overrides; a test
  verifies the resolved combined-loss/unique-frame flags.
- Remote run: `/root/rcz-runs/m3_fulltrain_8gpu_b4_flow_rgb10x_unique_20260920`.
- Shared previews:
  `/public/0_DATA/2_Avatar/zhizhou_share/rcz/plotdemo/m3_fulltrain_8gpu_b4_flow_rgb10x_unique_20260920`.
- All source is captured in remote `source_snapshot.tar.gz` before launch.
  No credentials are added to code or documentation.

## Launch checks

- 76 renderer / codec / loss / monitoring / optimizer tests passed, including
  no replacement, empty/single-visible-frame fallbacks, noise bin membership,
  additive rank aggregation, and unchanged loss/gradients with diagnostics on.
- Python compilation, recipe shell syntax and diff whitespace checks passed.
- Launcher PID **176794**. Before launch GPU 1 acquired another task but retained
  about 62 GiB free; this job previously needed about 24–26 GiB per card. No
  other process was stopped or altered. Remaining cards were idle.
- All eight ranks started (PIDs **176882–176889**). W&B online:
  <https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/5ifdafdw>.
  Runtime config confirms flow/player-flow/RGB/edge weights **1/1/1/.25**,
  `player_unique`, noise-bin logging enabled, batch 4 and unchanged LR groups.
- Actual update verified at **7370** (five new steps): full flow .220588,
  player flow .527141, RGB .034985, edge .018209, total .787265 (finite).
  Peak allocated max **20.01 GiB**, peak reserved max **20.35 GiB**.
- First five-step interval: 320 decoded selections, **267** with a visible
  player (**83.44%**). Empty/single-visible-frame clips therefore still exist;
  they are explicitly reported rather than silently claimed fully supervised.
  Pixel low/mid/high means .023976/.029728/.048565 with counts 86/80/101;
  player-flow means .487712/.354854/.479952 with counts 309/307/391.
  These are initial measurements, not evidence of an improving trend.

## Completed result (200 updates; training remains stopped)

The run finished normally at global **7565**, saved `step_0007565.pt`, completed
W&B synchronization, and all eight workers exited. No continuation was started.

Same four val-ID pure-noise probes, equal-weight clip means (lower is better):

| Checkpoint | Player RGB L1 | Full-image RGB L1 |
| --- | ---: | ---: |
| Before, 7365 | 0.098090 | 0.076353 |
| 7400 | 0.093585 | 0.071872 |
| 7500 | 0.093560 | 0.068939 |
| Final, 7565 | 0.094814 | 0.073925 |

7500 has the best metrics on these few validation probes: player error -4.6%
and whole-image error -9.7% versus initialization. Final changes are -3.3% and
-3.2%, respectively. The four train probes do not show consistent improvement:
their final mean player error increased 2.3% and full-image error increased 10.2%.
This is not proof of dataset-wide generalization or resolved player identity.
Visual inspection of final comparison frames still shows wrong appearance,
pose and placement, without the large drift seen in the pixels-only trial.

Fixed-validation high-noise player flow changed only **0.481441 -> 0.479007**
between 7400 and 7565; decoded high-noise player RGB changed
**0.048245 -> 0.046967**. These are within-run comparisons under the new frame
selection, not an exact 7365 baseline of the new supervised objective.

Next proposed diagnostic was a fixed-noise reference substitution test. It has
**not** been executed; this trial does not establish reference usage.
