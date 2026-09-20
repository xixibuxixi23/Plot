# M3 full-train: decoded player-only loss ablation (2026-09-19)

User requested switching off the other main losses to test whether decoded
player appearance supervision is being overwhelmed. This is a short diagnostic
run, not a decision to remove scene supervision permanently.

## Parent and objective

- Parent: `/root/rcz-runs/m3_fulltrain_8gpu_b4_player_flow_20260919` on
  `root@vr.turbo-ai.com:20470`. STOP marker saved global step **7365** and
  completed the parent's W&B run. Existing checkpoints/results are retained.
- Initialize all M3 weights from `step_0007365.pt`. Warm-start resets AdamW
  moments, but preserves the absolute step counter. Final step **7565** means
  **200 new optimizer updates**, not 7565 new updates.
- Old objective: `full_flow + player_flow + .1 * player_RGB_L1 + .025 * player_edge_L1`.
- Trial objective: **`.1 * player_RGB_L1 + .025 * player_edge_L1`**.
  Both latent flow coefficients and all other pixel/identity/counterfactual
  coefficients are zero. RGB/edge coefficients are unchanged.
- Example parent step 7350: full flow .185034, player flow .415040,
  weighted RGB/edge .003659, total .603734. Pixel auxiliary terms are about
  **0.61% of scalar loss**, which does **not** establish their gradient share.
- Full-scene flow is still computed/logged as a diagnostic with zero weight.
  `player_flow_loss` is logged as zero when disabled, not evidence of perfect
  denoising. Total losses before/after removing terms are not comparable.

## Controls and interpretation

- Same validated full train pool (25,229 episodes, 2,294,147 c9 windows),
  8 GPUs `0,1,2,3,4,5,6,7`, per-GPU batch 4, accumulation 1, global batch 32.
- Full M3 trainable, VAE frozen; base LR `1e-5`, appearance and appended
  appearance input columns `5e-5`. BF16; 1 clean frame + 8 future frames.
- Same uniform random block noise level and two decoded future frames chosen
  by the existing player selection rule. GT instance masks supervise loss only.
- Resetting optimizer history is an additional difference; this is a practical
  pixels-only trial, not a perfectly controlled attribution of all changes to
  one coefficient. Full-dataset sampling restarts with the original seed.
- Keep fixed-seed 20-step pure-noise generation on the same 4 train + 4 val-ID
  probes. Score player **and full-image** metrics to detect background drift.
  These few probes are not a full validation-set estimate.
- Baseline is regenerated from the exact **7365** initialization before any
  update, not reused from the old 6500/7250 checkpoints.
- Save, validate and visualize at absolute 7400, 7500 and final 7565.
  Pixel-only training may reduce reconstruction loss without improving rollout
  or identity; assess the generated videos, not merely lower total loss.

## Paths and entry point

- Recipe: `train_scripts/recipes/m3/train_m3_simple_fulltrain_player_pixels_only.sh`.
- Required environment: `PLOT_DATASET_ROOT`, `INIT_CHECKPOINT`, `FINAL_STEP`,
  `OUTPUT_DIR`, `CUDA_VISIBLE_DEVICES`.
- Remote run: `/root/rcz-runs/m3_fulltrain_8gpu_b4_player_pixels_only_20260919`.
- Shared previews:
  `/public/0_DATA/2_Avatar/zhizhou_share/rcz/plotdemo/m3_fulltrain_8gpu_b4_player_pixels_only_20260919`.
- `before_7365/` holds the evaluator's wrapped `probes` metrics;
  `visualizations/step_*/` holds trainer flat `visual/...` metrics and videos.
- W&B online; the runtime `wandb_run.json` records the new run ID.

Tests added for exact pixels-only total/gradient, frozen decoder gradient
transmission, and zero loss/gradient for an empty future-player mask despite
positive flow and full-entity diagnostic losses.

## Launch and checks

- Recipe shell syntax passed; renderer/loss/optimizer/codec/monitoring tests:
  **70 passed**. The pixels-only tests explicitly compare autograd gradients
  against the weighted pixel terms, not only the numerical total.
- Parent launcher and all eight ranks exited before this run started.
- Trial launcher PID **2979407**, started at approximately 14:05 UTC.
  Initially executes the serial baseline evaluator, then becomes torchrun.
  Log: remote run `experiment.log`.
- All eight exact-7365 baseline probes completed before torchrun started:
  train probe mean player L1 **0.0920273**, full-image L1 **0.0611586**;
  val-ID probe mean player L1 **0.0980900**, full-image L1 **0.0763531**.
  These are equal-weight means over four clips per split.
- W&B online run: <https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/919symsk>.
  Runtime config confirms both flow weights zero, RGB/edge .1/.025,
  461,064,506 trainable M3 parameters, and the original learning-rate groups.
- Actual training verified at **7370** (five new updates), all eight ranks live.
  Total = auxiliary = **0.005034307** = `.1 * .045457918 + .025 * .019540599`.
  Global flow diagnostic **0.272552** is excluded from that total.
  Peak allocated max **20.01 GiB**, peak reserved max **20.33 GiB**.
  No improvement claim is made from this first batch or from the mechanically
  smaller total loss after disabling the two main terms.

## Completed result

Completed 200 updates, **7365 -> 7565**, on 2026-09-19 at about 14:46 UTC.
Final checkpoint `step_0007565.pt` saved; W&B synchronized and all workers exited.

Four fixed val-ID pure-noise probes, equal-weight means:

| Checkpoint | Player RGB L1 | Full-image RGB L1 |
| --- | ---: | ---: |
| Before, 7365 | 0.098090 | 0.076353 |
| 7400 | 0.086271 | 0.090640 |
| 7500 | 0.087764 | 0.095220 |
| Final, 7565 | 0.096799 | 0.120231 |

Early player improvement did not persist; final player error is only 1.3%
lower while whole-image error is **57.5% higher**. Large misplaced color regions
and background degradation are visible. Within-run fixed-validation decoded
player L1 decreases .031365 -> .029096 (7400 -> 7565), illustrating why lower
noisy-GT reconstruction loss alone is not sufficient evidence of better rollout.
This short experiment does not support permanently disabling both flow losses.
The next combined-loss trial starts from the intact 7365 parent, not these
pixels-only final weights.
