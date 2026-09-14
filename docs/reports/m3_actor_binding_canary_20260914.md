# M3 actor-binding intervention canary (2026-09-14)

## Decision

The intervention is technically stable but ineffective in the 100-step canary. Do not
extend it to 500 or 10,000 steps. Correct, shuffled, and zero player references produce
nearly the same player pixels, so the renderer still does not causally bind a resident's
four reference views to that resident's appearance.

The failed implementation remains in Git history at `4d0e75e`; the working branch was
returned to the preceding unified-reference implementation after this audit.

## Host and run configuration

- Host: `zhizhou-avgen-js-public`, port 20470.
- Hardware: 8 x NVIDIA H200, 143771 MiB each.
- Per-GPU batch: 2; effective source-window batch: 16.
- Warm start: unified checkpoint step 26700, followed by a 10-step smoke test and a
  100-step canary to step 26810.
- Peak reserved memory: 75.83 GiB per GPU.
- W&B: offline run `offline-run-20260914_132339-lqtnezlv`.

The experimental intervention combined:

1. training-only masking of other-player pixels in the first known RGB frame with
   probability 0.6;
2. identity loss weight 0.3, applied only for noise time at least 0.6;
3. one actor-separated reference adapter, with shared parameters reused in the final
   four spatial DiT blocks; and
4. reset of the unified adapter while preserving the trained reference encoder and
   backbone.

The 10-step smoke test and the 100-step distributed run both completed without OOM or
runtime errors. The full local test suite passed before execution: 117 passed, 1 warning.

## Fixed validation

The fixed 16-batch validation at step 26810 reported:

| Metric | Value |
|---|---:|
| flow loss | 0.327668 |
| identity loss on eligible high-noise samples | 0.197866 |
| high-noise identity similarity | 0.379072 |
| high-noise identity ranking accuracy | 0.187500 |

The high-noise identity metrics are not directly comparable with the earlier all-noise
metrics. They provide no positive reason to scale this run.

## Causal reference audit

Each probe was generated with identical noise under correct references, resident-level
cyclically shuffled references, and zero references. `Shuffled delta` and `zero delta`
are output changes inside the ground-truth player region relative to the correct-reference
generation.

| Probe | Correct player L1 | Shuffled player L1 | Shuffled delta | Zero delta | Correct / shuffled identity similarity |
|---|---:|---:|---:|---:|---:|
| construction | 0.180324 | 0.180320 | 0.001676 | 0.006017 | 0.712853 / 0.712770 |
| four-player | 0.049273 | 0.049326 | 0.001402 | 0.001385 | 0.648218 / 0.648240 |
| three-resident combat | 0.158870 | 0.158740 | 0.001318 | 0.001536 | 0.466577 / 0.469126 |

Shuffling does not make player reconstruction worse and barely changes the generated
pixels. In the combat probe, the shuffled identity similarity is slightly higher. Zero
references also have very little effect in two of three probes. This is a failed causal
conditioning test, despite nonempty projected player regions and distinct source
reference images.

The adapter did receive gradients: at step 26810 its formerly zero-initialized output
projection has weight norm 0.529247 (absolute mean 0.000822). The failure is therefore
not explained by a frozen or disconnected parameter. The current identity objective can
be satisfied from noisy target/history/spatial cues without requiring the actor-specific
reference path.

## Artifacts

- Smoke checkpoint:
  `outputs/m3_h200_actor_bind_smoke_b2_8gpu_26700_26710_20260914/step_0026710.pt`
- Canary checkpoint:
  `outputs/m3_h200_actor_bind_canary_b2_8gpu_26710_26810_20260914/step_0026810.pt`
- Validation and visualization directory:
  `outputs/m3_h200_actor_bind_canary_b2_8gpu_26710_26810_20260914/`
- Correct/shuffled/zero videos and metrics:
  `outputs/m3_actor_bind_reference_audit_step26810_20260914/`

## Recommended next step

Architecture placement and loss reweighting alone have now failed repeated causal
audits. The next experiment should directly supervise reference-to-avatar appearance:
pretrain a small player-only renderer or latent residual on masked player crops using
reference images plus pose, then integrate that trained component into M3. A stronger
alternative is to collect paired counterfactual renders with the same scene, pose, and
camera but swapped skins. Either route makes the reference image identifiable; another
long continuation of the present objective does not.
