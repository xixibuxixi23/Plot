# M3 unified player-reference canary report (2026-09-14)

## Decision

Do not start the 10,000-step run yet. The simplified one-branch implementation is
stable and checkpoint-compatible, but the current objective does not reliably bind a
resident's reference images to that resident's rendered appearance. Longer training
would currently be an expensive architecture/objective gamble rather than a validated
scale-up.

The working tree was returned to the best-tested unified implementation after the
failed late-injection experiment. The failed experiment remains in Git history and all
of its checkpoints remain on disk.

## Hardware and availability

- `js-public`, port 20474: 8 x NVIDIA H200 (143771 MiB). Used for all runs below.
- `zhizhou-avgen-2`, port 20494: 8 x NVIDIA H200, not H100. All eight GPUs were and
  remain occupied by another user's `/private/liqi_workspace/occupy_gpu.py`, about
  130596 MiB and 100% utilization per GPU. No process on that host was modified.
- The two hosts see the same repository and dataset filesystem.

## Stable operating point

- Batch 4 per GPU: OOM on H200.
- Batch 2 per GPU: stable on 8 GPUs.
- Effective source-window batch: 16.
- Effective rendered-view batch: 32 (`target_views_per_window=2`).
- Peak PyTorch reserved memory: about 74.33 GiB per GPU.
- W&B was kept offline; every run contains an offline run directory that can be synced
  later under the intended account.

## Current simplified implementation

- One `PlayerReferenceEncoder` retains all four native RGBA view token grids.
- One `UnifiedPlayerReferenceAdapter` is used for the whole DiT.
- There is no four-way hard view selection or pre-attention view averaging.
- The legacy pooled appearance, projected appearance, and per-block reference branches
  are mutually exclusive with unified mode.
- Self-player control remains in the target action/state path. The reference ROI excludes
  the observing target slot and is for other visible residents.
- A real step-26500 checkpoint migrates without missing required parameters.

## Validation results

All rows use the same fixed 16 validation batches. Total and identity losses in rows
26700/26800 are not numerically comparable with earlier rows because the identity loss
weight/margin changed; flow, similarity, and ranking remain comparable diagnostics.

| Step | Configuration | Flow loss | Identity similarity | Identity ranking |
|---:|---|---:|---:|---:|
| 26500 | legacy reference/appearance checkpoint | 0.30918 | 0.65389 | 0.42578 |
| 26600 | unified, 100 steps, identity weight 0.1 | 0.32592 | 0.63839 | 0.37891 |
| 26700 | unified, +100 steps, identity weight 1.0, last 4 spatial blocks | 0.32703 | 0.64173 | 0.35156 |
| 26800 | experimental late-four-block injection, reset adapter, 100 steps | 0.32920 | 0.64081 | 0.32812 |

The unified runs did not beat the legacy checkpoint on flow, identity similarity, or
identity ranking.

## Causal reference audit

Each probe was generated three times with the same noise: correct references, resident
references cyclically shuffled, and zero references. The key number is the generated
player-region L1 difference from the correct-reference output.

| Checkpoint | Probe | Shuffled delta | Zero-reference delta |
|---|---|---:|---:|
| 26500 legacy | construction | 0.00202 | 0.00775 |
| 26500 legacy | four-player | 0.00150 | 0.00228 |
| 26500 legacy | combat | 0.00324 | 0.02189 |
| 26600 unified | construction | 0.00261 | 0.08348 |
| 26600 unified | four-player | 0.00149 | 0.00967 |
| 26600 unified | combat | 0.00520 | 0.05318 |
| 26700 stronger identity | construction | 0.01723 | 0.06945 |
| 26700 stronger identity | four-player | 0.00153 | 0.00369 |
| 26700 stronger identity | combat | 0.00379 | 0.04677 |
| 26800 late injection | construction | 0.00192 | 0.00206 |
| 26800 late injection | four-player | 0.00144 | 0.00147 |
| 26800 late injection | combat | 0.00149 | 0.00178 |

The unified adapter learns a strong generic dependency on nonzero reference imagery, but
resident shuffling usually changes the output very little. Masking visible players in
the first rollout frame did not fix this, and ROI/token inspection confirmed that the
probes have nonempty projected resident regions and nonidentical reference encodings.
Therefore the unresolved issue is identity binding in the training signal, not an empty
projection or missing source images.

The late-injection experiment made both reference dependence and image quality worse in
the 100-step budget, so it was reverted rather than scaled.

## Artifacts

- Best legacy checkpoint:
  `outputs/m3_h200_2node_identity_canary_b2_from26000_to26500_20260914/step_0026500.pt`
- Best unified canary checkpoint for continued diagnosis:
  `outputs/m3_h200_unified_reference_identity1_blocks4_b2_8gpu_26600_26700_20260914/step_0026700.pt`
- Failed late-injection checkpoint:
  `outputs/m3_h200_unified_reference_late4_reset_b2_8gpu_26700_26800_20260914/step_0026800.pt`
- Reference-audit videos and JSON:
  `outputs/m3_unified_reference_condition_audit_step26600_20260914/`,
  `outputs/m3_unified_reference_condition_audit_step26700_20260914/`, and
  `outputs/m3_unified_reference_condition_audit_step26800_20260914/`.
- Pixel-VAE player audit:
  `docs/reports/m3_pixel_vae_player_audit_20260914.md`.

## Recommended next experiment

Keep the existing Pixel VAE. Before another long run, add a training intervention that
forces actor-specific reference use and validate it with shuffled-reference probes on a
larger fixed cohort. A reasonable minimal candidate is one shared reference adapter used
at a small number of late blocks plus an explicit actor-reference binding objective; it
must first beat step 26500 on fixed validation and produce a materially larger shuffled
delta than sampling noise. Do not infer success from zero-reference ablation alone.

## Git state

- Branch: `h200/m3-unified-reference-tokens`
- Current code includes the unified implementation, resume support, audit tooling, and
  explicit revert commits for the failed late-injection experiment.
- Local tests: 114 passed, 1 warning after the reverts.
- Push is not complete: the shared HTTPS credential belongs to `liqilin-eudaemonia`,
  which GitHub rejects for `xixibuxixi23/Plot`. The repository-local commit author is
  `xixibuxixi23 <xixibuxixi23@users.noreply.github.com>`; no global credential was changed.
