# Run registry

This directory is the small, human-readable index of important PLOT runs. It
does not contain checkpoints or bulk metrics. Update the registry when a run
starts, resumes, stops, changes its selected checkpoint, or completes.

## Active runs

| Model | Run ID | Code | Machine | Start/resume | Latest observed | Output | Tracking |
|---|---|---|---|---:|---:|---|---|
| M1 | `m1_flow_2nodes_b32_eval10k_v2` | legacy source hashes; compatible PLOT migration at `4b6ce14` | `H200` legacy 2-node run | 1,127,000 | 1,476,560 at 2026-09-13 21:41 UTC | `Polis/outputs/m1_flow_2nodes_b32_eval10k_v2` | local logs |
| M3 | `m3_h100_8gpu_blockcausal8_cache64_full_from17000_to30000_20260913` | process loaded `5e874fde7e32da3b1cec3a08ecb5f2ef305754bb` | `H100` 8-GPU run | 17,000 | 25,000 at 2026-09-13 21:41 UTC | `/data/huangyh/hxh/Plot/outputs/m3_h100_8gpu_blockcausal8_cache64_full_from17000_to30000_20260913` | [W&B run](https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/3pgtybhk) |

The M1 code predates the clean PLOT Git history. Its distributed preflight log
records SHA-256 for source and cache files, but it does not have a single clean
commit. Do not label it with the old `Polis` HEAD, which does not describe the
working tree used by training. Commit `4b6ce14` migrates the maintained M1 flow
code to the `plot` namespace; the immutable step-1,127,000 resume checkpoint
strict-loads with all 288 tensors matched. A multi-GPU resume smoke test is still
required before moving a live M1 run to PLOT.

The M3 branch is a linear series through BF16 alignment, pixel-health
supervision, deep-condition experiments, view-aware appearance and later
identity-supervision work. The active Python processes loaded `5e874fd` at
launch. The physical worktree subsequently advanced while those processes were
alive, so its current HEAD must not be reported as the run commit. This is a
documented legacy exception; future jobs use detached run worktrees.

## Completed or superseded runs

Add only runs that are needed to explain the parentage of an active or selected
checkpoint. Machine-local exploratory outputs can remain unlisted.

| Model | Run ID | Final/selected step | Parent or successor | Notes |
|---|---|---:|---|---|
| M3 | `m3_h100_8gpu_view_appearance_stage2_spatial4_from11500_to30000_20260913` | 17,000 | parent of the active block-causal run | View-aware appearance warm start |

## Required fields for new entries

- unique run ID and model;
- exact Git commit, or an explicit `legacy source hashes` exception;
- dataset repository and immutable revision;
- parent checkpoint path and SHA-256 when resuming;
- logical machine label, output path and W&B URL;
- current/final step, status and selected evaluation metrics;
- Hugging Face path after a checkpoint is published.
