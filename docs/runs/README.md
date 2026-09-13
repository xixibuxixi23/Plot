# Run registry

This directory is the small, human-readable index of important PLOT runs. It
does not contain checkpoints or bulk metrics. Update the registry when a run
starts, resumes, stops, changes its selected checkpoint, or completes.

## Active runs

No active M1 or M3 training process was found at the 2026-09-13 UTC audit on the
shared cluster or historical H100 host. The assigned next-run hosts are
`zhizhou-avgen-2` (M1) and `js-public` (M3); hardware inspection reports eight
H200 GPUs on each host. A detached worktree prepared for a possible resume is
not an active run.

## Stopped, resumable runs

| Model | Run ID | Code | Machine | Last train step | Resume checkpoint | Original target | Status | Tracking |
|---|---|---|---|---:|---:|---:|---|---|
| M1 | `m1_flow_2nodes_b32_eval10k_v2` | legacy source hashes; compatible PLOT migration at `4b6ce14` | historical shared H200 run; next host `zhizhou-avgen-2` | 1,476,560 | 1,476,000 | 2,000,000 | `stopped_by_user` at 2026-09-13 21:37 UTC; selected checkpoint staged and SHA-256 verified | local logs |
| M3 | `m3_h100_8gpu_entity_reference_stage1_from25000_to27000_20260914` | launch provenance reconstructed as `60c83b9`; no launch manifest was saved | historical 8-GPU H100 run; next host `js-public` | 26,000 | 26,000 | 27,000 | selected checkpoint staged, SHA-256 verified, and strict-loaded on target | [W&B run](https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/7kbesehq) |

The M1 code predates the clean PLOT Git history. Its distributed preflight log
records SHA-256 for source and cache files, but it does not have a single clean
commit. Do not label it with the old `Polis` HEAD, which does not describe the
working tree used by training. Commit `4b6ce14` migrates the maintained M1 flow
code to the `plot` namespace; the immutable step-1,127,000 resume checkpoint
strict-loads with all 288 tensors matched. A multi-GPU resume smoke test is still
required before moving a live M1 run to PLOT. A detached H200 worktree at the
current integrated source is prepared for that validation; no resume has been
launched.

The newly selected step-1,476,000 checkpoint also strict-loads all 288 model
tensors on `zhizhou-avgen-2`. It was saved with 16 ranks, batch 32 per rank, and
`batch_in_epoch=15`. A one-host eight-GPU resume is therefore not topology-exact:
it must explicitly discard the partial epoch. Batch 64 per GPU would preserve
the old effective batch of 512 if it passes the memory/stability smoke test;
batch 32 would change the effective batch to 256. Do not launch until this
choice is recorded.

The M3 lineage continued after the previously recorded step-25,000 run. Commit
`60c83b9` added native-resolution entity-reference attention immediately before
the step-25,000 to step-27,000 stage was launched; that stage produced the
selected complete step-26,000 checkpoint. Commit `33270b9` then added explicit
identity supervision. Its canary reached step 26,420 but saved no checkpoint,
so step 26,000 remains the only valid resume point. Resuming identity
supervision also requires the independently trained player-identity encoder at
step 3,300. The exact launch commit for the entity-reference run is
reconstructed from Git/config timestamps because that historical run lacks a
`run_manifest.json`.

## Completed or superseded runs

Add only runs that are needed to explain the parentage of an active or selected
checkpoint. Machine-local exploratory outputs can remain unlisted.

| Model | Run ID | Final/selected step | Parent or successor | Notes |
|---|---|---:|---|---|
| M3 | `m3_h100_8gpu_view_appearance_stage2_spatial4_from11500_to30000_20260913` | 17,000 | parent of the stopped/resumable block-causal run | View-aware appearance warm start |
| M3 | `m3_h100_8gpu_blockcausal8_cache64_full_from17000_to30000_20260913` | 25,000 | parent of the selected entity-reference step-26,000 checkpoint | W&B run `3pgtybhk`; no longer the preferred resume point |
| M3 | `m3_h100_8gpu_identity_supervision_from26000_to30000_20260914` | no checkpoint (last logged step 26,420) | resume again from selected step 26,000 | Stopped canary; [W&B run](https://wandb.ai/ckx23-tsinghua-university/plot-m3/runs/awnop74b) |

## Required fields for new entries

- unique run ID and model;
- exact Git commit, or an explicit `legacy source hashes` exception;
- dataset repository and immutable revision;
- parent checkpoint path and SHA-256 when resuming;
- logical machine label, output path and W&B URL;
- current/final step, status and selected evaluation metrics;
- Hugging Face path after a checkpoint is published.
