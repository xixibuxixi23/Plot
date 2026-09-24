# M4 large state policies: two two-node 100k runs

Run root: `/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d/runs/m4-state/large-100k-20260922-v1`.

| Group | Model | Parallelism |
| --- | --- | --- |
| rcz3 + rcz4 | zombie_melee | one model, 16 DDP ranks, eight GPUs per node |
| rcz5 + rcz6 | language_builder (text construction) | one model, 16 DDP ranks, eight GPUs per node |

Existing GPU occupancy jobs are not stopped or modified. Node addresses in the launcher were observed inside the exact authorized taiji_client targets and are used for distributed rendezvous, not direct SSH.

## Architecture

Separate checkpoints; no parameter sharing across the zombie and builder models. Width 768, 12 heads, 12 deep policy blocks; MLP expansion 4. Approximate trainable size is 117M for zombie and 146M for builder (exact count changes slightly with item-vocabulary size). Previous state pilots were approximately 8M.

The geometric path directly reuses M3 code: block/unknown embedding width 32, exact M3 raster-camera conversion, `VoxelMeshRasterizer` on the 48-cubed grid, 192 depth layers at 36x64, `DepthPatchEmbedder` with kernel 6/stride 4/output 16 and eight sinusoidal depth features. Its 47 depth groups flatten to 752 channels; a 2x2 `PixelPatchEmbedder` yields 576 geometric tokens. There is no RGB/image-latent input. Weights are trained independently from scratch; this is reuse of M3's encoding method, not M3 weights or denoising blocks. Geometry uses the current target observation crop, not future geometry or a teacher target map.

Main self-attention sequence: 576 geometry tokens, eight self-history tokens, eight future-action queries. Every block additionally cross-attends to individual other-resident history tokens (state, held item, type, already executed actions); residents remain an unordered set, and absent/out-of-range residents are masked. A null token makes empty neighborhoods safe. Builder blocks have a separate text cross-attention to frozen T5 shared-task and current-agent token banks. Text is injected at every layer, not only at the final decoder. Queries output eight structured future actions. Q/K RMS normalization, pre-LayerNorm and activation checkpointing are enabled.

No stable actor-ID lookup or hidden teacher target offsets are model inputs. Historical resident actions end at the incoming transition to the current state. There is no future action/state leakage. Only eight state frames are retained; long-term planner memory and task-origin supervision are not added in this experiment. This trainer is not wired into the live ClosedLoopPipeline.

## Data and training

Source is `data/polis_two_player_fixed_skins_complete_20260917_360p`. Use all eligible episodes/windows for the selected profile from the original train/val_id splits, anchor stride 8, no episode/window cap. Keep usable-episode, policy-supervision, health/alive and terminal filters. Cache includes the exact observed FOV-based M3 camera and all residents' incoming historical actions. Builder instructions follow the original shared/per-agent manifest contract and matched frozen T5 cache. Missing text fails explicitly.

Each group targets 100,000 optimizer updates. Per-GPU batch 4, global batch 64; approximately 6.4M sample presentations per run, with normal epoch reshuffling. BF16, AdamW 2e-4, weight decay .01, 1,000-step warmup then cosine decay to 10% of peak LR, clip global gradient norm to 1. Training is uniform over windows, not attack-window replacement sampling. Key loss has weight 2 and positive dig/attack weight 4; builder also has positive place weight 4. Future-horizon weights are 8 through 1. These are logged explicitly; validation loss is unweighted on the natural held-out distribution.

Validation covers every eligible held-out window exactly once across ranks, with no padded duplicate samples, first at update 1,000, then every 5,000 and at 100,000. Reports separate key/hotbar/mouse losses and dig/attack/place TP/FP/FN, precision/recall/F1 at thresholds .1/.3/.5. Final builder evaluation also masks text and mismatches text after mixing episodes, reporting the actual changed-text fraction. These are imitation metrics, not in-engine task success.

Checkpoints: atomic latest every 1,000, best unweighted validation loss, milestones every 10,000. Checkpoints contain optimizer, step, model/data/training config, best loss and all ranks' RNG states. Resume requires the same world size, batch size, dataset and schedule; call the trainer with `--resume`. Source snapshots and hashes are recorded when training starts. Existing smaller-run artifacts are preserved.

## Preflight and operation

`prepare_state_large.sh PROFILE` builds the full state cache. `run_state_large_node.sh NODE smoke` runs three full-size synthetic CUDA updates on the exact two-node/16-rank topology and checks depth-stack equivalence with M3. Both groups passed this check, with finite gradients and successful collective reduction on all 16 ranks. Synthetic batch-one memory peaks were approximately 2.8 GiB for zombie and 3.4 GiB for builder; these are not production memory or performance estimates.

`run_large_trainer_check.sh 0/1` performs a separate two-update full-size builder test on real bounded data, writes checkpoints, and reloads them with `--resume`. It is not part of either 100k run. Production launcher requires a successful group smoke, waits for full-cache completion (bounded to 24h), refuses to train after a preparation traceback, and checks cache coverage/schema/label alignment before launching.

Production logs are `logs/train-rczN.log`; prep logs are `logs/prepare-PROFILE.log`; configs and checkpoints are under `models/PROFILE/`. `COMPLETE.json` with steps 100000 is the completion marker. Starting preparation or a waiting launcher is not evidence that parameter updates have started.

## Launch handoff

Both group smoke reports exist and all 16 ranks passed for each group. Ten unit tests passed, including the additional M3-camera/incoming-action adapter check. The real-data 16-rank trainer completed two updates, full selected validation and final text probes, saved its checkpoint, and successfully reloaded it with the optimizer/RNG state (`TRAINER_CHECK_AND_RESUME_PASS` on both nodes). That reload occurred at the completed two-step boundary; it verifies restoration, not additional post-resume optimization steps.

All four production launcher processes were verified running. At handoff they were waiting for full-cache completion, not yet performing the 100k parameter updates. Cache generation runs on rcz3 for zombie and rcz5 for language_builder. The launchers automatically proceed through readiness verification to training when their full cache is complete. Existing pilot/full-10k runs and occupancy processes were preserved.
