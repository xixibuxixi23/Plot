# M4-State expanded experiment, 2026-09-21

Run: `expanded-20260921-v2`, under the shared project `runs/m4-state/`.
The experiment changes data coverage and supervision after the initial pilot
failed to beat a constant action-distribution baseline. This is not an isolated
single-factor ablation or a full-data training result.

| Node | Profile | Attack positive BCE weight | Window sampler |
| --- | --- | ---: | --- |
| rcz3 | villager_peaceful | 1 | uniform |
| rcz4 | zombie_melee | 8 | 50% attack-containing / 50% other |
| rcz5 | skeleton_swordsman | 8 | 50% attack-containing / 50% other |
| rcz6 | villager_defender | 8 | 50% attack-containing / 50% other |

Each node uses GPU 0 and leaves the user's occupancy process intact.
Parameters: 3,000 updates, batch 8, AdamW 2e-4, BF16, seed 42, same 8.19M
architecture. Positive weighting modifies only positive labels of attack key
(action index 8 / structured key index 7); other keys and negative attack labels
retain their original loss. Seven contract tests include this gradient check.

Select 64 train and 16 val-ID episodes, with seeded shuffled candidate order.
Pick up to 16 uniformly spaced windows per agent/episode. Additionally retain
every legal attack window in the **training** split. Val-ID receives no attack
oversampling. The summary records actual window counts, scenario coverage,
attack steps, and all-zero action counts. Public source data
is unchanged. This validation is still subsampled and is not full-release eval.

Cache preparation uses four CPU threads per node and bounded single-frame
voxel reads. Training preloads the compact state cache, avoiding repeated large
NPZ/video reads. Training item IDs are mapped by item names as in the pilot.

Every 500 steps, evaluate original unweighted structured loss and default-0.5
attack precision/recall/F1, TP/FP/FN on the unresampled validation set. Fit a
smoothed, per-horizon constant marginal baseline from training labels only.
Checkpoint files are `latest.pt`, `best-val-loss.pt`, `best-attack-f1.pt`;
the saved config contains the source hashes and `source/` captures the model,
data adapter and trainer used. Step zero is included in checkpoint comparisons
to expose cases where the random baseline is better than the trained policy.

At the final step, also rotate complete input bundles within each validation
batch, keeping labels fixed. `val_shuffled_inputs` is a state/action-history
dependence diagnostic; it does not isolate geometry from history and samples
within a batch may belong to the same episode.

Original pilot artifacts remain at `pilot-20260921-v1`; source files about to
change were copied to its `source-before-expanded/` directory. This experiment
still leaves guard goals masked and does not integrate the existing closed-loop
scheduler or claim measured M3/M4 concurrency or real-engine task success.

Launcher (run only inside the correctly mapped node shell):

```bash
bash train_scripts/run_state_policy_expanded.sh rcz3 expanded-20260921-v2
```

Use the matching label for each node. The launcher prepares the cache, tests,
trains, then exits; no perpetual job or monitor is installed.
