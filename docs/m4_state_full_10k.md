# M4-State full-data 10k run

Run: `full-10k-20260921-v3`

Remote root: `/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d/runs/m4-state/full-10k-20260921-v3`

| Node | Independent network |
| --- | --- |
| rcz3 | villager_peaceful |
| rcz4 | zombie_melee |
| rcz5 | skeleton_swordsman |
| rcz6 | villager_defender |

Each node runs `Plot/train_scripts/run_state_policy_full.sh NODE full-10k-20260921-v3`.
The detached job first prepares its full cache, then trains 10,000 optimizer updates from scratch.
Existing GPU occupancy processes remain untouched. Each training job uses GPU 0 only.

## Data definition

- Source: `data/polis_two_player_fixed_skins_complete_20260917_360p` (25,691 train index rows; 1,352 val_id index rows).
- All eligible episodes for each profile; all legal stride-8 windows. No episode limit or per-episode window cap.
- Original train/val_id split retained; no cross-split episode reuse.
- Existing usable-episode, NPC supervision, terminal, health-valid and alive masks retained.
- Input: current voxel crop and eight committed engine-state observations; labels: next eight actions. No RGB.
- Full means full eligible sampling pool, not guaranteed exhaustive coverage within 10k updates. Combat sampling uses replacement with equal attack/nonattack window mass.
- Actual eligible counts are written to each cache's `summary.json` once preparation completes.

## Training and evaluation

- Network architecture unchanged (~8.19M parameters), batch 8, AdamW lr 2e-4, BF16, seed 42.
- Peaceful: uniform sampling, positive attack weight 1. Combat: balanced attack-window sampling, positive attack weight 8 (as in expanded experiment).
- Full natural-distribution validation every 1,000 updates and at step 0; final shuffled-input diagnostic.
- Training diagnostic loss uses a fixed 1,024-window subset only. This does not limit training or validation data.
- Constant baseline fitted using all training labels and evaluated on the full validation set.
- Save latest, best validation-loss and best attack-F1 weights separately.
- Loss and F1 use recorded actions, not engine behavior success. Full-data validation scores should not be directly compared to the earlier small validation subset as if they were identical tests.
- `models/PROFILE/COMPLETE.json` is the completion marker; `logs/NODE.log` includes preparation and training progress.

## Implementation / verification

Full preparation decompresses voxel frames in one forward pass per episode and writes compressed per-window caches; eight episode workers per node. Training streams compressed caches with four loader workers and does not preload the full dataset. Action-label arrays are stored separately to avoid loading geometry for balancing and baseline calculation.

All seven tests passed on all four nodes, including direct-vs-sequential compressed-cache equivalence over multiple anchors/targets. A two-update CUDA smoke run with the updated trainer and four loader workers completed on rcz4 (`smoke-zombie/COMPLETE.json`); this is a functional check, not the full-data experiment result.

Status at handoff: four detached full-data preparation jobs running; no full-data 10k result yet.
