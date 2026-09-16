# M3 fixed-skin pilot (2026-09-15)

## Decision

Fixed skins are a useful simplification, but 250 additional steps do not yet
solve player blur. On three held-out fixed-skin 64-frame rollouts, player-region
L1 improved on every probe and improved by 5.88% as an unweighted macro
average. The output recovered more of the expected player color and silhouette
in the difficult S02 close-ups. The result remains visibly blurred and the
full-frame L1 regressed on two of the three probes, so the next checkpoint is a
longer balanced S01--S07 continuation rather than a claim that the issue is
finished.

The frozen Pixel VAE is not the active bottleneck. On the same maximum-player
frames its player-region reconstruction L1 was 0.0133--0.0269, substantially
below M3's 0.0483--0.1472 after the pilot.

## Protocol

- Starting checkpoint: step 35250.
- Pilot checkpoint: step 35500 (250 steps).
- Training snapshot: 2,042 completed episodes, 1,024 S01 candidates plus 1,024
  S02 candidates, with six noncanonical-skin episodes removed.
- Model: all 461,335,893 parameters trainable.
- Hardware: 8x H200; batch 1/GPU; effective global batch 8; BF16.
- Player losses: latent region upweight 8, RGB player L1 weight 8, edge weight
  2, four decoded loss frames.
- Appearance path: unified reference tokens, 16x8 grid, target geometry, injected
  at DiT blocks 3, 7, and 11.
- No S11 sampling, shuffled reference, zero reference, counterfactual loss, or
  prefix masking.
- Evaluation: normal rollout only, identical held-out windows and seeds before
  and after fine-tuning. The three held-out episode IDs are explicitly excluded
  from the larger follow-up snapshot.

## Held-out results

| Probe | Player L1 @35250 | Player L1 @35500 | Relative change | Global L1 @35250 | Global L1 @35500 |
|---|---:|---:|---:|---:|---:|
| S01 build near | 0.04950 | 0.04828 | -2.45% | 0.03098 | 0.03199 |
| S02 motion near A | 0.16081 | 0.14505 | -9.80% | 0.06396 | 0.05999 |
| S02 motion near B | 0.15151 | 0.14721 | -2.84% | 0.05417 | 0.05698 |
| Macro average | 0.12061 | 0.11351 | -5.88% | -- | -- |
| Player-pixel weighted | 0.12964 | 0.12192 | -5.95% | -- | -- |

Pixel VAE reconstruction on the same maximum-player frame:

| Probe | VAE player L1 | VAE global L1 |
|---|---:|---:|
| S01 build near | 0.01642 | 0.01814 |
| S02 motion near A | 0.01332 | 0.01362 |
| S02 motion near B | 0.02695 | 0.02094 |

## Artifacts

- Pilot checkpoint:
  `outputs/m3_fixed_skin_s01_s02_pilot_35250_35500_20260915/step_0035500.pt`
- Numeric comparison:
  `outputs/m3_fixed_skin_quality_35250_vs_35500_20260915/comparison_summary.json`
- Normal GT/generated videos:
  `outputs/m3_fixed_skin_quality_35250_vs_35500_20260915/{baseline,finetuned}/`
- Three-way stills and player crops:
  `outputs/m3_fixed_skin_quality_35250_vs_35500_20260915/previews/`
- Exact-frame VAE ceiling:
  `outputs/m3_fixed_skin_quality_35250_vs_35500_20260915/vae_ceiling/`
- Larger immutable follow-up snapshot:
  `derived/fixed_skin_s01_s07_balanced_20260915/` (9,800 episodes,
  984,012 valid windows)

## Follow-up gate

Continue step 35500 to step 36000 on the balanced immutable S01--S07 snapshot,
then repeat the exact same normal-only probes. Collection is still running and
S08--S10 are not included in this gate.
