# State-conditioned M4 TextAgent pilot

The existing M4 already routes a `language_builder` family through shared-task and current-agent text cross-attention, but depends on frozen M3 video features. This experiment adds an independent state-conditioned `language_builder` profile without removing or changing the old visual branch.

## Architecture

Reuse the NPC state trunk: current 48-cubed known-masked geometry, eight committed resident states, own incoming actions, and eight future action queries. Append two banks of frozen T5 token features (shared task and current-agent instruction) projected to hidden size 256 with separate role embeddings. Each future action query attends to state and text together. Outputs retain the 23-dimensional structured action contract, including dig, place, hotbar and mouse actions.

Text IDs are only frozen-feature lookup keys, not learned task-ID embeddings. Source text follows the original M4 contract: manifest `task_text`; current instruction from `agent_task_texts` or `agent_subtasks`, otherwise shared text. No teacher target coordinates, future state, target offsets or completion labels are passed to the model. Goal remains masked.

Use the matched catalog/cache in `downloads/plot-checkpoints-main/m4_text/`; catalog equals `Plot/checkpoints/m4_text/text_catalog.json`. Its encoder metadata identifies `google/t5-v1_1-base`, 64 tokens, hidden size 768. Cache coverage is checked before sample preparation; missing instructions fail instead of becoming empty text.

## Pilot configuration

- Node rcz3, GPU 0, existing occupancy job untouched.
- Run root: `/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d/runs/m4-state/text-pilot-20260922-v1`.
- 256 shuffled eligible training episodes, 64 independent val_id episodes, seed 42.
- At most 16 stride-8 windows per target agent per episode; language_policy_train_mask, alive, valid health, usable episode and termination filters retained.
- 3,000 updates from scratch; batch 8, AdamW 2e-4, BF16, uniform sampling, no combat oversampling/positive weighting.
- Full selected validation set every 500 steps; fixed 1,024-window train diagnostic.
- Final diagnostics: full inputs shuffled; only shared/current text shuffled together; all text attention masked out. Text-only shuffle reports the fraction with changed current text because repeated instructions may remain unchanged.
- Place and dig/attack precision, recall and F1 are recorded alongside loss. The legacy `attack_*` fields mean action index 8 (dig/attack), NOT necessarily combat for this profile.

## Inference and limits

`CachedInstructionEncoder(catalog, cache)(shared, current)` returns text_condition tensors. Pass these to `StatePolicySession.capture(memory, text_condition=...)` then `act`. Capture clones instructions so a later command change cannot mutate an already dispatched state snapshot.

This inference helper accepts cached instructions only. Unseen instructions require matching frozen-T5 encoding; there is no automatic random/hash fallback. The current trial uses English manifest instructions, not Chinese paraphrase augmentation. It does not establish arbitrary-language generalization, dynamic instruction-switch execution, or in-engine construction success. Dataset states are engine ground truth, not M2 predictions. No M3 concurrency scheduler or existing ClosedLoopPipeline routing was changed.

Existing NPC checkpoint configs remain compatible (`text_hidden_size` defaults to zero). Tests cover non-language behavior plus text sensitivity, nonzero text-projection gradients, padding-mask invariance and finite all-text-masked inference.

Verification on rcz3: all eight tests passed; a real cached language sample completed CUDA BF16 forward/backward with nonzero text-projection gradient and `[1,8,23]` decoded actions. Model size: 8,387,718 parameters. An existing full-10k NPC checkpoint also reloaded strictly under the updated code. These are functional checks, not language-following performance results.
