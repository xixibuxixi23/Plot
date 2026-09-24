"""Consecutive player-only M2 windows for differentiable long rollouts."""
from __future__ import annotations

import random

from torch.utils.data import IterableDataset, get_worker_info

from .player_transition_stream import episode_player_windows, collate_player_transition


def episode_player_rollouts(path, items, split, rollout_blocks=8, vocabulary=None):
    windows = list(episode_player_windows(path, items, split, stride=8, vocabulary=vocabulary))
    if rollout_blocks < 1:
        raise ValueError("rollout_blocks must be positive")
    run = []
    for sample in windows:
        if run and int(sample["metadata"]["start"]) != int(run[-1]["metadata"]["start"]) + 8:
            run = []
        run.append(sample)
        if len(run) == rollout_blocks:
            yield run
            # Non-overlapping sequences keep I/O and epoch size bounded.
            run = []


class PlayerRolloutStream(IterableDataset):
    def __init__(self, index_payload, split, items, rollout_blocks=8, seed=7,
                 rank=0, world_size=1, fixed_order=False, class_to_raw=None):
        self.root = index_payload["dataset_root"]
        self.records = index_payload["splits"][split]
        self.split = split
        self.items = dict(items)
        self.rollout_blocks = int(rollout_blocks)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.fixed_order = bool(fixed_order)
        if class_to_raw is None:
            self.vocabulary = None
        else:
            from .fill_dataset import BlockVocabulary
            self.vocabulary = BlockVocabulary(tuple(class_to_raw))
        self.epoch = 0

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        workers = worker.num_workers if worker else 1
        rng = random.Random(self.seed + (0 if self.fixed_order else self.epoch))
        order = list(range(len(self.records)))
        rng.shuffle(order)
        self.epoch += 1
        shard = self.rank * workers + worker_id
        shards = self.world_size * workers
        for index in order[shard::shards]:
            record = self.records[index]
            yield from episode_player_rollouts(
                f"{self.root}/{record['path']}", self.items, self.split,
                self.rollout_blocks, self.vocabulary)


def collate_player_rollouts(sequences):
    blocks = len(sequences[0])
    if any(len(sequence) != blocks for sequence in sequences):
        raise ValueError("rollout sequences must have equal length")
    return [collate_player_transition([sequence[block] for sequence in sequences])
            for block in range(blocks)]
