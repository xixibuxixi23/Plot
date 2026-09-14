"""Grouped S11 renderer samples for causal appearance supervision."""

from __future__ import annotations

from collections import defaultdict

import torch
from torch.utils.data import Dataset

from .renderer_dataset import TextAgentRendererDataset


class AppearanceCounterfactualRendererDataset(Dataset):
    """Return all skin variants of one trajectory window as one item.

    The ordinary renderer collator flattens the returned list while preserving
    its order. This lets flow training reuse identical noise/time for each
    variant and prevents DDP from separating counterfactual siblings.
    """

    def __init__(self, root, vocabulary, *, variants_per_group=4, **kwargs):
        if variants_per_group < 2:
            raise ValueError("counterfactual groups need at least two variants")
        if int(kwargs.get("targets_per_window", 1)) != 1:
            raise ValueError("counterfactual grouping requires targets_per_window=1")
        self.base = TextAgentRendererDataset(root, vocabulary, **kwargs)
        self.group_size = int(variants_per_group)
        self.vocabulary = self.base.vocabulary
        self.item_vocabulary = self.base.item_vocabulary
        grouped = defaultdict(list)
        for episode_id, start, target in self.base.index:
            if isinstance(target, (tuple, list)):
                raise ValueError("counterfactual base index must contain scalar targets")
            _, manifest = self.base.episodes[int(episode_id)]
            group_id = manifest.get("appearance_group_id")
            variant = manifest.get("appearance_variant_index")
            if group_id is None or variant is None:
                continue
            relative_start = int(start) - int(manifest.get("model_start_observation", 0))
            grouped[(str(group_id), relative_start, int(target))].append(
                (int(variant), int(episode_id), int(start), int(target))
            )
        self.index = []
        for key in sorted(grouped):
            rows = sorted(grouped[key])
            variants = [row[0] for row in rows]
            if variants != list(range(self.group_size)):
                raise ValueError(
                    f"appearance group {key} has variants {variants}, expected "
                    f"0..{self.group_size - 1}"
                )
            self.index.append(tuple(row[1:] for row in rows))
        if not self.index:
            raise ValueError("no complete S11 appearance counterfactual groups found")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        samples = [self.base._read_target(*row) for row in self.index[index]]
        reference = samples[0]["conditions"]

        # Each variant is captured by a fresh game-server process. Minetest's
        # numeric content IDs can shift even though S11's named blocks and all
        # geometry are unchanged. A counterfactual batch must differ only in
        # appearance, so use variant 0's already-encoded voxel condition for
        # every sibling. Keep the raw recordings untouched for auditability.
        for sample in samples[1:]:
            conditions = sample["conditions"]
            for name, value in reference.items():
                if name in {"voxel_classes", "voxel_known", "player_skin", "player_reference"}:
                    continue
                if isinstance(value, torch.Tensor) and not torch.equal(value, conditions[name]):
                    raise ValueError(
                        f"S11 counterfactual group changed non-appearance condition {name!r}"
                    )
            conditions["voxel_classes"] = reference["voxel_classes"]
            conditions["voxel_known"] = reference["voxel_known"]
        return samples
