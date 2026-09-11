from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticFillDataset(Dataset):
    """Small deterministic dataset for contract and training smoke tests."""

    def __init__(self, length: int = 16, size: int = 16, agents: int = 2, classes: int = 8):
        self.length, self.size, self.agents, self.classes = length, size, agents, classes

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(index)
        coords = np.stack(np.meshgrid(*([np.arange(self.size)] * 3), indexing="ij"))
        target = ((coords[2] < self.size // 2).astype(np.int64) + 1)
        target = (target + ((coords[0] + index) // max(1, self.size // 4))) % self.classes
        known = np.zeros_like(target, dtype=bool)
        if index % 2:
            known[: self.size // 2] = True
        fill = ~known
        images = rng.random((self.agents, 3, 32, 48), dtype=np.float32)
        camera_position = rng.normal(0, 0.1, (self.agents, 3)).astype(np.float32)
        camera_direction = rng.normal(0, 1, (self.agents, 3)).astype(np.float32)
        camera_direction /= np.linalg.norm(camera_direction, axis=-1, keepdims=True)
        return {
            "voxel_context": torch.from_numpy(np.where(known, target, 0)).long(),
            "known_mask": torch.from_numpy(known),
            "fill_mask": torch.from_numpy(fill),
            "target": torch.from_numpy(target).long(),
            "target_valid": torch.ones_like(torch.from_numpy(fill)),
            "images": torch.from_numpy(images),
            "agent_mask": torch.ones(self.agents, dtype=torch.bool),
            "camera_position": torch.from_numpy(camera_position),
            "camera_direction": torch.from_numpy(camera_direction),
            "camera_valid": torch.ones(self.agents, dtype=torch.bool),
            "fov_x": torch.full((self.agents,), 1.82),
            "fov_y": torch.full((self.agents,), 1.26),
        }
