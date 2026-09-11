"""Capture each committed world state before advancing to the next transition."""
from __future__ import annotations

import numpy as np
import torch

from plot.data.renderer_dataset import raster_camera


class RendererMemoryBlock:
    """One target resident, one fixed anchor, eight successive memory snapshots.

    Call append after each M2 commit and required M1 fill, not eight times after
    the last transition. Arrays are copied immediately, so later writes cannot
    retroactively alter an earlier render condition.
    """

    def __init__(self, anchor):
        self.anchor = np.asarray(anchor, np.int64).copy()
        if self.anchor.shape != (3,):
            raise ValueError("anchor must be an integer world coordinate")
        self.frames = []

    def append(self, memory, *, transition_index, camera_world, camera_direction, fov_x):
        if len(self.frames) >= 8:
            raise ValueError("render block already contains eight states")
        if self.frames and transition_index != self.frames[-1][0] + 1:
            raise ValueError("memory snapshots must follow contiguous transitions")
        blocks, known = memory.read_tile(self.anchor)
        if not known.all():
            raise ValueError("M1 must fill UNFILLED voxels before live M3 rendering")
        camera = raster_camera(np.asarray(camera_world), np.asarray(camera_direction),
                               np.asarray(fov_x), self.anchor)
        self.frames.append((transition_index, blocks.copy(), known.copy(), camera.copy()))

    def conditions(self, *, device="cpu"):
        if len(self.frames) != 8:
            raise ValueError("eight committed states are required")
        return {key: torch.as_tensor(np.stack([row[i] for row in self.frames]), device=device)[None]
                for key, i in (("voxel_classes", 1), ("voxel_known", 2), ("raster_camera", 3))}
