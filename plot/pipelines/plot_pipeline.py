from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from plot.models.fill import FillNetwork
from plot.geometry import tile_global_coordinates
from plot.world_memory import WorldMemory


@dataclass(frozen=True)
class FillResult:
    requested_tiles: int
    proposed_voxels: int
    committed_voxels: int
    known_voxels: int


class PlotPipeline:
    """M1 -> WorldMemory component used by the full ClosedLoopPipeline."""

    def __init__(self, fill_model: FillNetwork, *, device: str | torch.device = "cpu"):
        self.device = torch.device(device)
        self.fill_model = fill_model.to(self.device).eval()
        self.memory = WorldMemory()

    @torch.no_grad()
    def fill_resident_windows(
        self,
        centers: np.ndarray,
        images: torch.Tensor,
        agent_mask: torch.Tensor | None = None,
    ) -> FillResult:
        centers = np.asarray(centers, dtype=np.int64).reshape(-1, 3)
        if images.ndim != 4 or images.shape[0] != len(centers):
            raise ValueError("images must have shape [A,3,H,W] and match centers")
        if agent_mask is None:
            agent_mask = torch.ones(len(centers), dtype=torch.bool)

        # All proposals read the same committed snapshot. The tile owner is moved
        # to image slot zero; the remaining image set stays shared.
        proposals: dict[tuple[int, int, int], tuple[float, int]] = {}
        proposed = 0
        for owner, center in enumerate(centers):
            context, known = self.memory.read_tile(center)
            fill = ~known
            if not np.any(fill):
                continue
            order = [owner] + [i for i in range(len(centers)) if i != owner]
            logits = self.fill_model(
                torch.from_numpy(context)[None].to(self.device),
                torch.from_numpy(known)[None].to(self.device),
                torch.from_numpy(fill)[None].to(self.device),
                images[order][None].to(self.device),
                agent_mask[order][None].to(self.device),
            )[0]
            probabilities = logits.softmax(0)
            confidence, classes = probabilities.max(0)
            coords = tile_global_coordinates(center)
            local_coords = np.argwhere(fill)
            proposed += len(local_coords)
            for local in local_coords:
                key = tuple(int(v) for v in coords[tuple(local)])
                score = float(confidence[tuple(local)])
                value = int(classes[tuple(local)])
                previous = proposals.get(key)
                if previous is None or score > previous[0]:
                    proposals[key] = (score, value)

        if proposals:
            coordinates = np.asarray(list(proposals), dtype=np.int64)
            values = np.asarray([value for _, value in proposals.values()], dtype=np.int32)
            committed = self.memory.commit_points(coordinates, values)
        else:
            committed = 0
        return FillResult(len(centers), proposed, committed, self.memory.known_voxel_count)
