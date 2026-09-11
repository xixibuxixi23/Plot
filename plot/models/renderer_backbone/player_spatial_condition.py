"""Lightweight screen-space condition for other players."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PlayerSpatialCondition(nn.Module):
    """Project every other player to a soft screen-space rectangle.

    The branch only needs state already present in the dataset plus the existing
    four-view identity appearance embedding. It deliberately does not model a
    skeleton or require body-part annotations.
    """

    def __init__(
        self,
        appearance_dim: int,
        output_channels: int = 16,
        height: int = 36,
        width: int = 64,
        player_height: float = 1.8,
        player_aspect: float = 0.45,
        edge_sharpness: float = 2.0,
    ) -> None:
        super().__init__()
        if output_channels < 3:
            raise ValueError("output_channels must be at least 3")
        self.output_channels = output_channels
        self.height = height
        self.width = width
        self.player_height = player_height
        self.player_aspect = player_aspect
        self.edge_sharpness = edge_sharpness
        self.appearance_projection = nn.Linear(appearance_dim, output_channels - 2, bias=False)

    def _project(self, points, camera_position, forward, right, down, tan_half_fov_x):
        relative = points - camera_position
        depth = (relative * forward).sum(dim=-1)
        camera_x = (relative * right).sum(dim=-1)
        camera_y = (relative * down).sum(dim=-1)
        safe_depth = depth.clamp_min(1e-4)
        tan_half_fov_y = tan_half_fov_x * (self.height / self.width)
        u = (camera_x / safe_depth / tan_half_fov_x + 1.0) * (self.width - 1) / 2.0
        v = (camera_y / safe_depth / tan_half_fov_y + 1.0) * (self.height - 1) / 2.0
        return u, v, depth

    def forward(self, cond: dict, appearance_embedding: torch.Tensor) -> torch.Tensor:
        position = cond["player_position"]
        camera_position = cond["camera_position"]
        camera_direction = cond["camera_direction"]
        camera = cond["camera"]
        player_valid = cond.get("player_valid")

        bsz, timesteps, agents, _ = position.shape
        if camera_position.shape != position.shape or camera_direction.shape != position.shape:
            raise ValueError("camera_position and camera_direction must match player_position")
        if appearance_embedding.shape[:2] != (bsz, agents):
            raise ValueError("appearance_embedding must have shape [B,A,D]")
        if player_valid is None:
            player_valid = torch.ones((bsz, timesteps, agents), device=position.device, dtype=torch.bool)
        else:
            player_valid = player_valid.to(device=position.device, dtype=torch.bool)

        dtype = position.dtype
        forward = F.normalize(camera_direction, dim=-1, eps=1e-6)
        world_down = torch.tensor((0.0, 0.0, -1.0), device=position.device, dtype=dtype)
        right = F.normalize(torch.cross(world_down.expand_as(forward), forward, dim=-1), dim=-1, eps=1e-6)
        down = F.normalize(torch.cross(forward, right, dim=-1), dim=-1, eps=1e-6)

        # [B,T,target,source,3]
        camera_position = camera_position[:, :, :, None]
        forward_t = forward[:, :, :, None]
        right_t = right[:, :, :, None]
        down_t = down[:, :, :, None]
        feet = position[:, :, None].expand(-1, -1, agents, -1, -1)
        head = feet.clone()
        head[..., 2] += self.player_height
        tan_half_fov_x = torch.tan(camera[..., -1].clamp(0.05, 3.0) / 2.0)[:, :, :, None]

        foot_u, foot_v, foot_depth = self._project(
            feet, camera_position, forward_t, right_t, down_t, tan_half_fov_x
        )
        head_u, head_v, head_depth = self._project(
            head, camera_position, forward_t, right_t, down_t, tan_half_fov_x
        )
        center_u = (foot_u + head_u) / 2.0
        center_v = (foot_v + head_v) / 2.0
        box_h = (foot_v - head_v).abs().clamp(1.0, float(self.height))
        box_w = (box_h * self.player_aspect).clamp(1.0, float(self.width))
        depth = (foot_depth + head_depth) / 2.0

        grid_y = torch.arange(self.height, device=position.device, dtype=dtype)
        grid_x = torch.arange(self.width, device=position.device, dtype=dtype)
        dy = (grid_y.view(1, 1, 1, 1, self.height, 1) - center_v[..., None, None]).abs()
        dx = (grid_x.view(1, 1, 1, 1, 1, self.width) - center_u[..., None, None]).abs()
        mask = torch.sigmoid((box_h[..., None, None] / 2.0 - dy) * self.edge_sharpness)
        mask = mask * torch.sigmoid((box_w[..., None, None] / 2.0 - dx) * self.edge_sharpness)

        source_valid = player_valid[:, :, None, :]
        target_valid = player_valid[:, :, :, None]
        not_self = ~torch.eye(agents, device=position.device, dtype=torch.bool)[None, None]
        visible = source_valid & target_valid & not_self & (depth > 0.05)
        mask = mask * visible[..., None, None].to(dtype)

        inverse_depth = 1.0 / (1.0 + depth.clamp_min(0.0))

        appearance = torch.tanh(self.appearance_projection(appearance_embedding.to(dtype)))
        appearance = appearance[:, None, None].expand(-1, timesteps, agents, -1, -1)
        source_features = torch.cat(
            (
                torch.ones_like(depth)[..., None],
                inverse_depth[..., None],
                appearance,
            ),
            dim=-1,
        )
        spatial = (mask[..., None, :, :] * source_features[..., :, None, None]).sum(dim=3)
        return spatial.contiguous()
