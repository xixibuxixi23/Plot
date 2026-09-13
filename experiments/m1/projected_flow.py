"""Camera projection of frozen image features into existing flow queries."""

import torch
from torch import nn
from torch.nn import functional as F
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT


def project_points(points, position, direction, fov_x, fov_y):
    forward = F.normalize(direction.float(), dim=-1)
    up = torch.zeros_like(forward)
    up[..., 2] = 1
    right = torch.linalg.cross(forward, up)
    alternate = torch.zeros_like(forward)
    alternate[..., 1] = 1
    right = F.normalize(
        torch.where(
            right.square().sum(-1, keepdim=True) < 1e-8,
            torch.linalg.cross(forward, alternate),
            right,
        ),
        dim=-1,
    )
    up = F.normalize(torch.linalg.cross(right, forward), dim=-1)
    delta = points[None, None] - position.float()[:, :, None]
    depth = (delta * forward[:, :, None]).sum(-1)
    u = (
        (delta * right[:, :, None]).sum(-1)
        / depth.clamp_min(1e-8)
        / torch.tan(fov_x.float()[:, :, None] / 2)
    )
    v = (
        -(delta * up[:, :, None]).sum(-1)
        / depth.clamp_min(1e-8)
        / torch.tan(fov_y.float()[:, :, None] / 2)
    )
    grid = torch.stack([u, v], -1)
    valid = (depth > 1e-6) & (grid.abs() <= 1).all(-1) & torch.isfinite(grid).all(-1)
    return torch.where(valid[..., None], grid, torch.zeros_like(grid)), valid


def projected_features(data):
    # Volume [X,Y,Z], patch centre indices 3.5,11.5,...,43.5 in a 48 cube.
    cond = data["image_rays"].float()
    coord = (torch.arange(6, device=cond.device).float() + 0.5) / 6 - 0.5
    points = torch.stack(torch.meshgrid(coord, coord, coord, indexing="ij"), -1).reshape(-1, 3)
    grid, valid = project_points(
        points, data["camera_position"], data["camera_direction"], data["fov_x"], data["fov_y"]
    )
    valid = valid & (data["agent_mask"] & data["camera_valid"])[:, :, None]
    b, v = cond.shape[:2]
    features = F.grid_sample(
        cond[:, :, :16].reshape(b * v, 16, 36, 64),
        grid.reshape(b * v, 216, 1, 2),
        align_corners=False,
        padding_mode="border",
    )
    features = features.reshape(b, v, 16, 216).permute(0, 1, 3, 2)
    features = torch.where(valid[..., None], features, torch.zeros_like(features))
    return features.sum(1) / valid.sum(1).clamp_min(1)[..., None]


class ProjectedFlow(MultiViewVoxelDiT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.projection_adapter = nn.Linear(16, self.config["width"], bias=False)
        nn.init.zeros_(self.projection_adapter.weight)

    def forward(self, x, t, image_rays, view_valid, projection):
        return super().forward(x, t, image_rays, view_valid, self.projection_adapter(projection))
