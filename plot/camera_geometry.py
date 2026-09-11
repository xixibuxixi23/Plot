from __future__ import annotations

import torch
from torch.nn import functional as F


def camera_to_world_rotation(forward: torch.Tensor) -> torch.Tensor:
    """Build roll-free camera-to-world rotations with columns right, up, forward."""
    forward = F.normalize(forward, dim=-1, eps=1e-6)
    world_up = torch.zeros_like(forward)
    world_up[..., 2] = 1.0
    right = torch.linalg.cross(forward, world_up, dim=-1)
    alternate_up = torch.zeros_like(forward)
    alternate_up[..., 1] = 1.0
    alternate_right = torch.linalg.cross(forward, alternate_up, dim=-1)
    right = torch.where(
        right.square().sum(-1, keepdim=True) < 1e-8, alternate_right, right
    )
    right = F.normalize(right, dim=-1, eps=1e-6)
    up = F.normalize(torch.linalg.cross(right, forward, dim=-1), dim=-1, eps=1e-6)
    return torch.stack((right, up, forward), dim=-1)


def canonicalize_cameras(
    position: torch.Tensor,
    direction: torch.Tensor,
    *,
    position_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express cameras relative to view zero, eliminating global SE(3) gauge."""
    rotation = camera_to_world_rotation(direction)
    reference_rotation = rotation[:, :1]
    relative_rotation = reference_rotation.transpose(-1, -2) @ rotation
    relative_translation = (
        reference_rotation.transpose(-1, -2)
        @ ((position - position[:, :1]) * position_scale).unsqueeze(-1)
    ).squeeze(-1)
    return relative_translation, relative_rotation


def matrix_to_rotation_6d(rotation: torch.Tensor) -> torch.Tensor:
    """Encode a rotation by its first two columns."""
    return torch.cat((rotation[..., :, 0], rotation[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Continuous 6D rotation representation with column-vector convention."""
    first = F.normalize(value[..., :3], dim=-1, eps=1e-6)
    second_raw = value[..., 3:]
    second = second_raw - (first * second_raw).sum(-1, keepdim=True) * first
    second = F.normalize(second, dim=-1, eps=1e-6)
    third = torch.linalg.cross(first, second, dim=-1)
    return torch.stack((first, second, third), dim=-1)


def canonicalize_points(
    points_relative_world: torch.Tensor,
    reference_direction: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-axis relative points into the first camera coordinate frame."""
    reference_rotation = camera_to_world_rotation(reference_direction)
    return torch.einsum(
        "bij,bvhwj->bvhwi", reference_rotation.transpose(-1, -2), points_relative_world
    )


def local_camera_rays(
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return unit camera-frame rays [B,V,H,W,3] as right, up, forward."""
    dtype, device = fov_x.dtype, fov_x.device
    x = (torch.arange(width, dtype=dtype, device=device) + 0.5) * (2.0 / width) - 1.0
    y = 1.0 - (torch.arange(height, dtype=dtype, device=device) + 0.5) * (2.0 / height)
    x = x[None, None, None, :] * torch.tan(fov_x[..., None, None] * 0.5)
    y = y[None, None, :, None] * torch.tan(fov_y[..., None, None] * 0.5)
    x = x.expand(-1, -1, height, -1)
    y = y.expand(-1, -1, -1, width)
    rays = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    return F.normalize(rays, dim=-1, eps=1e-6)


def unproject_depth_to_reference(
    depth: torch.Tensor,
    relative_translation: torch.Tensor,
    relative_rotation: torch.Tensor,
    local_rays: torch.Tensor,
) -> torch.Tensor:
    """Unproject normalized ray depth into normalized first-camera point maps."""
    rotated_rays = torch.einsum("bvij,bvhwj->bvhwi", relative_rotation, local_rays)
    points = relative_translation[:, :, None, None] + depth[..., None] * rotated_rays
    return points.movedim(-1, 2)
