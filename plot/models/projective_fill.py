"""M1 with fixed voxel queries and direct image sampling, without point splatting.

Camera positions use the dataset's cube-centred, size-normalized convention.
Poses/FOV are explicit inputs: callers must declare whether poses are supplied
metadata or predictions. This module never reads reconstruction targets.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


def project_voxels(position, direction, fov_x, fov_y, size):
    """Return image grid [B,V,S**3,2], axial depth and geometric validity."""
    forward = F.normalize(direction.float(), dim=-1)
    up = torch.zeros_like(forward)
    up[..., 2] = 1
    right = torch.linalg.cross(forward, up, dim=-1)
    alternative = torch.zeros_like(up)
    alternative[..., 1] = 1
    right = torch.where(
        right.square().sum(-1, keepdim=True) < 1e-8,
        torch.linalg.cross(forward, alternative, dim=-1), right,
    )
    right = F.normalize(right, dim=-1)
    up = F.normalize(torch.linalg.cross(right, forward, dim=-1), dim=-1)
    axis = (torch.arange(size, device=position.device).float() - (size - 1) / 2) / size
    xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    delta = xyz[None, None] - position.float()[:, :, None]
    depth = (delta * forward[:, :, None]).sum(-1)
    x = (delta * right[:, :, None]).sum(-1) / (depth.clamp_min(1e-6) * (fov_x / 2).tan()[..., None])
    y = -(delta * up[:, :, None]).sum(-1) / (depth.clamp_min(1e-6) * (fov_y / 2).tan()[..., None])
    grid = torch.stack((x, y), -1)
    valid = (depth > 0) & (grid.abs() <= 1).all(-1)
    # Invalid projections are out of frame, avoiding extreme sampling coordinates.
    grid = torch.where(valid[..., None], grid, torch.full_like(grid, 2))
    return grid, depth, valid


def block(cin, cout, dim):
    conv = nn.Conv2d if dim == 2 else nn.Conv3d
    return nn.Sequential(
        conv(cin, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
        conv(cout, cout, 3, padding=1), nn.GroupNorm(4, cout), nn.GELU(),
    )


@dataclass
class ProjectiveFillArgs:
    num_block_classes: int
    voxel_size: int = 48
    image_channels: int = 32
    channels: int = 24

    def build(self):
        return ProjectiveFillNetwork(**asdict(self))


class ProjectiveFillNetwork(nn.Module):
    """Trainable image pyramid -> fixed queries -> one spatial U-Net -> classes."""

    def __init__(self, num_block_classes, voxel_size=48, image_channels=32, channels=24):
        super().__init__()
        if voxel_size % 4 or image_channels % 4 or channels % 4:
            raise ValueError("size and channel widths must be divisible by four")
        self.num_block_classes = num_block_classes
        self.voxel_size = voxel_size
        self.image0 = block(3, image_channels, 2)
        self.image1 = block(image_channels, image_channels * 2, 2)
        self.image2 = block(image_channels * 2, image_channels * 2, 2)
        self.image_fuse = nn.Conv2d(image_channels * 5, image_channels, 1)
        self.view_embed = nn.Sequential(
            nn.Conv3d(image_channels + 4, channels, 1), nn.GELU(),
        )
        self.view_score = nn.Conv3d(channels, 1, 1)
        self.context = nn.Embedding(num_block_classes + 1, 8)
        self.enc0 = block(channels + 8 + 2 + 3, channels, 3)
        self.enc1 = block(channels, channels * 2, 3)
        self.middle = block(channels * 2, channels * 4, 3)
        self.dec1 = block(channels * 6, channels * 2, 3)
        self.dec0 = block(channels * 3, channels, 3)
        self.classifier = nn.Conv3d(channels, num_block_classes, 1)

    def forward(self, voxel_context, known_mask, fill_mask, images, agent_mask,
                camera_position, camera_direction, fov_x, fov_y):
        b, v = images.shape[:2]
        s = self.voxel_size
        image = images.reshape(b * v, *images.shape[2:])
        shallow = self.image0(F.avg_pool2d(image, 2))
        middle = self.image1(F.avg_pool2d(shallow, 2))
        deep = self.image2(F.avg_pool2d(middle, 2))
        feature = self.image_fuse(torch.cat((shallow,
            F.interpolate(middle, shallow.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(deep, shallow.shape[-2:], mode="bilinear", align_corners=False)), 1))
        grid, depth, valid = project_voxels(camera_position, camera_direction, fov_x, fov_y, s)
        valid = valid & agent_mask[..., None].bool()
        sampled = F.grid_sample(feature.float(), grid.reshape(b * v, s * s, s, 2),
                                align_corners=False).reshape(b * v, -1, s, s, s)
        geometry = torch.cat((grid, depth[..., None], valid[..., None].float()), -1)
        geometry = geometry.reshape(b * v, s, s, s, 4).movedim(-1, 1)
        state = self.view_embed(torch.cat((sampled, geometry), 1))
        state = state.reshape(b, v, -1, s, s, s)
        valid = valid.reshape(b, v, 1, s, s, s)
        scores = self.view_score(state.flatten(0, 1)).reshape(b, v, 1, s, s, s).float()
        weights = scores.masked_fill(~valid, -1e4).softmax(1) * valid
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        fused = (state * weights).sum(1)
        indices = torch.where(known_mask, voxel_context, self.num_block_classes)
        context = self.context(indices).movedim(-1, 1)
        axis = torch.linspace(-1, 1, s, device=images.device)
        xyz = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), 0)
        x0 = self.enc0(torch.cat((fused, context, known_mask[:, None], fill_mask[:, None],
                                 xyz[None].expand(b, -1, -1, -1, -1)), 1))
        x1 = self.enc1(F.avg_pool3d(x0, 2))
        x2 = self.middle(F.avg_pool3d(x1, 2))
        x1 = self.dec1(torch.cat((x1, F.interpolate(x2, x1.shape[-3:], mode="trilinear", align_corners=False)), 1))
        x0 = self.dec0(torch.cat((x0, F.interpolate(x1, x0.shape[-3:], mode="trilinear", align_corners=False)), 1))
        return self.classifier(x0)

    @torch.no_grad()
    def predict(self, **inputs):
        predicted = self(**inputs).argmax(1)
        writable = inputs["fill_mask"].bool() & ~inputs["known_mask"].bool()
        return torch.where(writable, predicted, inputs["voxel_context"])
