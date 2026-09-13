"""Domain-specific identity features for Minecraft resident appearances."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def crop_masked_players(images, masks, *, output_size=(128, 64), padding=0.12):
    """Crop masked players into fixed RGBA tensors while preserving aspect ratio.

    Bounding boxes are supervision metadata, so their discrete computation does
    not need gradients.  Resizing and padding remain differentiable with respect
    to ``images`` for the generated-frame identity loss.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [B,3,H,W]")
    if masks.shape != (len(images), 1, *images.shape[-2:]):
        raise ValueError("masks must be [B,1,H,W] and align with images")
    out_h, out_w = map(int, output_size)
    if out_h < 1 or out_w < 1 or padding < 0:
        raise ValueError("invalid crop output size or padding")
    crops, valid = [], []
    for image, mask in zip(images, masks):
        foreground = mask[0] > 0.5
        points = foreground.nonzero(as_tuple=False)
        if not len(points):
            crops.append(image.new_zeros(4, out_h, out_w))
            valid.append(False)
            continue
        y0, x0 = points.amin(0).tolist()
        y1, x1 = (points.amax(0) + 1).tolist()
        pad_y = max(1, round((y1 - y0) * padding))
        pad_x = max(1, round((x1 - x0) * padding))
        y0, y1 = max(0, y0 - pad_y), min(image.shape[-2], y1 + pad_y)
        x0, x1 = max(0, x0 - pad_x), min(image.shape[-1], x1 + pad_x)
        alpha = mask[:, y0:y1, x0:x1].to(image.dtype).clamp(0, 1)
        rgba = torch.cat((image[:, y0:y1, x0:x1] * alpha, alpha), dim=0)
        scale = min(out_h / rgba.shape[-2], out_w / rgba.shape[-1])
        resized_h = max(1, min(out_h, round(rgba.shape[-2] * scale)))
        resized_w = max(1, min(out_w, round(rgba.shape[-1] * scale)))
        resized = F.interpolate(
            rgba[None], size=(resized_h, resized_w), mode="bilinear", align_corners=False
        )[0]
        top = (out_h - resized_h) // 2
        bottom = out_h - resized_h - top
        left = (out_w - resized_w) // 2
        right = out_w - resized_w - left
        crops.append(F.pad(resized, (left, right, top, bottom)))
        valid.append(True)
    return torch.stack(crops), torch.tensor(valid, device=images.device)


class _IdentityTower(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1), nn.GroupNorm(4, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GroupNorm(8, 128), nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.GroupNorm(16, 256), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(), nn.Linear(512, 256), nn.SiLU(), nn.Linear(256, embedding_dim)
        )

    def forward(self, value):
        return F.normalize(self.projection(self.features(value)), dim=-1, eps=1e-6)


class PlayerIdentityEncoder(nn.Module):
    """Map canonical four-view skins and rendered player crops to one space."""

    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.crop_size = (128, 64)
        self.reference_tower = _IdentityTower(self.embedding_dim)
        self.crop_tower = _IdentityTower(self.embedding_dim)
        self.view_embedding = nn.Parameter(torch.randn(4, self.embedding_dim) * 0.02)

    def encode_reference(self, reference):
        if reference.ndim != 5 or reference.shape[1:3] != (4, 4):
            raise ValueError("reference must be [B,4,4,H,W]")
        batch = len(reference)
        per_view = self.reference_tower(reference.flatten(0, 1)).unflatten(0, (batch, 4))
        per_view = per_view + self.view_embedding[None].to(per_view.dtype)
        return F.normalize(per_view.mean(1), dim=-1, eps=1e-6)

    def encode_crop(self, crop):
        if crop.ndim != 4 or crop.shape[1] != 4:
            raise ValueError("crop must be [B,4,H,W]")
        return self.crop_tower(crop)

    def forward(self, crop, reference):
        return self.encode_crop(crop), self.encode_reference(reference)


def multi_positive_contrastive_loss(query, key, identity, valid, *, temperature=0.07):
    """Symmetric InfoNCE, treating duplicate skin hashes as extra positives."""
    if query.shape != key.shape or query.ndim != 2:
        raise ValueError("query and key must be matching [B,D] tensors")
    if identity.shape != (len(query),) or valid.shape != (len(query),):
        raise ValueError("identity and valid must be [B]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    keep = valid.bool()
    if keep.sum() < 2:
        return query.sum() * 0
    query, key, identity = query[keep], key[keep], identity[keep]

    def direction(left, right):
        logits = left @ right.transpose(0, 1) / temperature
        positives = identity[:, None] == identity[None, :]
        numerator = torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
        denominator = torch.logsumexp(logits, dim=1)
        return (denominator - numerator).mean()

    return (direction(query, key) + direction(key, query)) / 2


@dataclass(frozen=True)
class PlayerIdentityCheckpoint:
    embedding_dim: int = 128
    crop_size: tuple[int, int] = (128, 64)
