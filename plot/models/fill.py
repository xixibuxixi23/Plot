from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels), nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _valid_heads(width: int, requested: int) -> int:
    heads = min(width, requested)
    while width % heads:
        heads -= 1
    return heads


@dataclass
class FillNetworkArgs:
    num_block_classes: int
    voxel_embedding_dim: int = 24
    image_feature_dim: int = 128
    base_channels: int = 32
    attention_heads: int = 4
    max_views: int = 4

    @property
    def name(self) -> str:
        return "FillNetwork"

    def build(self) -> "FillNetwork":
        return FillNetwork(**asdict(self))


class FillNetwork(nn.Module):
    """Spatial multi-view M1 producing a voxel tile and relative cameras.

    Images remain spatial tokens and are cross-attended by the 12^3 voxel
    bottleneck. Camera position is normalized by 48 and direction is unit ENU.
    """

    def __init__(
        self, num_block_classes: int, voxel_embedding_dim: int = 24,
        image_feature_dim: int = 128, base_channels: int = 32,
        attention_heads: int = 4, max_views: int = 4,
    ):
        super().__init__()
        self.num_block_classes = int(num_block_classes)
        self.unknown_index = self.num_block_classes
        self.max_views = int(max_views)
        middle_channels = base_channels * 4
        image_heads = _valid_heads(image_feature_dim, attention_heads)
        voxel_heads = _valid_heads(middle_channels, attention_heads)

        self.block_embedding = nn.Embedding(self.num_block_classes + 1, voxel_embedding_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
            nn.Conv2d(64, image_feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(_valid_heads(image_feature_dim, 8), image_feature_dim), nn.SiLU(),
            nn.AdaptiveAvgPool2d((6, 10)),
        )
        self.view_embedding = nn.Parameter(torch.randn(max_views, image_feature_dim) * 0.02)
        self.image_position = nn.Parameter(torch.randn(60, image_feature_dim) * 0.02)
        view_layer = nn.TransformerEncoderLayer(
            image_feature_dim, image_heads, image_feature_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.view_transformer = nn.TransformerEncoder(view_layer, num_layers=2)
        self.camera_head = nn.Sequential(
            nn.LayerNorm(image_feature_dim), nn.Linear(image_feature_dim, image_feature_dim),
            nn.GELU(), nn.Linear(image_feature_dim, 6),
        )

        self.enc0 = _ConvBlock(voxel_embedding_dim + 2, base_channels)
        self.enc1 = _ConvBlock(base_channels, base_channels * 2)
        self.middle = _ConvBlock(base_channels * 2, middle_channels)
        self.image_to_voxel = nn.Linear(image_feature_dim, middle_channels)
        self.voxel_position = nn.Sequential(
            nn.Linear(3, middle_channels), nn.SiLU(), nn.Linear(middle_channels, middle_channels)
        )
        self.cross_norm = nn.LayerNorm(middle_channels)
        self.cross_attention = nn.MultiheadAttention(
            middle_channels, voxel_heads, dropout=0.0, batch_first=True
        )
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(middle_channels), nn.Linear(middle_channels, middle_channels * 4),
            nn.GELU(), nn.Linear(middle_channels * 4, middle_channels),
        )
        self.dec1 = _ConvBlock(middle_channels + base_channels * 2, base_channels * 2)
        self.dec0 = _ConvBlock(base_channels * 2 + base_channels, base_channels)
        self.classifier = nn.Conv3d(base_channels, self.num_block_classes, 1)

    def _image_features(self, images, agent_mask):
        if images.ndim != 5:
            raise ValueError("images must have shape [B,A,3,H,W]")
        batch, views = images.shape[:2]
        if views > self.max_views:
            raise ValueError(f"received {views} views, max_views={self.max_views}")
        fmap = self.image_encoder(images.reshape(batch * views, *images.shape[2:]))
        tokens = fmap.flatten(2).transpose(1, 2).reshape(batch, views, 60, -1)
        tokens = tokens + self.image_position[None, None] + self.view_embedding[None, :views, None]
        pooled = self.view_transformer(tokens.mean(2), src_key_padding_mask=~agent_mask.bool())
        return tokens.reshape(batch, views * 60, -1), pooled

    @staticmethod
    def _grid_coordinates(spatial, device, dtype):
        axes = [torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype) for n in spatial]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

    def forward(
        self, voxel_context, known_mask, fill_mask, images, agent_mask, *, return_aux=False,
    ):
        if voxel_context.ndim != 4:
            raise ValueError("voxel_context must have shape [B,X,Y,Z]")
        indices = torch.where(
            known_mask, voxel_context.clamp(0, self.num_block_classes - 1),
            torch.full_like(voxel_context, self.unknown_index),
        )
        embedding = self.block_embedding(indices).movedim(-1, 1)
        masks = torch.stack((known_mask, fill_mask), dim=1).to(embedding.dtype)
        x0 = self.enc0(torch.cat((embedding, masks), dim=1))
        x1 = self.enc1(F.avg_pool3d(x0, 2))
        middle = self.middle(F.avg_pool3d(x1, 2))

        image_tokens, view_tokens = self._image_features(images, agent_mask)
        image_tokens = self.image_to_voxel(image_tokens)
        query = middle.flatten(2).transpose(1, 2)
        coords = self._grid_coordinates(middle.shape[-3:], query.device, query.dtype)
        query = query + self.voxel_position(coords)[None]
        padding = (~agent_mask.bool()).unsqueeze(-1).expand(-1, -1, 60).reshape(images.shape[0], -1)
        attended, _ = self.cross_attention(
            self.cross_norm(query), image_tokens, image_tokens,
            key_padding_mask=padding, need_weights=False,
        )
        query = query + attended
        query = query + self.cross_ffn(query)
        middle = query.transpose(1, 2).reshape_as(middle)

        up1 = F.interpolate(middle, size=x1.shape[-3:], mode="trilinear", align_corners=False)
        up1 = self.dec1(torch.cat((up1, x1), dim=1))
        up0 = F.interpolate(up1, size=x0.shape[-3:], mode="trilinear", align_corners=False)
        logits = self.classifier(self.dec0(torch.cat((up0, x0), dim=1)))
        camera = self.camera_head(view_tokens)
        output = {
            "voxel_logits": logits,
            "camera_position": camera[..., :3],
            "camera_direction": F.normalize(camera[..., 3:], dim=-1, eps=1e-6),
        }
        return output if return_aux else logits


def masked_fill_loss(logits, target, fill_mask, target_valid=None, voxel_weight=None):
    mask = fill_mask.bool()
    if target_valid is not None:
        mask &= target_valid.bool()
    if not torch.any(mask):
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, target.long(), reduction="none")
    weights = mask.to(loss.dtype)
    if voxel_weight is not None:
        weights = weights * voxel_weight.to(loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1)


def camera_pose_loss(position, direction, target_position, target_direction, valid):
    weights = valid.to(position.dtype)
    denom = weights.sum().clamp_min(1)
    pos = F.smooth_l1_loss(position, target_position, reduction="none").mean(-1)
    angle = 1.0 - F.cosine_similarity(direction, target_direction, dim=-1)
    return (pos * weights).sum() / denom, (angle * weights).sum() / denom
