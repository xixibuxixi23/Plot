from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from plot.models.projection import raycast_voxel_semantic_targets, raycast_voxel_targets
from plot.camera_geometry import (
    canonicalize_cameras,
    local_camera_rays,
    rotation_6d_to_matrix,
    unproject_depth_to_reference,
)


@dataclass
class GeometryBootstrapArgs:
    feature_dim: int = 192
    attention_heads: int = 6
    stages: int = 3
    patch_height: int = 12
    patch_width: int = 20
    output_height: int = 24
    output_width: int = 40
    max_views: int = 4
    num_block_classes: int = 0
    multiscale_highres: bool = False
    highres_semantic_head: bool = False
    highres_point_refinement: bool = False

    @property
    def name(self) -> str:
        return "GeometryBootstrap"

    def build(self) -> "GeometryBootstrap":
        return GeometryBootstrap(**asdict(self))


class GeometryBootstrap(nn.Module):
    """VGGT-style lightweight geometry front-end in the first-view gauge."""

    def __init__(
        self,
        feature_dim: int = 192,
        attention_heads: int = 6,
        stages: int = 3,
        patch_height: int = 12,
        patch_width: int = 20,
        output_height: int = 24,
        output_width: int = 40,
        max_views: int = 4,
        num_block_classes: int = 0,
        multiscale_highres: bool = False,
        highres_semantic_head: bool = False,
        highres_point_refinement: bool = False,
    ):
        super().__init__()
        if feature_dim % attention_heads:
            raise ValueError("feature_dim must be divisible by attention_heads")
        self.feature_dim = int(feature_dim)
        self.patch_height = int(patch_height)
        self.patch_width = int(patch_width)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.max_views = int(max_views)
        self.num_block_classes = int(num_block_classes)
        self.multiscale_highres = bool(multiscale_highres)
        self.highres_semantic_enabled = bool(highres_semantic_head)
        self.highres_point_refinement = bool(highres_point_refinement)
        if self.highres_semantic_enabled and not self.multiscale_highres:
            raise ValueError("high-resolution semantics require multiscale_highres=True")
        if self.highres_semantic_enabled and self.num_block_classes <= 0:
            raise ValueError("high-resolution semantics require block classes")
        if self.highres_point_refinement and not self.highres_semantic_enabled:
            raise ValueError("high-resolution point refinement requires semantic features")
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 48, 7, stride=2, padding=3),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.Conv2d(96, feature_dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((patch_height, patch_width)),
        )
        patches = patch_height * patch_width
        self.patch_position = nn.Parameter(torch.randn(patches, feature_dim) * 0.02)
        self.view_embedding = nn.Parameter(torch.randn(max_views, feature_dim) * 0.02)
        self.reference_embedding = nn.Parameter(torch.randn(2, feature_dim) * 0.02)
        self.camera_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.frame_blocks = nn.ModuleList()
        self.global_blocks = nn.ModuleList()
        for _ in range(stages):
            self.frame_blocks.append(
                nn.TransformerEncoderLayer(
                    feature_dim, attention_heads, feature_dim * 4, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True,
                )
            )
            self.global_blocks.append(
                nn.TransformerEncoderLayer(
                    feature_dim, attention_heads, feature_dim * 4, dropout=0.0,
                    activation="gelu", batch_first=True, norm_first=True,
                )
            )
        self.output_norm = nn.LayerNorm(feature_dim)
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, 9)
        )
        dense_hidden = max(64, feature_dim // 2)
        self.dense_head = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.GroupNorm(8, feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, dense_hidden, 3, padding=1),
            nn.GroupNorm(8, dense_hidden),
            nn.GELU(),
            nn.Conv2d(dense_hidden, 6, 1),
        )
        self.semantic_head = (
            nn.Conv2d(feature_dim, self.num_block_classes, 1)
            if self.num_block_classes > 0 else None
        )
        if self.multiscale_highres:
            self.shallow_adapter = nn.Sequential(
                nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU()
            )
            self.middle_adapter = nn.Sequential(
                nn.Conv2d(96, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU()
            )
            self.multiscale_fusion = nn.Sequential(
                nn.Conv2d(feature_dim + 96, feature_dim, 1),
                nn.GroupNorm(8, feature_dim), nn.GELU(),
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            )
            # Preserve the exact warm-start function until the new branches learn.
            nn.init.zeros_(self.multiscale_fusion[-1].weight)
            nn.init.zeros_(self.multiscale_fusion[-1].bias)
        if self.highres_semantic_enabled:
            semantic_width = 64
            self.highres_semantic_shallow = nn.Sequential(
                nn.Conv2d(48, semantic_width, 3, padding=1),
                nn.GroupNorm(8, semantic_width), nn.GELU(),
            )
            self.highres_semantic_middle = nn.Sequential(
                nn.Conv2d(96, semantic_width, 3, padding=1),
                nn.GroupNorm(8, semantic_width), nn.GELU(),
            )
            self.highres_semantic_deep = nn.Sequential(
                nn.Conv2d(feature_dim, semantic_width, 1),
                nn.GroupNorm(8, semantic_width), nn.GELU(),
            )
            self.highres_semantic_fusion = nn.Sequential(
                nn.Conv2d(semantic_width * 3, semantic_width * 2, 3, padding=1),
                nn.GroupNorm(8, semantic_width * 2), nn.GELU(),
                nn.Conv2d(semantic_width * 2, semantic_width, 3, padding=1),
                nn.GroupNorm(8, semantic_width), nn.GELU(),
            )
            self.highres_semantic_head = nn.Conv2d(
                semantic_width, self.num_block_classes, 1
            )
            if self.highres_point_refinement:
                self.highres_point_refiner = nn.Sequential(
                    nn.Conv2d(semantic_width, semantic_width, 3, padding=1),
                    nn.GroupNorm(8, semantic_width), nn.GELU(),
                    nn.Conv2d(semantic_width, 3, 1),
                )
                nn.init.zeros_(self.highres_point_refiner[-1].weight)
                nn.init.zeros_(self.highres_point_refiner[-1].bias)

    def forward(self, images: torch.Tensor, view_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must have shape [B,V,3,H,W]")
        batch, views = images.shape[:2]
        if views > self.max_views:
            raise ValueError(f"received {views} views, max_views={self.max_views}")
        if not torch.all(view_mask[:, 0].bool()):
            raise ValueError("view zero must be valid because it defines the reconstruction gauge")
        flat_images = images.reshape(batch * views, *images.shape[2:])
        if self.multiscale_highres:
            shallow = self.image_encoder[:3](flat_images)
            middle = self.image_encoder[3:6](shallow)
            feature = self.image_encoder[6:](middle)
        else:
            feature = self.image_encoder(flat_images)
        patches = feature.flatten(2).transpose(1, 2).reshape(batch, views, -1, self.feature_dim)
        reference_ids = torch.zeros(views, dtype=torch.long, device=images.device)
        reference_ids[0] = 1
        patches = patches + self.patch_position[None, None]
        patches = patches + self.view_embedding[None, :views, None]
        patches = patches + self.reference_embedding[reference_ids][None, :, None]
        cameras = self.camera_token.expand(batch, views, -1, -1)
        cameras = cameras + self.view_embedding[None, :views, None]
        cameras = cameras + self.reference_embedding[reference_ids][None, :, None]
        tokens = torch.cat((cameras, patches), dim=2)
        tokens_per_view = tokens.shape[2]
        global_padding = (~view_mask.bool()).unsqueeze(-1).expand(-1, -1, tokens_per_view)
        global_padding = global_padding.reshape(batch, views * tokens_per_view)
        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            tokens = frame_block(tokens.reshape(batch * views, tokens_per_view, self.feature_dim))
            tokens = tokens.reshape(batch, views * tokens_per_view, self.feature_dim)
            tokens = global_block(tokens, src_key_padding_mask=global_padding)
            tokens = tokens.reshape(batch, views, tokens_per_view, self.feature_dim)
        tokens = self.output_norm(tokens)
        camera_features = tokens[:, :, 0]
        patch_features = tokens[:, :, 1:].transpose(-1, -2)
        patch_features = patch_features.reshape(
            batch * views, self.feature_dim, self.patch_height, self.patch_width
        )
        patch_features = F.interpolate(
            patch_features, size=(self.output_height, self.output_width),
            mode="bilinear", align_corners=False,
        )
        if self.multiscale_highres:
            shallow_dense = F.adaptive_avg_pool2d(
                self.shallow_adapter(shallow), (self.output_height, self.output_width)
            )
            middle_dense = F.adaptive_avg_pool2d(
                self.middle_adapter(middle), (self.output_height, self.output_width)
            )
            residual = self.multiscale_fusion(
                torch.cat((patch_features, shallow_dense, middle_dense), dim=1)
            )
            patch_features = patch_features + residual
        dense = self.dense_head(patch_features).reshape(
            batch, views, 6, self.output_height, self.output_width
        )
        pose = self.pose_head(camera_features)
        translation = pose[..., :3]
        rotation = rotation_6d_to_matrix(pose[..., 3:])
        identity = torch.eye(3, device=rotation.device, dtype=rotation.dtype)
        rotation = torch.cat((identity[None, None].expand(batch, 1, -1, -1), rotation[:, 1:]), 1)
        translation = torch.cat((torch.zeros_like(translation[:, :1]), translation[:, 1:]), 1)
        valid = view_mask[:, :, None].to(translation.dtype)
        output = {
            "relative_translation": translation * valid,
            "relative_rotation": rotation,
            "depth": dense[:, :, 0].sigmoid(),
            "point_map": dense[:, :, 1:4],
            "visibility_logits": dense[:, :, 4],
            "log_uncertainty": dense[:, :, 5].clamp(-5.0, 5.0),
            "dense_features": patch_features.reshape(
                batch, views, self.feature_dim, self.output_height, self.output_width
            ),
        }
        if self.semantic_head is not None:
            output["pixel_semantic_logits"] = self.semantic_head(patch_features).reshape(
                batch, views, self.num_block_classes, self.output_height, self.output_width
            )
        if self.highres_semantic_enabled:
            highres_size = shallow.shape[-2:]
            shallow_semantic = self.highres_semantic_shallow(shallow)
            middle_semantic = F.interpolate(
                self.highres_semantic_middle(middle), size=highres_size,
                mode="bilinear", align_corners=False,
            )
            deep_semantic = F.interpolate(
                self.highres_semantic_deep(patch_features), size=highres_size,
                mode="bilinear", align_corners=False,
            )
            semantic_features = self.highres_semantic_fusion(torch.cat(
                (shallow_semantic, middle_semantic, deep_semantic), dim=1
            ))
            output["highres_pixel_semantic_logits"] = self.highres_semantic_head(
                semantic_features
            ).reshape(batch, views, self.num_block_classes, *highres_size)
            if self.highres_point_refinement:
                coarse_point_map = F.interpolate(
                    output["point_map"].reshape(
                        batch * views, 3, self.output_height, self.output_width
                    ),
                    size=highres_size, mode="bilinear", align_corners=False,
                )
                point_residual = self.highres_point_refiner(semantic_features)
                output["highres_point_map"] = (
                    coarse_point_map + point_residual
                ).reshape(batch, views, 3, *highres_size)
        return output


@torch.no_grad()
def geometry_targets_from_voxels(
    target: torch.Tensor,
    target_valid: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    camera_valid: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    air_class: int,
    height: int,
    width: int,
    samples: int,
    max_distance: float,
    supervised_height_fraction: float = 1.0,
    return_semantics: bool = False,
) -> dict[str, torch.Tensor]:
    """Raycast dense first-view-gauge supervision from an engine voxel tile."""
    size = target.shape[-1]
    occupancy = ((target != air_class) & target_valid.bool()).float()
    translation, rotation = canonicalize_cameras(
        camera_position.float(), camera_direction.float(), position_scale=float(size)
    )
    silhouettes, depths, ray_validity, semantic_classes = [], [], [], []
    local_rays = local_camera_rays(
        fov_x.float(), fov_y.float(), height, width
    )
    for view in range(camera_position.shape[1]):
        pose = (
            camera_position[:, view].float(), camera_direction[:, view].float(),
            fov_x[:, view].float(), fov_y[:, view].float(),
        )
        kwargs = dict(
            height=height, width=width, samples=samples, max_distance=max_distance
        )
        if return_semantics:
            silhouette, depth, rays, semantic_class = raycast_voxel_semantic_targets(
                target, target_valid, *pose, air_class=air_class, **kwargs
            )
            semantic_classes.append(semantic_class)
        else:
            silhouette, depth, rays = raycast_voxel_targets(occupancy, *pose, **kwargs)
        silhouettes.append(silhouette)
        depths.append(depth / max_distance)
        origin = camera_position[:, view].float() * float(size) + (size - 1) * 0.5
        endpoint = origin[:, None, None] + rays * max_distance
        endpoint_inside = ((endpoint >= -0.5) & (endpoint < size - 0.5)).all(-1)
        known_ray = silhouette.bool() | endpoint_inside
        supervised_rows = max(1, min(height, round(height * supervised_height_fraction)))
        row_mask = torch.arange(height, device=target.device) < supervised_rows
        ray_validity.append(known_ray & row_mask[None, :, None])
    normalized_depth = torch.stack(depths, dim=1)
    normalized_translation = translation / max_distance
    point_map = unproject_depth_to_reference(
        normalized_depth, normalized_translation, rotation, local_rays
    )
    point_map = point_map * torch.stack(silhouettes, dim=1)[:, :, None]
    result = {
        "relative_translation": normalized_translation,
        "relative_rotation": rotation,
        "depth": normalized_depth,
        "point_map": point_map,
        "visibility": torch.stack(silhouettes, dim=1),
        "camera_valid": camera_valid.bool(),
        "pixel_valid": torch.stack(ray_validity, dim=1) & camera_valid[:, :, None, None].bool(),
        "local_rays": local_rays,
    }
    if return_semantics:
        result["semantic_class"] = torch.stack(semantic_classes, dim=1)
    return result
