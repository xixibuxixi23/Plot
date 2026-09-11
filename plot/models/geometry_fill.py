from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from plot.models.geometry_bootstrap import GeometryBootstrap, GeometryBootstrapArgs
from plot.camera_geometry import camera_to_world_rotation


def decode_voxel_prediction(
    output: dict[str, torch.Tensor], air_class: int, occupancy_threshold: float = 0.5
) -> torch.Tensor:
    """Decode occupancy first, then material, for explicit-occupancy models."""
    logits = output["voxel_logits"]
    occupancy_logits = output.get("occupancy_logits")
    if occupancy_logits is None:
        return logits.argmax(1)
    non_air_logits = logits.clone()
    non_air_logits[:, air_class] = torch.finfo(non_air_logits.dtype).min
    material = non_air_logits.argmax(1)
    occupied = occupancy_logits[:, 0].sigmoid() >= occupancy_threshold
    return torch.where(occupied, material, torch.full_like(material, air_class))


def trilinear_splat(
    features: torch.Tensor,
    positions: torch.Tensor,
    weights: torch.Tensor,
    spatial_shape: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence-weighted trilinear point splat into an axis-aligned 3D grid.

    ``positions`` are continuous voxel-index coordinates in ``(x, y, z)`` order.
    The returned feature volume is normalized by accumulated confidence, while
    the second result retains that confidence as an explicit evidence channel.
    """
    if features.ndim != 3 or positions.shape != features.shape[:2] + (3,):
        raise ValueError("expected features [B,N,C] and positions [B,N,3]")
    if weights.shape != features.shape[:2]:
        raise ValueError("weights must have shape [B,N]")
    batch, points, channels = features.shape
    size_x, size_y, size_z = spatial_shape
    # BF16 atomics are both less stable and incompatible with FP32 coordinate
    # weights. Accumulate the sparse evidence in FP32 under mixed precision.
    features_float = features.float()
    positions_float = positions.float()
    weights_float = weights.float()
    base = torch.floor(positions_float).long()
    fraction = positions_float - base.to(positions_float.dtype)
    output = features_float.new_zeros(batch, channels, size_x * size_y * size_z)
    support = features_float.new_zeros(batch, 1, size_x * size_y * size_z)
    for offset_x in (0, 1):
        for offset_y in (0, 1):
            for offset_z in (0, 1):
                offset = torch.tensor(
                    (offset_x, offset_y, offset_z), device=positions.device,
                    dtype=torch.long,
                )
                index = base + offset
                valid = (
                    (index[..., 0] >= 0) & (index[..., 0] < size_x)
                    & (index[..., 1] >= 0) & (index[..., 1] < size_y)
                    & (index[..., 2] >= 0) & (index[..., 2] < size_z)
                )
                corner = torch.where(offset.bool(), fraction, 1.0 - fraction).prod(-1)
                corner = corner * weights_float * valid.to(weights_float.dtype)
                clamped = torch.stack((
                    index[..., 0].clamp(0, size_x - 1),
                    index[..., 1].clamp(0, size_y - 1),
                    index[..., 2].clamp(0, size_z - 1),
                ), dim=-1)
                linear = (clamped[..., 0] * size_y + clamped[..., 1]) * size_z
                linear = linear + clamped[..., 2]
                output.scatter_add_(
                    2, linear[:, None].expand(batch, channels, points),
                    (features_float * corner[..., None]).transpose(1, 2),
                )
                support.scatter_add_(2, linear[:, None], corner[:, None])
    output = output / support.clamp_min(1e-6)
    return (
        output.reshape(batch, channels, *spatial_shape),
        support.reshape(batch, 1, *spatial_shape),
    )


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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass
class GeometryConditionedFillArgs:
    num_block_classes: int
    air_class: int = 0
    voxel_size: int = 48
    max_distance: float = 32.0
    voxel_embedding_dim: int = 24
    splat_channels: int = 24
    base_channels: int = 24
    geometry_feature_dim: int = 192
    geometry_attention_heads: int = 6
    geometry_stages: int = 3
    geometry_patch_height: int = 12
    geometry_patch_width: int = 20
    geometry_output_height: int = 24
    geometry_output_width: int = 40
    max_views: int = 4
    full_resolution_surface: bool = False
    explicit_occupancy: bool = False
    visibility_evidence: bool = False
    free_space_samples: int = 8
    adaptive_visibility_fusion: bool = False
    pixel_semantic_head: bool = False
    pixel_semantic_splat: bool = False
    geometry_multiscale_highres: bool = False
    highres_pixel_semantic_head: bool = False
    highres_pixel_semantic_splat: bool = False
    direct_highres_visible_head: bool = False
    highres_point_refinement: bool = False

    @property
    def name(self) -> str:
        return "GeometryConditionedFillNetwork"

    def build(self) -> "GeometryConditionedFillNetwork":
        return GeometryConditionedFillNetwork(**asdict(self))


class GeometryConditionedFillNetwork(nn.Module):
    """M1 decoder grounded by the pretrained image-to-geometry front-end."""

    def __init__(
        self,
        num_block_classes: int,
        air_class: int = 0,
        voxel_size: int = 48,
        max_distance: float = 32.0,
        voxel_embedding_dim: int = 24,
        splat_channels: int = 24,
        base_channels: int = 24,
        geometry_feature_dim: int = 192,
        geometry_attention_heads: int = 6,
        geometry_stages: int = 3,
        geometry_patch_height: int = 12,
        geometry_patch_width: int = 20,
        geometry_output_height: int = 24,
        geometry_output_width: int = 40,
        max_views: int = 4,
        full_resolution_surface: bool = False,
        explicit_occupancy: bool = False,
        visibility_evidence: bool = False,
        free_space_samples: int = 8,
        adaptive_visibility_fusion: bool = False,
        pixel_semantic_head: bool = False,
        pixel_semantic_splat: bool = False,
        geometry_multiscale_highres: bool = False,
        highres_pixel_semantic_head: bool = False,
        highres_pixel_semantic_splat: bool = False,
        direct_highres_visible_head: bool = False,
        highres_point_refinement: bool = False,
    ):
        super().__init__()
        if voxel_size % 2:
            raise ValueError("voxel_size must be even")
        self.num_block_classes = int(num_block_classes)
        self.air_class = int(air_class)
        if not 0 <= self.air_class < self.num_block_classes:
            raise ValueError("air_class is outside the classifier vocabulary")
        self.unknown_index = self.num_block_classes
        self.voxel_size = int(voxel_size)
        self.max_distance = float(max_distance)
        self.full_resolution_surface = bool(full_resolution_surface)
        self.explicit_occupancy = bool(explicit_occupancy)
        self.visibility_evidence = bool(visibility_evidence)
        self.adaptive_visibility_fusion = bool(adaptive_visibility_fusion)
        self.pixel_semantic_head = bool(pixel_semantic_head)
        self.pixel_semantic_splat = bool(pixel_semantic_splat)
        self.geometry_multiscale_highres = bool(geometry_multiscale_highres)
        self.highres_pixel_semantic_head = bool(highres_pixel_semantic_head)
        self.highres_pixel_semantic_splat = bool(highres_pixel_semantic_splat)
        self.direct_highres_visible_head = bool(direct_highres_visible_head)
        self.highres_point_refinement = bool(highres_point_refinement)
        if self.highres_pixel_semantic_head and not self.geometry_multiscale_highres:
            raise ValueError("high-resolution semantics require multiscale geometry")
        if self.highres_pixel_semantic_splat and not self.highres_pixel_semantic_head:
            raise ValueError("high-resolution semantic splat requires its semantic head")
        if self.direct_highres_visible_head and not self.highres_pixel_semantic_head:
            raise ValueError("direct visible head requires high-resolution semantics")
        if self.pixel_semantic_splat and not self.pixel_semantic_head:
            raise ValueError("pixel semantic splat requires the semantic head")
        if self.adaptive_visibility_fusion and not self.visibility_evidence:
            raise ValueError("adaptive visibility fusion requires visibility evidence")
        self.free_space_samples = int(free_space_samples)
        if self.visibility_evidence and self.free_space_samples < 1:
            raise ValueError("free_space_samples must be positive")
        self.geometry = GeometryBootstrap(
            feature_dim=geometry_feature_dim,
            attention_heads=geometry_attention_heads,
            stages=geometry_stages,
            patch_height=geometry_patch_height,
            patch_width=geometry_patch_width,
            output_height=geometry_output_height,
            output_width=geometry_output_width,
            max_views=max_views,
            num_block_classes=(
                num_block_classes
                if self.pixel_semantic_head or self.highres_pixel_semantic_head else 0
            ),
            multiscale_highres=self.geometry_multiscale_highres,
            highres_semantic_head=self.highres_pixel_semantic_head,
            highres_point_refinement=self.highres_point_refinement,
        )
        self.splat_projection = nn.Linear(geometry_feature_dim, splat_channels)
        if self.pixel_semantic_splat:
            self.pixel_semantic_embedding = nn.Parameter(
                torch.zeros(num_block_classes, splat_channels)
            )
        if self.highres_pixel_semantic_splat:
            # Zero initialization exactly preserves the source checkpoint at
            # the start of decoder adaptation.
            self.highres_pixel_semantic_embedding = nn.Parameter(
                torch.zeros(num_block_classes, splat_channels)
            )
        self.anchor_pool = nn.Sequential(
            nn.Linear(geometry_feature_dim, geometry_feature_dim), nn.GELU(),
        )
        self.anchor_position_head = nn.Linear(geometry_feature_dim, 3)
        self.anchor_direction_head = nn.Linear(geometry_feature_dim, 3)
        nn.init.zeros_(self.anchor_position_head.weight)
        nn.init.constant_(self.anchor_position_head.bias, 0.0)
        with torch.no_grad():
            self.anchor_position_head.bias.copy_(torch.tensor((0.5 / 48, 0.5 / 48, 1.6 / 48)))

        self.block_embedding = nn.Embedding(self.num_block_classes + 1, voxel_embedding_dim)
        if self.direct_highres_visible_head:
            self.direct_visible_embedding = nn.Parameter(
                self.block_embedding.weight[:self.num_block_classes].detach().clone()
            )
            self.direct_visible_classifier = nn.Conv3d(
                voxel_embedding_dim, self.num_block_classes, 1
            )
            # The direct ray path is an exact no-op at warm start.
            nn.init.zeros_(self.direct_visible_classifier.weight)
            nn.init.zeros_(self.direct_visible_classifier.bias)
        surface_channels = 1 if self.full_resolution_surface else 0
        self.enc0 = _ConvBlock(voxel_embedding_dim + 2 + surface_channels, base_channels)
        self.enc1 = _ConvBlock(base_channels, base_channels * 2)
        fused_channels = base_channels * 2 + splat_channels + 1
        self.fuse = _ConvBlock(fused_channels, base_channels * 2)
        self.middle = _ConvBlock(base_channels * 2, base_channels * 4)
        self.dec1 = _ConvBlock(base_channels * 6, base_channels * 2)
        self.dec0 = _ConvBlock(base_channels * 3, base_channels)
        self.classifier = nn.Conv3d(base_channels, self.num_block_classes, 1)
        if self.explicit_occupancy:
            # A zero-initialized residual keeps old semantic checkpoints exactly
            # equivalent at warm start, then lets geometry learn occupancy
            # independently from uncertain block-material classification.
            self.occupancy_residual = nn.Conv3d(base_channels, 1, 1)
            nn.init.zeros_(self.occupancy_residual.weight)
            nn.init.zeros_(self.occupancy_residual.bias)
        if self.visibility_evidence:
            # Softplus keeps the evidence semantics fixed: surfaces can only
            # add occupancy and traversed free space can only remove it. The
            # small initialization keeps a v4 warm start nearly equivalent.
            self.surface_evidence_gain_raw = nn.Parameter(torch.tensor(-6.0))
            self.free_space_evidence_gain_raw = nn.Parameter(torch.tensor(-6.0))
        if self.adaptive_visibility_fusion:
            evidence_channels = max(8, base_channels // 2)
            self.visibility_refiner = nn.Sequential(
                nn.Conv3d(base_channels + 2, evidence_channels, 3, padding=1),
                nn.GroupNorm(min(4, evidence_channels), evidence_channels),
                nn.SiLU(),
                nn.Conv3d(evidence_channels, 1, 1),
            )
            nn.init.zeros_(self.visibility_refiner[-1].weight)
            nn.init.zeros_(self.visibility_refiner[-1].bias)

    def load_geometry_checkpoint(self, path: str | Path) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        expected = GeometryBootstrapArgs(**checkpoint["model_args"])
        if expected.feature_dim != self.geometry.feature_dim:
            raise ValueError("geometry checkpoint feature dimension does not match fill model")
        self.geometry.load_state_dict(checkpoint["model"])
        return checkpoint

    def freeze_geometry(self, frozen: bool = True) -> None:
        for parameter in self.geometry.parameters():
            parameter.requires_grad_(not frozen)

    def train_geometry_multiscale_only(self) -> None:
        """Freeze the geometry backbone except the newly added high-resolution fusion path."""
        if not self.geometry.multiscale_highres:
            raise ValueError("multiscale-only training requires geometry_multiscale_highres=True")
        self.freeze_geometry(True)
        for module_name in ("shallow_adapter", "middle_adapter", "multiscale_fusion"):
            for parameter in getattr(self.geometry, module_name).parameters():
                parameter.requires_grad_(True)

    def train_highres_semantic_only(self) -> None:
        """Train only the new 90x160 image-semantic branch.

        This deliberately freezes both the established geometry estimator and
        the 3D decoder, so pixel recognition can be measured without changing
        depth, pose, or voxel reconstruction quality.
        """
        if not self.highres_pixel_semantic_head:
            raise ValueError("semantic-only training requires the high-resolution head")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for module_name in (
            "highres_semantic_shallow", "highres_semantic_middle",
            "highres_semantic_deep", "highres_semantic_fusion",
            "highres_semantic_head",
        ):
            for parameter in getattr(self.geometry, module_name).parameters():
                parameter.requires_grad_(True)

    def train_direct_visible_only(self) -> None:
        """Train only the full-resolution per-ray voxel evidence path."""
        if not self.direct_highres_visible_head:
            raise ValueError("direct-only training requires the direct visible head")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.direct_visible_embedding.requires_grad_(True)
        for parameter in self.direct_visible_classifier.parameters():
            parameter.requires_grad_(True)

    def train_highres_point_only(self) -> None:
        """Freeze the established model and train only high-res 3D point residuals."""
        if not self.highres_point_refinement:
            raise ValueError("point-only training requires highres_point_refinement=True")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.geometry.highres_point_refiner.parameters():
            parameter.requires_grad_(True)

    def _anchor_pose(self, geometry: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = geometry["dense_features"][:, 0].mean(dim=(-2, -1))
        pooled = self.anchor_pool(pooled)
        position = self.anchor_position_head(pooled)
        direction_raw = self.anchor_direction_head(pooled)
        fallback = torch.zeros_like(direction_raw)
        fallback[..., 0] = 1.0
        direction_raw = direction_raw + fallback
        direction = F.normalize(direction_raw, dim=-1, eps=1e-6)
        return position, direction

    def _camera_poses(
        self,
        geometry: dict[str, torch.Tensor],
        anchor_position: torch.Tensor,
        anchor_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        reference_rotation = camera_to_world_rotation(anchor_direction)
        translation = geometry["relative_translation"] * (self.max_distance / self.voxel_size)
        translation = torch.einsum("bij,bvj->bvi", reference_rotation, translation)
        position = anchor_position[:, None] + translation
        rotation = reference_rotation[:, None] @ geometry["relative_rotation"]
        direction = F.normalize(rotation[..., 2], dim=-1, eps=1e-6)
        return position, direction

    def _splat_geometry(
        self,
        geometry: dict[str, torch.Tensor],
        anchor_position: torch.Tensor,
        anchor_direction: torch.Tensor,
        view_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None,
        torch.Tensor | None, torch.Tensor | None,
    ]:
        batch, views, _, height, width = geometry["point_map"].shape
        reference_rotation = camera_to_world_rotation(anchor_direction)
        points = geometry["point_map"].movedim(2, -1) * self.max_distance
        points = torch.einsum("bij,bvhwj->bvhwi", reference_rotation, points)
        full_index = points + anchor_position[:, None, None, None] * self.voxel_size
        full_index = full_index + (self.voxel_size - 1) * 0.5
        half_index = (full_index + 0.5) * 0.5 - 0.5
        dense = geometry["dense_features"].permute(0, 1, 3, 4, 2)
        dense = self.splat_projection(dense)
        if self.pixel_semantic_splat:
            semantic_probability = geometry["pixel_semantic_logits"].float().softmax(2)
            semantic_feature = torch.einsum(
                "bvchw,cs->bvhws", semantic_probability, self.pixel_semantic_embedding
            )
            dense = dense + semantic_feature.to(dense.dtype)
        features = dense.reshape(batch, -1, dense.shape[-1])
        confidence = geometry["visibility_logits"].sigmoid()
        confidence = confidence * torch.exp(-geometry["log_uncertainty"].clamp(-3.0, 3.0))
        confidence = confidence * view_mask[:, :, None, None].to(confidence.dtype)
        confidence = confidence.reshape(batch, -1)
        positions = half_index.reshape(batch, -1, 3)
        spatial = (self.voxel_size // 2,) * 3
        splat, support = trilinear_splat(features, positions, confidence, spatial)
        direct_visible_features = None
        direct_visible_support = None
        if self.highres_pixel_semantic_splat or self.direct_highres_visible_head:
            highres_logits = geometry["highres_pixel_semantic_logits"]
            highres_size = highres_logits.shape[-2:]
            point_map = geometry.get("highres_point_map")
            if point_map is None:
                point_map = F.interpolate(
                    geometry["point_map"].reshape(batch * views, 3, height, width),
                    size=highres_size, mode="bilinear", align_corners=False,
                ).reshape(batch, views, 3, *highres_size)
            highres_points = point_map.movedim(2, -1) * self.max_distance
            highres_points = torch.einsum(
                "bij,bvhwj->bvhwi", reference_rotation, highres_points
            )
            highres_full_index = (
                highres_points + anchor_position[:, None, None, None] * self.voxel_size
                + (self.voxel_size - 1) * 0.5
            )
            highres_half_index = (highres_full_index + 0.5) * 0.5 - 0.5
            semantic_probability = highres_logits.float().softmax(2)
            highres_confidence = F.interpolate(
                confidence.reshape(batch * views, 1, height, width),
                size=highres_size, mode="bilinear", align_corners=False,
            ).reshape(batch, views, *highres_size)
            if self.highres_pixel_semantic_splat:
                semantic_features = torch.einsum(
                    "bvchw,cs->bvhws", semantic_probability,
                    self.highres_pixel_semantic_embedding,
                )
                semantic_splat, _ = trilinear_splat(
                    semantic_features.reshape(batch, -1, semantic_features.shape[-1]),
                    highres_half_index.reshape(batch, -1, 3),
                    highres_confidence.reshape(batch, -1), spatial,
                )
                splat = splat + semantic_splat.to(splat.dtype)
            if self.direct_highres_visible_head:
                direct_features = torch.einsum(
                    "bvchw,ce->bvhwe", semantic_probability,
                    self.direct_visible_embedding,
                )
                direct_visible_features, direct_visible_support = trilinear_splat(
                    direct_features.reshape(batch, -1, direct_features.shape[-1]),
                    highres_full_index.reshape(batch, -1, 3),
                    highres_confidence.reshape(batch, -1),
                    (self.voxel_size,) * 3,
                )
        surface_support = None
        free_space_support = None
        if self.full_resolution_surface or self.visibility_evidence:
            full_positions = full_index.reshape(batch, -1, 3)
            unit_features = torch.ones(
                batch, full_positions.shape[1], 1, device=features.device, dtype=features.dtype
            )
            _, surface_support = trilinear_splat(
                unit_features, full_positions, confidence, (self.voxel_size,) * 3
            )
        if self.visibility_evidence:
            camera_position, _ = self._camera_poses(
                geometry, anchor_position, anchor_direction
            )
            camera_index = camera_position * self.voxel_size + (self.voxel_size - 1) * 0.5
            fractions = torch.linspace(
                0.05, 0.9, self.free_space_samples,
                device=full_index.device, dtype=full_index.dtype,
            )
            free_positions = camera_index[:, :, None, None, None]
            free_positions = free_positions + fractions[None, None, :, None, None, None] * (
                full_index[:, :, None] - free_positions
            )
            free_positions = free_positions.reshape(batch, -1, 3)
            free_confidence = confidence.reshape(batch, views, height, width)
            free_confidence = free_confidence[:, :, None].expand(
                -1, -1, self.free_space_samples, -1, -1
            ).reshape(batch, -1)
            free_features = torch.ones(
                batch, free_positions.shape[1], 1,
                device=features.device, dtype=features.dtype,
            )
            _, free_space_support = trilinear_splat(
                free_features, free_positions, free_confidence,
                (self.voxel_size,) * 3,
            )
        return (
            splat, support, surface_support, free_space_support,
            direct_visible_features, direct_visible_support,
        )

    def forward(
        self,
        voxel_context: torch.Tensor,
        known_mask: torch.Tensor,
        fill_mask: torch.Tensor,
        images: torch.Tensor,
        agent_mask: torch.Tensor,
        *,
        splat_camera_position: torch.Tensor | None = None,
        splat_camera_direction: torch.Tensor | None = None,
        splat_teacher_weight: float = 1.0,
        return_aux: bool = False,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        geometry = self.geometry(images, agent_mask)
        anchor_position, anchor_direction = self._anchor_pose(geometry)
        # Pose and geometry have their own metric supervision. Letting voxel CE
        # optimize the splat coordinates creates a degenerate shortcut in which
        # the camera drifts to make a particular tile easier to classify.
        splat_position = anchor_position.detach()
        splat_direction = anchor_direction.detach()
        teacher_weight = float(splat_teacher_weight)
        if not 0.0 <= teacher_weight <= 1.0:
            raise ValueError("splat_teacher_weight must be between zero and one")
        if splat_camera_position is not None:
            teacher_position = splat_camera_position[:, 0]
            splat_position = teacher_weight * teacher_position + (1.0 - teacher_weight) * splat_position
        if splat_camera_direction is not None:
            teacher_direction = splat_camera_direction[:, 0]
            splat_direction = F.normalize(
                teacher_weight * teacher_direction + (1.0 - teacher_weight) * splat_direction,
                dim=-1, eps=1e-6,
            )
        (
            splat, support, surface_support, free_space_support,
            direct_visible_features, direct_visible_support,
        ) = self._splat_geometry(
            geometry, splat_position, splat_direction, agent_mask
        )

        indices = torch.where(
            known_mask, voxel_context.clamp(0, self.num_block_classes - 1),
            torch.full_like(voxel_context, self.unknown_index),
        )
        embedding = self.block_embedding(indices).movedim(-1, 1)
        masks = torch.stack((known_mask, fill_mask), dim=1).to(embedding.dtype)
        if surface_support is not None:
            masks = torch.cat((masks, surface_support.clamp_max(1.0)), dim=1)
        x0 = self.enc0(torch.cat((embedding, masks), dim=1))
        x1 = self.enc1(F.avg_pool3d(x0, 2))
        fused = self.fuse(torch.cat((x1, splat, support.clamp_max(1.0)), dim=1))
        middle = self.middle(F.avg_pool3d(fused, 2))
        up1 = F.interpolate(middle, size=fused.shape[-3:], mode="trilinear", align_corners=False)
        up1 = self.dec1(torch.cat((up1, fused), dim=1))
        up0 = F.interpolate(up1, size=x0.shape[-3:], mode="trilinear", align_corners=False)
        decoded = self.dec0(torch.cat((up0, x0), dim=1))
        semantic_logits = self.classifier(decoded)
        if self.direct_highres_visible_head:
            direct_residual = self.direct_visible_classifier(
                direct_visible_features.to(semantic_logits.dtype)
            )
            semantic_logits = semantic_logits + (
                direct_visible_support.clamp(0.0, 1.0).to(direct_residual.dtype)
                * direct_residual
            )
        occupancy_logits = None
        completion_occupancy_logits = None
        visibility_residual = None
        visibility_confidence = None
        logits = semantic_logits
        if self.explicit_occupancy:
            air = semantic_logits[:, self.air_class : self.air_class + 1]
            non_air_indices = [
                index for index in range(self.num_block_classes) if index != self.air_class
            ]
            non_air = semantic_logits[:, non_air_indices]
            base_occupancy_logits = torch.logsumexp(non_air, dim=1, keepdim=True) - air
            occupancy_logits = base_occupancy_logits + self.occupancy_residual(decoded)
            completion_occupancy_logits = occupancy_logits
            if self.visibility_evidence:
                surface_evidence = surface_support.float().clamp(0.0, 1.0)
                free_evidence = free_space_support.float().clamp(0.0, 1.0)
                free_evidence = free_evidence * (1.0 - surface_evidence)
                surface_gain = F.softplus(self.surface_evidence_gain_raw)
                free_space_gain = F.softplus(self.free_space_evidence_gain_raw)
                occupancy_logits = occupancy_logits + (
                    surface_gain * surface_evidence - free_space_gain * free_evidence
                )
                if self.adaptive_visibility_fusion:
                    visibility_confidence = torch.maximum(
                        surface_evidence, free_evidence
                    ).clamp(0.0, 1.0)
                    visibility_residual = self.visibility_refiner(
                        torch.cat((decoded, surface_evidence, free_evidence), dim=1)
                    )
                    occupancy_logits = occupancy_logits + (
                        visibility_confidence * visibility_residual
                    )
            log_air = F.logsigmoid(-occupancy_logits)
            log_non_air = F.logsigmoid(occupancy_logits) + F.log_softmax(non_air, dim=1)
            logits = semantic_logits.new_empty(semantic_logits.shape)
            logits[:, self.air_class : self.air_class + 1] = log_air.to(logits.dtype)
            logits[:, non_air_indices] = log_non_air.to(logits.dtype)
        camera_position, camera_direction = self._camera_poses(
            geometry, anchor_position, anchor_direction
        )
        output = {
            "voxel_logits": logits,
            "occupancy_logits": occupancy_logits,
            "completion_occupancy_logits": completion_occupancy_logits,
            "visibility_residual": visibility_residual,
            "visibility_confidence": visibility_confidence,
            "camera_position": camera_position,
            "camera_direction": camera_direction,
            "anchor_position": anchor_position,
            "anchor_direction": anchor_direction,
            "splat_features": splat,
            "splat_support": support,
            "surface_support": surface_support,
            "free_space_support": free_space_support,
            "direct_visible_support": direct_visible_support,
            "surface_evidence_gain": (
                F.softplus(self.surface_evidence_gain_raw)
                if self.visibility_evidence else None
            ),
            "free_space_evidence_gain": (
                F.softplus(self.free_space_evidence_gain_raw)
                if self.visibility_evidence else None
            ),
            "geometry": geometry,
        }
        return output if return_aux else logits
