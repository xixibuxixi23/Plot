from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from plot.models.geometry_bootstrap import geometry_targets_from_voxels
from plot.models.geometry_fill import decode_voxel_prediction
from plot.models.projection import first_hit_projection_loss
from plot.models.visible_supervision import visible_voxel_masks


@dataclass
class GeometryFillTrainerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 2e-3
    max_grad_norm: float = 1.0
    air_class: int = 0
    air_weight: float = 0.25
    visible_voxel_weight: float = 2.0
    occupancy_weight: float = 0.5
    visible_only: bool = False
    ray_free_weight: float = 0.0
    visible_surface_occupancy_weight: float = 1.0
    visible_material_class_ids: tuple[int, ...] = ()
    visible_material_class_weight: float = 1.0
    projection_weight: float = 0.0
    projection_surface_weight: float = 1.0
    projection_free_space_weight: float = 0.5
    projection_silhouette_weight: float = 0.25
    projection_depth_weight: float = 0.25
    projection_edge_weight: float = 3.0
    projection_height: int = 24
    projection_width: int = 40
    projection_samples: int = 64
    projection_alpha_reference_step: float = 0.0
    projection_max_distance: float = 32.0
    evidence_learning_rate_multiplier: float = 1.0
    semantic_head_learning_rate_multiplier: float = 1.0
    geometry_multiscale_learning_rate_multiplier: float = 1.0
    camera_position_weight: float = 5.0
    camera_direction_weight: float = 1.0
    pixel_semantic_weight: float = 0.0
    pixel_semantic_class_ids: tuple[int, ...] = ()
    pixel_semantic_class_weight: float = 1.0
    highres_point_weight: float = 0.0
    teacher_splat_steps: int = 2_000
    teacher_splat_decay_steps: int = 3_000
    precision: str = "bf16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.ray_free_weight < 0 or self.visible_surface_occupancy_weight < 0:
            raise ValueError("visible occupancy weights must be nonnegative")
        if self.ray_free_weight and not self.visible_only:
            raise ValueError("ray_free_weight must be nonnegative and requires visible_only")
        if self.pixel_semantic_weight < 0 or self.pixel_semantic_class_weight < 1:
            raise ValueError("pixel semantic weights are invalid")
        if self.highres_point_weight < 0:
            raise ValueError("high-resolution point weight must be nonnegative")
        if self.visible_material_class_weight < 1:
            raise ValueError("visible material class weight must be at least one")
        if self.semantic_head_learning_rate_multiplier <= 0:
            raise ValueError("semantic head learning-rate multiplier must be positive")
        if self.geometry_multiscale_learning_rate_multiplier <= 0:
            raise ValueError("geometry multiscale learning-rate multiplier must be positive")


def _masked_weighted_mean(
    value: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor | None = None
) -> torch.Tensor:
    effective = mask.to(value.dtype)
    if weight is not None:
        effective = effective * weight.to(value.dtype)
    return (value * effective).sum() / effective.sum().clamp_min(1)


def geometry_fill_loss(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: GeometryFillTrainerConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = output["voxel_logits"].float()
    target = batch["target"].long()
    valid = batch["fill_mask"].bool() & batch["target_valid"].bool()
    voxel_weight = torch.where(
        target == config.air_class,
        torch.full_like(target, config.air_weight, dtype=logits.dtype),
        torch.ones_like(target, dtype=logits.dtype),
    )
    surface_support = output.get("surface_support")
    if surface_support is None:
        support = F.interpolate(
            output["splat_support"].float(), size=target.shape[-3:], mode="trilinear",
            align_corners=False,
        )[:, 0]
    else:
        support = surface_support.float()[:, 0]
    voxel_weight = voxel_weight * (
        1.0 + config.visible_voxel_weight * support.detach().clamp(0.0, 1.0)
    )
    voxel_error = F.cross_entropy(logits, target, reduction="none")
    voxel_loss = _masked_weighted_mean(voxel_error, valid, voxel_weight)

    occupancy_logits = output.get("occupancy_logits")
    if occupancy_logits is None:
        log_air = logits[:, config.air_class] - torch.logsumexp(logits, dim=1)
        occupancy_logits = torch.log1p(-log_air.exp().clamp(max=1.0 - 1e-6)) - log_air
    else:
        occupancy_logits = occupancy_logits.float()[:, 0]
    occupied_target = (target != config.air_class).to(logits.dtype)
    occupancy_error = F.binary_cross_entropy_with_logits(
        occupancy_logits,
        occupied_target,
        reduction="none",
    )
    occupancy_loss = _masked_weighted_mean(occupancy_error, valid)
    visible_metrics = {}
    if config.visible_only:
        if config.projection_weight != 0:
            raise ValueError("visible-only experiment requires projection_weight=0")
        surface = batch["visible_surface_mask"].bool() & valid
        free = batch["visible_free_mask"].bool() & valid
        visible_material_weight = torch.ones_like(voxel_error)
        upweighted_visible_material = torch.zeros_like(surface)
        for class_id in config.visible_material_class_ids:
            upweighted_visible_material |= target == class_id
        visible_material_weight = torch.where(
            upweighted_visible_material,
            torch.full_like(visible_material_weight, config.visible_material_class_weight),
            visible_material_weight,
        )
        voxel_loss = _masked_weighted_mean(voxel_error, surface, visible_material_weight)
        surface_loss = _masked_weighted_mean(occupancy_error, surface)
        free_loss = _masked_weighted_mean(occupancy_error, free)
        occupancy_loss = config.visible_surface_occupancy_weight * surface_loss + free_loss
        ray_free_loss = occupancy_loss.new_zeros(())
        if config.ray_free_weight:
            ray_free_loss = _masked_weighted_mean(
                occupancy_error, free, batch["visible_ray_free_weights"]
            )
            occupancy_loss = occupancy_loss + config.ray_free_weight * ray_free_loss
        decoded = decode_voxel_prediction(output, config.air_class)
        visible_metrics = {
            "visible_surface_occupancy_loss": surface_loss,
            "visible_free_occupancy_loss": free_loss,
            "ray_free_occupancy_loss": ray_free_loss,
            "visible_surface_material_accuracy": _masked_weighted_mean((decoded == target).float(), surface),
            "visible_upweighted_material_accuracy": _masked_weighted_mean(
                (decoded == target).float(), surface & upweighted_visible_material
            ),
            "visible_surface_recall": _masked_weighted_mean((decoded != config.air_class).float(), surface),
            "visible_free_accuracy": _masked_weighted_mean((decoded == config.air_class).float(), free),
        }

    camera_valid = batch["camera_valid"].bool() & batch["agent_mask"].bool()
    position_error = F.smooth_l1_loss(
        output["camera_position"].float(), batch["camera_position"].float(), reduction="none"
    ).mean(-1)
    direction_error = 1.0 - F.cosine_similarity(
        output["camera_direction"].float(), batch["camera_direction"].float(), dim=-1
    )
    camera_position_loss = _masked_weighted_mean(position_error, camera_valid)
    camera_direction_loss = _masked_weighted_mean(direction_error, camera_valid)
    total = voxel_loss + config.occupancy_weight * occupancy_loss
    pixel_semantic_loss = total.new_zeros(())
    pixel_semantic_accuracy = total.new_zeros(())
    pixel_semantic_upweighted_accuracy = total.new_zeros(())
    if config.pixel_semantic_weight > 0:
        pixel_logits = output["geometry"].get("highres_pixel_semantic_logits")
        if pixel_logits is None:
            pixel_logits = output["geometry"].get("pixel_semantic_logits")
        if pixel_logits is None:
            raise ValueError("pixel_semantic_weight requires a pixel semantic model head")
        semantic_target = geometry_targets_from_voxels(
            batch["target"], batch["target_valid"], batch["camera_position"],
            batch["camera_direction"], camera_valid, batch["fov_x"], batch["fov_y"],
            air_class=config.air_class, height=pixel_logits.shape[-2],
            width=pixel_logits.shape[-1], samples=config.projection_samples,
            max_distance=config.projection_max_distance,
            supervised_height_fraction=0.82,
            return_semantics=True,
        )
        semantic_class = semantic_target["semantic_class"]
        semantic_mask = semantic_target["pixel_valid"] & semantic_target["visibility"].bool()
        semantic_error = F.cross_entropy(
            pixel_logits.float().flatten(0, 1), semantic_class.flatten(0, 1), reduction="none"
        ).reshape_as(semantic_class)
        semantic_weights = torch.ones_like(semantic_error)
        upweighted_mask = torch.zeros_like(semantic_mask)
        for class_id in config.pixel_semantic_class_ids:
            upweighted_mask |= semantic_class == class_id
        semantic_weights = torch.where(
            upweighted_mask,
            torch.full_like(semantic_weights, config.pixel_semantic_class_weight),
            semantic_weights,
        )
        pixel_semantic_loss = _masked_weighted_mean(
            semantic_error, semantic_mask, semantic_weights
        )
        semantic_prediction = pixel_logits.argmax(2)
        pixel_semantic_accuracy = _masked_weighted_mean(
            (semantic_prediction == semantic_class).float(), semantic_mask
        )
        pixel_semantic_upweighted_accuracy = _masked_weighted_mean(
            (semantic_prediction == semantic_class).float(), semantic_mask & upweighted_mask
        )
        total = total + config.pixel_semantic_weight * pixel_semantic_loss
    highres_point_loss = total.new_zeros(())
    highres_point_mae_blocks = total.new_zeros(())
    if config.highres_point_weight > 0:
        highres_point_map = output["geometry"].get("highres_point_map")
        if highres_point_map is None:
            raise ValueError("highres_point_weight requires a high-resolution point head")
        point_target = geometry_targets_from_voxels(
            batch["target"], batch["target_valid"], batch["camera_position"],
            batch["camera_direction"], camera_valid, batch["fov_x"], batch["fov_y"],
            air_class=config.air_class, height=highres_point_map.shape[-2],
            width=highres_point_map.shape[-1], samples=config.projection_samples,
            max_distance=config.projection_max_distance,
            supervised_height_fraction=0.82,
        )
        point_mask = point_target["pixel_valid"] & point_target["visibility"].bool()
        point_abs_error = (
            highres_point_map.float() - point_target["point_map"].float()
        ).abs().mean(2)
        point_smooth_error = F.smooth_l1_loss(
            highres_point_map.float(), point_target["point_map"].float(), reduction="none"
        ).mean(2)
        highres_point_loss = _masked_weighted_mean(point_smooth_error, point_mask)
        highres_point_mae_blocks = (
            _masked_weighted_mean(point_abs_error, point_mask)
            * config.projection_max_distance
        )
        total = total + config.highres_point_weight * highres_point_loss
    projection_metrics = {
        "projection_surface_loss": total.new_zeros(()),
        "projection_free_space_loss": total.new_zeros(()),
        "projection_silhouette_loss": total.new_zeros(()),
        "projection_depth_loss": total.new_zeros(()),
    }
    if config.projection_weight > 0:
        projection_metrics = first_hit_projection_loss(
            occupancy_logits, target, batch["fill_mask"], batch["target_valid"],
            batch["camera_position"], batch["camera_direction"],
            camera_valid, batch["fov_x"], batch["fov_y"],
            air_class=config.air_class, height=config.projection_height,
            width=config.projection_width, samples=config.projection_samples,
            alpha_reference_step=config.projection_alpha_reference_step,
            max_distance=config.projection_max_distance,
            edge_weight=config.projection_edge_weight,
        )
        projection_loss = (
            config.projection_surface_weight * projection_metrics["projection_surface_loss"]
            + config.projection_free_space_weight
            * projection_metrics["projection_free_space_loss"]
            + config.projection_silhouette_weight
            * projection_metrics["projection_silhouette_loss"]
            + config.projection_depth_weight * projection_metrics["projection_depth_loss"]
        )
        total = total + config.projection_weight * projection_loss
        projection_metrics["projection_loss"] = projection_loss
    total = total + config.camera_position_weight * camera_position_loss
    total = total + config.camera_direction_weight * camera_direction_loss
    return total, {
        **visible_metrics,
        "loss": total,
        "voxel_loss": voxel_loss,
        "occupancy_loss": occupancy_loss,
        "camera_position_loss": camera_position_loss,
        "camera_direction_loss": camera_direction_loss,
        "pixel_semantic_loss": pixel_semantic_loss,
        "pixel_semantic_accuracy": pixel_semantic_accuracy,
        "pixel_semantic_upweighted_accuracy": pixel_semantic_upweighted_accuracy,
        "highres_point_loss": highres_point_loss,
        "highres_point_mae_blocks": highres_point_mae_blocks,
        "splat_support_fraction": (support > 1e-4).float().mean(),
        "surface_evidence_gain": (
            output["surface_evidence_gain"].float()
            if output.get("surface_evidence_gain") is not None else total.new_zeros(())
        ),
        "free_space_evidence_gain": (
            output["free_space_evidence_gain"].float()
            if output.get("free_space_evidence_gain") is not None else total.new_zeros(())
        ),
        "visibility_residual_abs": (
            output["visibility_residual"].float().abs().mean()
            if output.get("visibility_residual") is not None else total.new_zeros(())
        ),
        "visibility_evidence_fraction": (
            (output["visibility_confidence"].float() > 1e-4).float().mean()
            if output.get("visibility_confidence") is not None else total.new_zeros(())
        ),
        **projection_metrics,
    }


class GeometryFillTrainer:
    def __init__(self, model: nn.Module, config: GeometryFillTrainerConfig):
        self.model = model.to(config.device)
        self.config = config
        evidence_parameters = []
        semantic_head_parameters = []
        geometry_multiscale_parameters = []
        base_parameters = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "visibility_refiner" in name or "evidence_gain_raw" in name:
                evidence_parameters.append(parameter)
            elif "semantic" in name or name.endswith("pixel_semantic_embedding"):
                semantic_head_parameters.append(parameter)
            elif any(
                f"geometry.{module_name}." in name
                for module_name in ("shallow_adapter", "middle_adapter", "multiscale_fusion")
            ):
                geometry_multiscale_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        parameter_groups = []
        if base_parameters:
            parameter_groups.append({"params": base_parameters})
        if evidence_parameters:
            parameter_groups.append({
                "params": evidence_parameters,
                "lr": config.learning_rate * config.evidence_learning_rate_multiplier,
            })
        if semantic_head_parameters:
            parameter_groups.append({
                "params": semantic_head_parameters,
                "lr": config.learning_rate * config.semantic_head_learning_rate_multiplier,
            })
        if geometry_multiscale_parameters:
            parameter_groups.append({
                "params": geometry_multiscale_parameters,
                "lr": config.learning_rate
                * config.geometry_multiscale_learning_rate_multiplier,
            })
        self.optimizer = torch.optim.AdamW(
            parameter_groups, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=config.precision == "fp16" and str(config.device).startswith("cuda")
        )
        self.step = 0
        self._accumulating = False
        self.last_metrics: dict[str, float] = {}

    @property
    def raw_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _move(self, batch):
        batch = {key: value.to(self.config.device, non_blocking=True) for key, value in batch.items()}
        if self.config.visible_only and (
            "visible_surface_mask" not in batch
            or (self.config.ray_free_weight and "visible_ray_free_weights" not in batch)
        ):
            masks = visible_voxel_masks(
                batch["target"], batch["target_valid"], batch["camera_position"],
                batch["camera_direction"], batch["camera_valid"] & batch["agent_mask"],
                batch["fov_x"], batch["fov_y"], self.config.air_class,
                height=self.config.projection_height, width=self.config.projection_width,
                samples=self.config.projection_samples,
                max_distance=self.config.projection_max_distance,
                return_ray_weights=bool(self.config.ray_free_weight),
            )
            surface, free = masks[:2]
            batch["visible_surface_mask"], batch["visible_free_mask"] = surface, free
            if self.config.ray_free_weight:
                batch["visible_ray_free_weights"] = masks[2]
        return batch

    def _autocast(self):
        enabled = str(self.config.device).startswith("cuda") and self.config.precision != "fp32"
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    def teacher_splat_weight(self) -> float:
        if self.step < self.config.teacher_splat_steps:
            return 1.0
        if self.config.teacher_splat_decay_steps <= 0:
            return 0.0
        progress = (self.step - self.config.teacher_splat_steps) / self.config.teacher_splat_decay_steps
        return max(0.0, 1.0 - progress)

    def _forward(self, batch):
        return self.model(
            batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
            batch["images"], batch["agent_mask"],
            splat_camera_position=batch["camera_position"],
            splat_camera_direction=batch["camera_direction"],
            splat_teacher_weight=self.teacher_splat_weight(), return_aux=True,
        )

    def train_step(self, batch, *, update: bool = True, loss_divisor: int = 1) -> float:
        self.model.train()
        batch = self._move(batch)
        if not self._accumulating:
            self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self._forward(batch)
        with torch.autocast(device_type=output["voxel_logits"].device.type, enabled=False):
            total, metrics = geometry_fill_loss(output, batch, self.config)
        self.scaler.scale(total / loss_divisor).backward()
        self._accumulating = not update
        if update:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
        self.last_metrics = {name: float(value.detach()) for name, value in metrics.items()}
        self.last_metrics["teacher_splat_weight"] = self.teacher_splat_weight()
        return self.last_metrics["loss"]

    @torch.no_grad()
    def evaluate_step(self, batch, *, predicted_splat: bool = True, return_output: bool = False):
        self.model.eval()
        batch = self._move(batch)
        with self._autocast():
            if predicted_splat:
                output = self.model(
                    batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
                    batch["images"], batch["agent_mask"], return_aux=True,
                )
            else:
                output = self._forward(batch)
        total, losses = geometry_fill_loss(output, batch, self.config)
        valid = batch["fill_mask"] & batch["target_valid"]
        prediction = decode_voxel_prediction(output, self.config.air_class)
        correct = (prediction == batch["target"]) & valid
        occupied = (batch["target"] != self.config.air_class) & valid
        predicted_occupied = (prediction != self.config.air_class) & valid
        intersection = (occupied & predicted_occupied).sum()
        union = (occupied | predicted_occupied).sum()
        metrics = {name: float(value) for name, value in losses.items()}
        metrics.update(
            voxel_accuracy=float(correct.sum() / valid.sum().clamp_min(1)),
            non_air_accuracy=float(correct[occupied].sum() / occupied.sum().clamp_min(1)),
            occupancy_iou=float(intersection / union.clamp_min(1)),
        )
        return (metrics, output, batch) if return_output else metrics

    def save(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.raw_model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "step": self.step, "num_block_classes": self.raw_model.num_block_classes,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.raw_model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step = int(checkpoint.get("step", 0))
        return checkpoint
