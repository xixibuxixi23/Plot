from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from plot.models.geometry_bootstrap import geometry_targets_from_voxels
from plot.camera_geometry import unproject_depth_to_reference


@dataclass
class GeometryBootstrapTrainerConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 2e-3
    max_grad_norm: float = 1.0
    translation_weight: float = 1.0
    rotation_weight: float = 1.0
    visibility_weight: float = 0.5
    depth_weight: float = 1.0
    point_weight: float = 1.0
    consistency_weight: float = 0.5
    gradient_weight: float = 0.2
    uncertainty_regularizer: float = 0.05
    ray_samples: int = 64
    max_distance: float = 32.0
    supervised_height_fraction: float = 0.82
    air_class: int = 0
    precision: str = "bf16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(value.dtype)
    return (value * mask).sum() / mask.sum().clamp_min(1)


def _gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Edge-aware multi-channel gradient loss over valid neighboring rays."""
    if prediction.ndim == valid.ndim + 1:
        valid = valid.unsqueeze(2)
    horizontal_valid = valid[..., :, 1:] * valid[..., :, :-1]
    vertical_valid = valid[..., 1:, :] * valid[..., :-1, :]
    horizontal = (
        (prediction[..., :, 1:] - prediction[..., :, :-1])
        - (target[..., :, 1:] - target[..., :, :-1])
    ).abs()
    vertical = (
        (prediction[..., 1:, :] - prediction[..., :-1, :])
        - (target[..., 1:, :] - target[..., :-1, :])
    ).abs()
    return 0.5 * (
        _masked_mean(horizontal, horizontal_valid.expand_as(horizontal))
        + _masked_mean(vertical, vertical_valid.expand_as(vertical))
    )


def _geometry_bootstrap_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    config: GeometryBootstrapTrainerConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    camera_valid = target["camera_valid"]
    pose_valid = camera_valid.clone()
    pose_valid[:, 0] = False
    translation_error = F.smooth_l1_loss(
        output["relative_translation"], target["relative_translation"], reduction="none"
    ).mean(-1)
    relative_rotation = output["relative_rotation"].transpose(-1, -2) @ target["relative_rotation"]
    rotation_cosine = ((relative_rotation.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) * 0.5)
    rotation_error = 1.0 - rotation_cosine.clamp(-1.0, 1.0)
    translation_loss = _masked_mean(translation_error, pose_valid)
    rotation_loss = _masked_mean(rotation_error, pose_valid)

    pixel_valid = target["pixel_valid"]
    visibility = target["visibility"]
    visibility_loss = F.binary_cross_entropy_with_logits(
        output["visibility_logits"], visibility, reduction="none"
    )
    visibility_loss = _masked_mean(visibility_loss, pixel_valid.expand_as(visibility_loss))
    hit_valid = pixel_valid & visibility.bool()
    depth_error = F.smooth_l1_loss(output["depth"], target["depth"], reduction="none")
    point_error = F.smooth_l1_loss(
        output["point_map"], target["point_map"], reduction="none"
    ).mean(2)
    log_uncertainty = output["log_uncertainty"]
    joint_error = depth_error + point_error
    uncertainty_loss = torch.exp(-log_uncertainty) * joint_error
    uncertainty_loss = uncertainty_loss + config.uncertainty_regularizer * log_uncertainty
    uncertainty_loss = _masked_mean(uncertainty_loss, hit_valid)
    depth_loss = _masked_mean(depth_error, hit_valid)
    point_loss = _masked_mean(point_error, hit_valid)
    unprojected_point_map = unproject_depth_to_reference(
        output["depth"], output["relative_translation"], output["relative_rotation"],
        target["local_rays"],
    )
    consistency_error = F.smooth_l1_loss(
        output["point_map"], unprojected_point_map, reduction="none"
    ).mean(2)
    consistency_loss = _masked_mean(consistency_error, hit_valid)
    gradient_loss = _gradient_loss(output["depth"], target["depth"], hit_valid)
    gradient_loss = gradient_loss + _gradient_loss(
        output["point_map"], target["point_map"], hit_valid
    )
    total = config.translation_weight * translation_loss
    total = total + config.rotation_weight * rotation_loss
    total = total + config.visibility_weight * visibility_loss
    total = total + config.depth_weight * depth_loss + config.point_weight * point_loss
    total = total + config.consistency_weight * consistency_loss
    total = total + config.gradient_weight * gradient_loss + uncertainty_loss
    metrics = {
        "loss": total,
        "translation_loss": translation_loss,
        "rotation_loss": rotation_loss,
        "visibility_loss": visibility_loss,
        "depth_loss": depth_loss,
        "point_loss": point_loss,
        "consistency_loss": consistency_loss,
        "gradient_loss": gradient_loss,
        "uncertainty_loss": uncertainty_loss,
    }
    return total, metrics


def geometry_bootstrap_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    config: GeometryBootstrapTrainerConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate geometric losses in FP32 even when the backbone uses BF16."""
    used_output_keys = (
        "relative_translation", "relative_rotation", "depth", "point_map",
        "visibility_logits", "log_uncertainty",
    )
    output_float = {key: output[key].float() for key in used_output_keys}
    target_float = {
        key: value.float() if torch.is_floating_point(value) else value
        for key, value in target.items()
    }
    with torch.autocast(device_type=output["depth"].device.type, enabled=False):
        return _geometry_bootstrap_loss(output_float, target_float, config)


class GeometryBootstrapTrainer:
    def __init__(self, model: nn.Module, config: GeometryBootstrapTrainerConfig):
        self.model = model.to(config.device)
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
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
        return {key: value.to(self.config.device, non_blocking=True) for key, value in batch.items()}

    def _autocast(self):
        enabled = str(self.config.device).startswith("cuda") and self.config.precision != "fp32"
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    def _targets(self, batch, output):
        return geometry_targets_from_voxels(
            batch["target"], batch["target_valid"], batch["camera_position"],
            batch["camera_direction"], batch["camera_valid"] & batch["agent_mask"],
            batch["fov_x"], batch["fov_y"], air_class=self.config.air_class,
            height=output["depth"].shape[-2], width=output["depth"].shape[-1],
            samples=self.config.ray_samples, max_distance=self.config.max_distance,
            supervised_height_fraction=self.config.supervised_height_fraction,
        )

    def train_step(self, batch, *, update: bool = True, loss_divisor: int = 1) -> float:
        self.model.train()
        batch = self._move(batch)
        if not self._accumulating:
            self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            output = self.model(batch["images"], batch["agent_mask"])
        target = self._targets(batch, output)
        total, metrics = geometry_bootstrap_loss(output, target, self.config)
        backward_loss = total / loss_divisor
        self.scaler.scale(backward_loss).backward()
        self._accumulating = not update
        if update:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
        self.last_metrics = {name: float(value.detach()) for name, value in metrics.items()}
        return self.last_metrics["loss"]

    @torch.no_grad()
    def evaluate_step(self, batch, *, return_output: bool = False):
        self.model.eval()
        batch = self._move(batch)
        with self._autocast():
            output = self.model(batch["images"], batch["agent_mask"])
        target = self._targets(batch, output)
        _, losses = geometry_bootstrap_loss(output, target, self.config)
        visibility_prediction = output["visibility_logits"] > 0
        visibility_target = target["visibility"].bool()
        pixel_valid = target["camera_valid"][:, :, None, None]
        intersection = (visibility_prediction & visibility_target & pixel_valid).sum()
        union = ((visibility_prediction | visibility_target) & pixel_valid).sum()
        metrics = {name: float(value) for name, value in losses.items()}
        metrics["visibility_iou"] = float(intersection / union.clamp_min(1))
        return (metrics, output, target, batch) if return_output else metrics

    def save(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
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
