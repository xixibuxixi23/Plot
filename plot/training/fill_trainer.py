from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from plot.models.fill import camera_pose_loss, masked_fill_loss
from plot.models.projection import projection_consistency_loss


@dataclass
class FillTrainerConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 2e-3
    max_grad_norm: float = 1.0
    camera_position_weight: float = 5.0
    camera_direction_weight: float = 1.0
    projection_silhouette_weight: float = 0.0
    projection_depth_weight: float = 0.0
    projection_start_step: int = 0
    projection_warmup_steps: int = 0
    projection_height: int = 24
    projection_width: int = 40
    projection_samples: int = 64
    projection_max_distance: float = 32.0
    projection_batch_size: int = 0
    air_class: int = 0
    precision: str = "bf16"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FillTrainer:
    def __init__(self, model: nn.Module, config: FillTrainerConfig):
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

    def _losses(self, batch):
        output = self.model(
            batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
            batch["images"], batch["agent_mask"], batch.get("image_condition_mask"),
            return_aux=True,
        )
        voxel = masked_fill_loss(
            output["voxel_logits"], batch["target"], batch["fill_mask"],
            batch.get("target_valid"),
        )
        position, direction = camera_pose_loss(
            output["camera_position"], output["camera_direction"],
            batch["camera_position"], batch["camera_direction"],
            batch["camera_valid"]
            & batch["agent_mask"].bool()
            & batch.get(
                "image_condition_mask",
                torch.ones(len(batch["camera_valid"]), device=batch["camera_valid"].device, dtype=torch.bool),
            )[:, None],
        )
        zero = voxel.new_zeros(())
        silhouette, depth = zero, zero
        if self.config.projection_warmup_steps > 0:
            projection_scale = min(
                1.0,
                max(0.0, self.step - self.config.projection_start_step)
                / self.config.projection_warmup_steps,
            )
        else:
            projection_scale = float(self.step >= self.config.projection_start_step)
        projection_enabled = (
            projection_scale > 0
            and (
                self.config.projection_silhouette_weight > 0
                or self.config.projection_depth_weight > 0
            )
        )
        if projection_enabled:
            logits_batch = output["voxel_logits"]
            count = self.config.projection_batch_size
            if count > 0:
                count = min(count, len(logits_batch))
                logits_batch = logits_batch[:count]
            else:
                count = len(logits_batch)
            silhouette, depth = projection_consistency_loss(
                logits_batch, batch["target"][:count], batch["fill_mask"][:count],
                batch["target_valid"][:count], batch["camera_position"][:count],
                batch["camera_direction"][:count],
                (batch["camera_valid"] & batch["agent_mask"].bool())[:count],
                batch["fov_x"][:count], batch["fov_y"][:count],
                air_class=self.config.air_class,
                height=self.config.projection_height, width=self.config.projection_width,
                samples=self.config.projection_samples,
                max_distance=self.config.projection_max_distance,
            )
        total = voxel + self.config.camera_position_weight * position
        total = total + self.config.camera_direction_weight * direction
        total = total + projection_scale * (
            self.config.projection_silhouette_weight * silhouette
            + self.config.projection_depth_weight * depth
        )
        return total, voxel, position, direction, silhouette, depth, projection_scale, output

    def train_step(self, batch, *, update: bool = True, loss_divisor: int = 1) -> float:
        self.model.train()
        batch = self._move(batch)
        if not self._accumulating:
            self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            total, voxel, position, direction, silhouette, depth, projection_scale, _ = (
                self._losses(batch)
            )
            backward_loss = total / loss_divisor
        self.scaler.scale(backward_loss).backward()
        self._accumulating = not update
        if update:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.step += 1
        self.last_metrics = {
            "loss": float(total.detach()), "voxel_loss": float(voxel.detach()),
            "camera_position_loss": float(position.detach()),
            "camera_direction_loss": float(direction.detach()),
            "projection_silhouette_loss": float(silhouette.detach()),
            "projection_depth_loss": float(depth.detach()),
            "projection_scale": projection_scale,
        }
        return self.last_metrics["loss"]

    @torch.no_grad()
    def evaluate_step(self, batch, *, return_output: bool = False):
        self.model.eval()
        batch = self._move(batch)
        with self._autocast():
            total, voxel, position, direction, silhouette, depth, projection_scale, output = (
                self._losses(batch)
            )
        valid = batch["fill_mask"] & batch.get("target_valid", torch.ones_like(batch["fill_mask"]))
        correct = ((output["voxel_logits"].argmax(1) == batch["target"]) & valid).sum()
        metrics = {
            "loss": float(total), "voxel_loss": float(voxel),
            "camera_position_loss": float(position), "camera_direction_loss": float(direction),
            "projection_silhouette_loss": float(silhouette),
            "projection_depth_loss": float(depth), "projection_scale": projection_scale,
            "voxel_accuracy": float(correct / valid.sum().clamp_min(1)),
        }
        return (metrics, output, batch) if return_output else metrics

    def save(self, path: str | Path, extra: dict | None = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        model = self.raw_model
        payload = {
            "model": model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "step": self.step, "num_block_classes": model.num_block_classes,
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

    def initialize_model(self, path: str | Path):
        """Warm-start model weights while resetting optimizer and step."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        missing, unexpected = self.raw_model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = {"null_image_token"}
        if set(missing) - allowed_missing or unexpected:
            raise ValueError(f"Incompatible initialization: missing={missing}, unexpected={unexpected}")
        return {"source_step": int(checkpoint.get("step", 0)), "missing": list(missing)}
