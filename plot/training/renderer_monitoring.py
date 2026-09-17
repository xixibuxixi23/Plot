"""Fixed qualitative probes and video output for M3 validation."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch

from .renderer_trainer import RendererRollout, slice_conditions


@dataclass(frozen=True)
class RendererProbe:
    name: str
    dataset_index: int
    episode: str
    start: int
    target: int
    scenario_id: str
    event: str | None


PROBE_SPECS = (
    ("construction", "S01", {"block_placed", "block_dug"}),
    ("two_player_motion", "S02", set()),
    ("pve_combat", "S08", {"damage", "attack_contact"}),
    ("three_resident_combat", "S09", {"damage", "attack_contact"}),
    ("mixed_build_combat", "S10", {"damage"}),
)


def _episode_events(path: Path) -> list[dict]:
    event_path = path / "events.jsonl"
    if not event_path.exists():
        return []
    return [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]


def select_renderer_probes(dataset, specs=PROBE_SPECS) -> list[RendererProbe]:
    """Choose stable val-ID windows whose first predicted chunk shows the event."""
    event_cache: dict[int, list[dict]] = {}
    selected = []
    for name, scenario, desired_events in specs:
        fallback = None
        choice = None
        for dataset_index, (episode_id, start, target) in enumerate(dataset.index):
            path, manifest = dataset.episodes[episode_id]
            if manifest.get("scenario_id") != scenario:
                continue
            if fallback is None:
                fallback = (dataset_index, episode_id, start, target, None)
            if not desired_events:
                choice = fallback
                break
            if episode_id not in event_cache:
                event_cache[episode_id] = _episode_events(Path(path))
            events = event_cache[episode_id]
            for event in events:
                frame = int(event.get("observation_frame", -1))
                if event.get("event") in desired_events and start < frame <= start + 8:
                    # Prefer a camera belonging to an actor/recipient of the event.
                    participants = {event.get("actor"), event.get("source"), event.get("target")}
                    if f"agent{target}" in participants or choice is None:
                        choice = (dataset_index, episode_id, start, target, event.get("event"))
                        if f"agent{target}" in participants:
                            break
            if choice is not None and choice[-1] is not None:
                break
        choice = choice or fallback
        if choice is None:
            raise ValueError(
                f"no validation window found for visualization probe {name} ({scenario})"
            )
        dataset_index, episode_id, start, target, event = choice
        path, manifest = dataset.episodes[episode_id]
        selected.append(
            RendererProbe(
                name,
                dataset_index,
                str(path),
                int(start),
                int(target),
                str(manifest.get("scenario_id")),
                event,
            )
        )
    return selected


def save_probe_manifest(probes: Iterable[RendererProbe], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(probe) for probe in probes], indent=2))
    return path


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    result = frame.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 25), (0, 0, 0), -1)
    cv2.putText(
        result, text, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
    )
    return result


def write_comparison_video(
    path: str | Path,
    ground_truth: torch.Tensor,
    prediction: torch.Tensor,
    region_weight: torch.Tensor,
    *,
    fps: float = 8.0,
) -> Path:
    """Write GT | prediction, with supervised regions outlined."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gt = (ground_truth.detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(
        np.uint8
    )
    pred = (prediction.detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).numpy() * 255).astype(
        np.uint8
    )
    weights = region_weight.detach().float().cpu().numpy()
    h, w = gt.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create video {path}")
    try:
        for index, (truth, estimate) in enumerate(zip(gt, pred)):
            mask = cv2.resize(
                (weights[index, 0] > 1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            estimate_bgr = cv2.cvtColor(estimate, cv2.COLOR_RGB2BGR)
            cv2.drawContours(estimate_bgr, contours, -1, (0, 255, 255), 1)
            panel = np.concatenate(
                (
                    _label(cv2.cvtColor(truth, cv2.COLOR_RGB2BGR), f"GT t+{index + 1}"),
                    _label(estimate_bgr, "M3 rollout (yellow=entity region)"),
                ),
                axis=1,
            )
            writer.write(panel)
    finally:
        writer.release()
    return path


@torch.no_grad()
def render_probe(
    model, codec, sample: dict, output_path: str | Path, *, seed: int,
    denoising_steps: int = 20, precision: str = "bf16"
) -> dict:
    """Run the deployment path in the configured training precision."""
    if precision not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    device = next(model.parameters()).device
    rgb = sample["rgb"][:, :65].to(device)
    conditions = {key: value.to(device) for key, value in sample["conditions"].items()}
    conditions = slice_conditions(conditions, 0, 65)
    use_bf16 = precision == "bf16" and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        latent = codec.encode(rgb)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            latent[:, 1:65].shape, device=device, dtype=latent.dtype, generator=generator
        )
        rollout = RendererRollout(model, denoising_steps=denoising_steps)
        try:
            rollout.start(latent[:, :1], slice_conditions(conditions, 0, 1))
            predicted_latent = rollout.generate_64(noise, slice_conditions(conditions, 1, 65))
            prediction = codec.decode(predicted_latent)[0]
        finally:
            model.clear_cache()
    truth = rgb[0, 1:65]
    weight = sample["region_weight"][0, 1:65]
    write_comparison_video(output_path, truth, prediction, weight)
    error = (prediction.float() - truth.float()).abs()
    mse = error.square().mean().clamp_min(1e-12)
    if "pixel_region_mask" in sample:
        entity_mask = sample["pixel_region_mask"][0, 1:65].to(device).float()
    else:
        entity_mask = (weight > 1).to(device).float()
    entity_l1 = (error * entity_mask).sum() / (
        entity_mask.sum().clamp_min(1) * error.shape[1]
    )
    height, width = error.shape[-2:]
    left, right = round(190 / 640 * width), round(314 / 640 * width)
    top, bottom = round(300 / 360 * height), round(322 / 360 * height)
    health_l1 = error[..., top:bottom, left:right].mean()
    metrics = {
        "l1": float(error.mean()),
        "psnr": float(-10 * torch.log10(mse)),
        "entity_l1": float(entity_l1),
        "health_l1": float(health_l1),
    }
    if "player_region_mask" in sample:
        player_mask = sample["player_region_mask"][0, 1:65].to(device).float()
        player_pixels = player_mask.sum()
        if player_pixels > 0:
            player_l1 = (error * player_mask).sum() / (player_pixels * error.shape[1])
            pred_dx = (prediction[..., 1:] - prediction[..., :-1]).abs().mean(1, keepdim=True)
            true_dx = (truth[..., 1:] - truth[..., :-1]).abs().mean(1, keepdim=True)
            pred_dy = (prediction[..., 1:, :] - prediction[..., :-1, :]).abs().mean(1, keepdim=True)
            true_dy = (truth[..., 1:, :] - truth[..., :-1, :]).abs().mean(1, keepdim=True)
            mask_x = torch.maximum(player_mask[..., 1:], player_mask[..., :-1])
            mask_y = torch.maximum(player_mask[..., 1:, :], player_mask[..., :-1, :])
            pred_detail = (pred_dx * mask_x).sum() + (pred_dy * mask_y).sum()
            true_detail = (true_dx * mask_x).sum() + (true_dy * mask_y).sum()
            metrics.update(
                player_l1=float(player_l1),
                player_detail_ratio=float(pred_detail / true_detail.clamp_min(1e-8)),
                player_pixels=float(player_pixels),
            )
        else:
            metrics["player_pixels"] = 0.0
    return metrics
