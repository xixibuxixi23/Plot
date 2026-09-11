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
    ("four_player", "S06", set()),
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
    """Write GT | prediction | absolute error, with supervised regions outlined."""
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
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 3, h))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create video {path}")
    try:
        for index, (truth, estimate) in enumerate(zip(gt, pred)):
            error = np.abs(truth.astype(np.int16) - estimate.astype(np.int16)).astype(np.uint8)
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
                    _label(cv2.cvtColor(error, cv2.COLOR_RGB2BGR), "absolute RGB error"),
                ),
                axis=1,
            )
            writer.write(panel)
    finally:
        writer.release()
    return path


@torch.no_grad()
def render_probe(
    model, codec, sample: dict, output_path: str | Path, *, seed: int, denoising_steps: int = 20
) -> dict:
    """Run the deployment path: one known frame followed by eight cached chunks."""
    device = next(model.parameters()).device
    rgb = sample["rgb"][:, :65].to(device)
    conditions = {key: value.to(device) for key, value in sample["conditions"].items()}
    conditions = slice_conditions(conditions, 0, 65)
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
    return {"l1": float(error.mean()), "psnr": float(-10 * torch.log10(mse))}
