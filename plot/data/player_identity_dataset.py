"""Lightweight rendered-player/reference pairs for identity pretraining."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .fill_dataset import _read_video_frame


def _rgba_reference(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"missing appearance view: {path}")
    code = cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGBA
    image = cv2.cvtColor(image, code).astype(np.float32) / 255.0
    image[..., :3] *= image[..., 3:4]
    return np.moveaxis(image, -1, 0)


def _masked_crop(rgb: np.ndarray, mask: np.ndarray, output_size=(128, 64), padding=0.12):
    out_h, out_w = output_size
    points = np.argwhere(mask)
    if not len(points):
        return np.zeros((4, out_h, out_w), np.float32)
    y0, x0 = points.min(0)
    y1, x1 = points.max(0) + 1
    pad_y = max(1, round((y1 - y0) * padding))
    pad_x = max(1, round((x1 - x0) * padding))
    y0, y1 = max(0, y0 - pad_y), min(rgb.shape[-2], y1 + pad_y)
    x0, x1 = max(0, x0 - pad_x), min(rgb.shape[-1], x1 + pad_x)
    alpha = mask[y0:y1, x0:x1].astype(np.float32)[None]
    rgba = np.concatenate((rgb[:, y0:y1, x0:x1] * alpha, alpha), axis=0)
    scale = min(out_h / rgba.shape[-2], out_w / rgba.shape[-1])
    height = max(1, min(out_h, round(rgba.shape[-2] * scale)))
    width = max(1, min(out_w, round(rgba.shape[-1] * scale)))
    resized = cv2.resize(np.moveaxis(rgba, 0, -1), (width, height), interpolation=cv2.INTER_AREA)
    if resized.ndim == 2:
        resized = resized[..., None]
    canvas = np.zeros((out_h, out_w, 4), np.float32)
    top, left = (out_h - height) // 2, (out_w - width) // 2
    canvas[top:top + height, left:left + width] = resized
    return np.moveaxis(canvas, -1, 0)


class PlayerIdentityDataset(Dataset):
    """Sample one visible non-camera player without loading voxels or 65 RGB frames."""

    def __init__(self, root, window_index, *, image_size=(360, 640), crop_size=(128, 64),
                 min_pixels=64, retry=8, canonical_only=False):
        payload = torch.load(window_index, map_location="cpu", weights_only=False)
        dataset_root = Path(root)
        self.episodes = [
            (
                path if (path := Path(row["path"])).is_absolute() else dataset_root / path,
                row["manifest"],
            )
            for row in payload["episodes"]
        ]
        self.windows = [tuple(map(int, row)) for row in payload["windows"]]
        self.context_frames = int(payload["context_frames"])
        self.image_size, self.crop_size = tuple(image_size), tuple(crop_size)
        self.min_pixels, self.retry = int(min_pixels), int(retry)
        self.canonical_only = bool(canonical_only)
        if self.min_pixels < 1 or self.retry < 1:
            raise ValueError("min_pixels and retry must be positive")

    def __len__(self):
        return len(self.windows)

    def _sample(self, index):
        episode_id, start, target = self.windows[index]
        path, manifest = self.episodes[episode_id]
        agents = int(manifest["num_agents"])
        if self.canonical_only:
            human_slots = [
                slot for slot in range(agents)
                if manifest["agent_kinds"][f"agent{slot}"] == "human_like"
            ]
            if not human_slots:
                return None
            slot = human_slots[np.random.randint(len(human_slots))]
            references = np.stack([
                _rgba_reference(path / "players" / f"agent{slot}" / f"{view}.png")
                for view in ("front", "back", "left", "right")
            ])
            view = references[np.random.randint(4)]
            crop = _masked_crop(view[:3], view[3] > 0.02, self.crop_size)
            # Simulate distance, interpolation and lighting while retaining the
            # skin identity. The crop tower never sees the transparent RGB.
            scale = float(np.random.uniform(0.35, 1.0))
            small_h = max(8, round(self.crop_size[0] * scale))
            small_w = max(4, round(self.crop_size[1] * scale))
            small = cv2.resize(
                np.moveaxis(crop, 0, -1), (small_w, small_h), interpolation=cv2.INTER_AREA
            )
            crop = cv2.resize(
                small, self.crop_size[::-1], interpolation=cv2.INTER_LINEAR
            )
            brightness = float(np.random.uniform(0.7, 1.15))
            crop[..., :3] = np.clip(crop[..., :3] * brightness, 0, 1)
            crop[..., :3] *= crop[..., 3:4]
            crop = np.moveaxis(crop.astype(np.float32), -1, 0)
            digest = hashlib.blake2b(references.tobytes(), digest_size=8).digest()
            identity = int.from_bytes(digest, "little") & ((1 << 63) - 1)
            return {
                "crop": torch.from_numpy(crop.copy()),
                "reference": torch.from_numpy(references),
                "identity": torch.tensor(identity, dtype=torch.int64),
                "valid": torch.tensor(True),
                "pixels": torch.tensor(int((crop[3] > 0.02).sum()), dtype=torch.int64),
            }
        human_slots = [
            slot for slot in range(agents)
            if slot != target and manifest["agent_kinds"][f"agent{slot}"] == "human_like"
        ]
        if not human_slots:
            return None
        end = start + self.context_frames
        with np.load(path / manifest.get("training_data_file", "data.npz"), allow_pickle=False) as data:
            if "entity_render_object_id" not in data:
                return None
            entity_ids = list(data["entity_id"].astype(str))
            entity_slots = [entity_ids.index(f"agent{slot}") for slot in human_slots]
            masks = data["instance_mask"][start:end, target]
            render_ids = data["entity_render_object_id"][start:end, entity_slots]
            candidates = []
            for local_slot, slot in enumerate(human_slots):
                ids = render_ids[:, local_slot]
                valid_ids = (ids > 0) & (ids != np.iinfo(np.uint16).max)
                slot_masks = (masks == ids[:, None, None]) & valid_ids[:, None, None]
                coverage = slot_masks.reshape(len(slot_masks), -1).sum(1)
                for frame in np.flatnonzero(coverage >= self.min_pixels):
                    if frame > 0:
                        candidates.append((int(coverage[frame]), int(frame), slot, slot_masks[frame]))
        if not candidates:
            return None
        # Favor readable residents without always selecting the same frame.
        candidates.sort(key=lambda row: row[0], reverse=True)
        candidates = candidates[: min(16, len(candidates))]
        weights = np.sqrt(np.asarray([row[0] for row in candidates], np.float64))
        choice = candidates[np.random.choice(len(candidates), p=weights / weights.sum())]
        _, frame, slot, mask = choice
        videos = manifest.get("agent_video_files") or [f"rgb_agent{i}.mp4" for i in range(agents)]
        rgb = _read_video_frame(path / videos[target], start + frame, self.image_size)
        references = np.stack([
            _rgba_reference(path / "players" / f"agent{slot}" / f"{view}.png")
            for view in ("front", "back", "left", "right")
        ])
        digest = hashlib.blake2b(references.tobytes(), digest_size=8).digest()
        identity = int.from_bytes(digest, "little") & ((1 << 63) - 1)
        return {
            "crop": torch.from_numpy(_masked_crop(rgb, mask, self.crop_size)),
            "reference": torch.from_numpy(references),
            "identity": torch.tensor(identity, dtype=torch.int64),
            "valid": torch.tensor(True),
            "pixels": torch.tensor(choice[0], dtype=torch.int64),
        }

    def __getitem__(self, index):
        for offset in range(self.retry):
            sample = self._sample((index + offset * 104729) % len(self))
            if sample is not None:
                return sample
        return {
            "crop": torch.zeros(4, *self.crop_size),
            "reference": torch.zeros(4, 4, 256, 128),
            "identity": torch.tensor(-1, dtype=torch.int64),
            "valid": torch.tensor(False),
            "pixels": torch.tensor(0, dtype=torch.int64),
        }
