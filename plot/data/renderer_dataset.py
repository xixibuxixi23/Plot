"""Continuous TextAgent observations -> M3 clips, without changing public data."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .chunked_npz import open_npz
from .fill_dataset import BlockVocabulary, TextAgentFillDataset, _read_video_frames


RESIDENT_TYPES = {"human_like": 0, "npc_villager": 1, "npc_zombie": 2, "npc_skeleton": 3}


def merge_observation_crops(raw, centers, anchor, vocabulary, preferred_index=None):
    """Map this observation's 49³ cubes into one fixed 48³ target window.

    Missing coverage is explicitly UNFILLED, never mislabeled air. No future
    observation is used to fill a boundary. Centers are integer ENU metadata.
    """
    blocks = np.zeros((48, 48, 48), np.int64)
    known = np.zeros_like(blocks, bool)
    lower = np.asarray(anchor, np.int64) - 24
    order = list(range(len(raw)))
    if preferred_index is not None:
        order.remove(int(preferred_index))
        order.insert(0, int(preferred_index))
    for observer in order:
        cube, center = raw[observer], centers[observer]
        source_lower = np.asarray(center, np.int64) - 24
        lo, hi = np.maximum(lower, source_lower), np.minimum(lower + 48, source_lower + 49)
        if np.any(lo >= hi):
            continue
        src = tuple(
            slice(int(low), int(high))
            for low, high in zip(lo - source_lower, hi - source_lower)
        )
        dst = tuple(
            slice(int(low), int(high)) for low, high in zip(lo - lower, hi - lower)
        )
        values, valid = vocabulary.encode(cube[src][..., 0])
        # Rare one-voxel edit boundaries can be visible to clients one render
        # apart. The target resident's own same-frame crop is authoritative;
        # other synchronized crops only extend currently unknown coverage.
        write = valid & ~known[dst]
        np.copyto(blocks[dst], values, where=write)
        known[dst] |= valid
    return blocks, known


def raster_camera(camera_world, camera_direction, fov_x, anchor):
    """ENU world camera -> normalized cube W2C, with center used only here.

    Raw integer block positions are block centers; a 48³ half-open tile's
    geometric center is anchor - .5. Translation is -R C, not C itself.
    """
    forward = np.asarray(camera_direction, np.float32)
    norm = np.linalg.norm(forward, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm < 1e-6):
        raise ValueError("invalid camera direction")
    forward = forward / norm
    down = np.broadcast_to(np.array([0, 0, -1], np.float32), forward.shape).copy()
    singular = np.abs(forward[..., 2]) > .999
    down[singular] = [0, -1, 0]
    right = np.cross(down, forward)
    right /= np.linalg.norm(right, axis=-1, keepdims=True)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=-2)
    local = (np.asarray(camera_world) - (np.asarray(anchor) - .5)) / 48.
    translation = -np.einsum("...ij,...j->...i", rotation, local)
    return np.concatenate((rotation[..., :2, :].reshape(*local.shape[:-1], 6),
                           translation, np.asarray(fov_x)[..., None]), axis=-1).astype(np.float32)


def incoming_actions(actions, start, length):
    result = np.zeros((length, *actions.shape[1:]), np.float32)
    # The first observation is a window boundary, with a learned prefix token.
    result[1:] = actions[start:start + length - 1]
    return result


class TextAgentRendererDataset(Dataset):
    def __init__(self, root, vocabulary, *, split="train", context_frames=65,
                 stride=8, image_size=(360, 640), latent_size=(36, 64), max_agents=8,
                 window_index=None, targets_per_window=1, entity_region_upweight=4.0,
                 health_focus_index=None, health_focus_oversample=1,
                 chunk_cache_root=None):
        if context_frames < 9 or (context_frames - 1) % 8 or stride < 1:
            raise ValueError("context_frames must be 1+8*k; stride must be positive")
        if targets_per_window < 1:
            raise ValueError("targets_per_window must be positive")
        if health_focus_oversample < 1:
            raise ValueError("health_focus_oversample must be positive")
        self.targets_per_window = int(targets_per_window)
        self.health_focus_targets = set()
        if entity_region_upweight < 0:
            raise ValueError("entity_region_upweight must be nonnegative")
        self.entity_region_upweight = float(entity_region_upweight)
        self.vocabulary = vocabulary if isinstance(vocabulary, BlockVocabulary) else BlockVocabulary.load(vocabulary)
        self.context_frames, self.image_size, self.latent_size = context_frames, image_size, latent_size
        self.chunk_cache_root = Path(chunk_cache_root) if chunk_cache_root is not None else None
        self.episodes, self.index = [], []
        self.item_vocabulary = None
        if window_index is not None:
            cached = torch.load(window_index, map_location="cpu", weights_only=False)
            if cached.get("context_frames") != context_frames or cached.get("split") != split:
                raise ValueError("M3 window index context/split does not match the dataset")
            self.item_vocabulary = cached["item_vocabulary"]
            dataset_root = Path(root)
            self.episodes = [
                (
                    path if (path := Path(row["path"])).is_absolute() else dataset_root / path,
                    row["manifest"],
                )
                for row in cached["episodes"]
            ]
            self.index = [tuple(value) for value in cached["windows"]]
            if not self.index:
                raise ValueError("M3 window index is empty")
            if self.targets_per_window > 1:
                self.index = self._group_target_windows(self.index, self.targets_per_window)
            self._configure_health_focus(
                health_focus_index, int(health_focus_oversample), context_frames, split
            )
            return
        for path in sorted(Path(root).rglob("manifest.json")):
            manifest = json.loads(path.read_text())
            if manifest.get("split") != split:
                continue
            validation = path.with_name("validation.json")
            if not validation.exists() or not json.loads(validation.read_text()).get("usable"):
                continue
            agents = int(manifest["num_agents"])
            if not 1 <= agents <= max_agents:
                raise ValueError(f"unsupported number of residents in {path}")
            metadata = json.loads(path.with_name("training_metadata.json").read_text())
            items = metadata["item_vocabulary"]
            if self.item_vocabulary is not None and items != self.item_vocabulary:
                raise ValueError("item vocabularies differ; unify IDs before M3 training")
            self.item_vocabulary = items
            with np.load(path.parent / manifest.get("training_data_file", "data.npz")) as data:
                frames = len(data["cam_pos"])
                if len(data["action_continuous"]) != frames - 1:
                    raise ValueError("M3 requires T actions and T+1 observations")
                start = TextAgentFillDataset._model_start(data, manifest)
                health = data["player_health"]
                valid = data["player_health_valid"].astype(bool) & np.isfinite(health)
                terminations = np.asarray(data["termination_flag"], bool)
                if terminations.ndim > 1:
                    terminations = terminations.any(axis=tuple(range(1, terminations.ndim)))
            episode_id = len(self.episodes)
            self.episodes.append((path.parent, manifest))
            for begin in range(start, frames - context_frames + 1, stride):
                end = begin + context_frames
                if terminations[begin:end-1].any() or not valid[begin:end].all():
                    continue
                for target in range(agents):
                    if (health[begin:end, target] <= 0).any():
                        continue
                    self.index.append((episode_id, begin, target))
        if not self.index:
            raise ValueError("no accepted continuous M3 windows with valid HP found")
        if self.targets_per_window > 1:
            self.index = self._group_target_windows(self.index, self.targets_per_window)
        self._configure_health_focus(
            health_focus_index, int(health_focus_oversample), context_frames, split
        )

    def _configure_health_focus(self, path, oversample, context_frames, split):
        if path is None:
            if oversample != 1:
                raise ValueError("health_focus_oversample requires health_focus_index")
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("context_frames") != context_frames or payload.get("split") != split:
            raise ValueError("health focus index context/split does not match the dataset")
        rows = torch.as_tensor(payload["rows"], dtype=torch.int64)
        if rows.ndim != 2 or rows.shape[1] != 3:
            raise ValueError("health focus rows must be [N,3] episode/start/target")
        self.health_focus_targets = {tuple(map(int, row)) for row in rows.tolist()}
        if oversample > 1:
            focused = [row for row in self.index if self._is_health_focus_window(row)]
            self.index.extend(focused * (oversample - 1))

    def _is_health_focus_window(self, row):
        episode, start, target = row
        targets = target if isinstance(target, (tuple, list)) else (target,)
        return any((int(episode), int(start), int(slot)) in self.health_focus_targets
                   for slot in targets)

    @staticmethod
    def _group_target_windows(index, required):
        """Collapse adjacent per-target rows into one window with valid targets.

        Cached indexes are emitted in episode/start/target order. Requiring the
        requested number of targets keeps the flattened training batch size
        identical on every DDP rank.
        """
        grouped = []
        previous = None
        targets = []
        for episode, start, target in index:
            key = (int(episode), int(start))
            if previous is not None and key != previous:
                if len(targets) >= required:
                    grouped.append((*previous, tuple(targets)))
                targets = []
            if previous is not None and key < previous:
                raise ValueError("M3 window index must be sorted by episode and start")
            previous = key
            targets.append(int(target))
        if previous is not None and len(targets) >= required:
            grouped.append((*previous, tuple(targets)))
        if not grouped:
            raise ValueError(f"no M3 windows have {required} valid target residents")
        return grouped

    @classmethod
    def read_window(cls, episode, vocabulary, *, start, target, context_frames,
                    image_size=(360, 640), latent_size=(36, 64)):
        """Read one explicit window without rescanning an episode index."""
        path = Path(episode)
        manifest = json.loads((path / "manifest.json").read_text())
        instance = cls.__new__(cls)
        instance.vocabulary = (vocabulary if isinstance(vocabulary, BlockVocabulary)
                               else BlockVocabulary.load(vocabulary))
        instance.context_frames = int(context_frames)
        instance.image_size, instance.latent_size = image_size, latent_size
        instance.entity_region_upweight = 4.0
        instance.chunk_cache_root = None
        instance.health_focus_targets = set()
        instance.episodes = [(path, manifest)]
        instance.index = [(0, int(start), int(target))]
        metadata = json.loads((path / "training_metadata.json").read_text())
        instance.item_vocabulary = metadata["item_vocabulary"]
        return instance[0]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        episode_id, start, target = self.index[index]
        if isinstance(target, (tuple, list)):
            focused = [i for i, slot in enumerate(target)
                       if (int(episode_id), int(start), int(slot)) in self.health_focus_targets]
            order = []
            if focused:
                order.append(focused[torch.randint(len(focused), ()).item()])
            remaining = [i for i in range(len(target)) if i not in order]
            shuffled = torch.randperm(len(remaining)).tolist()
            order.extend(remaining[i] for i in shuffled[: self.targets_per_window - len(order)])
            return [self._read_target(episode_id, start, target[i]) for i in order]
        return self._read_target(episode_id, start, target)

    def _read_target(self, episode_id, start, target):
        path, manifest = self.episodes[episode_id]
        t, a = self.context_frames, int(manifest["num_agents"])
        end = start + t
        data_path = path / manifest.get("training_data_file", "data.npz")
        if self.chunk_cache_root is not None:
            relative_cache = manifest.get("m3_chunk_cache_file")
            if not relative_cache:
                raise ValueError(
                    "--chunk-cache-root requires an index produced by "
                    "materialize_m3_chunk_cache.py"
                )
            data_path = self.chunk_cache_root / relative_cache
            if not data_path.is_file():
                raise FileNotFoundError(f"missing M3 chunk cache: {data_path}")
        with open_npz(data_path) as data:
            centers = data["obs_voxel_center"][start:end]
            raw = data["obs_voxel_mt"][start:end]
            # obs0 is the prefix. Frames 1..8 use obs0's anchor, 9..16 obs8's, etc.
            anchors = np.stack([centers[0 if i == 0 else ((i-1)//8)*8, target] for i in range(t)])
            crops = [merge_observation_crops(raw[i], centers[i], anchors[i], self.vocabulary,
                                             preferred_index=target)
                     for i in range(t)]
            position = data["player_pos"][start:end].astype(np.float32)
            camera = data["cam_pos"][start:end].astype(np.float32)
            direction = data["cam_dir"][start:end].astype(np.float32)
            fov = data["fov_x"][start:end].astype(np.float32)
            fov = np.broadcast_to(fov[:, None], (t, a)).copy() if fov.ndim == 1 else fov
            health = data["player_health"][start:end].astype(np.float32)
            entity_ids = list(data["entity_id"].astype(str))
            entity_slots = [entity_ids.index(f"agent{i}") for i in range(a)]
            # Canonical item-vocabulary IDs are transition-aligned. The final
            # observation has no outgoing action, so it retains the last item.
            wielded = data["wielded_item_id"]
            held_indices = np.minimum(np.arange(start, end), len(wielded) - 1)
            held = wielded[held_indices, :a].astype(np.int64)
            item_count = max(self.item_vocabulary.values(), default=0) + 1
            if held.min(initial=0) < 0 or held.max(initial=0) >= item_count:
                raise ValueError("wielded_item_id is outside the canonical item vocabulary")
            # Dataset player angles are degrees; the network consumes radians.
            angles = np.deg2rad(np.stack((data["player_yaw"][start:end],
                                          data["player_pitch"][start:end]), axis=-1)).astype(np.float32)
            masks = data["instance_mask"][start:end, target]
            if masks.dtype != np.uint16:
                raise ValueError("instance_mask must retain the raw uint16 entity IDs")
            # Keep a separate mask for visible, non-camera human players.  The
            # legacy entity mask also contains mobs, attachments and the
            # first-person wield view, so it cannot measure appearance quality.
            player_masks = np.zeros_like(masks, dtype=bool)
            identity_player_mask = np.zeros_like(masks[0], dtype=bool)
            identity_frame = 0
            identity_slot = 0
            identity_pixels = 0
            if "entity_render_object_id" in data and "entity_kind" in data:
                render_ids = data["entity_render_object_id"][start:end, entity_slots]
                entity_kinds = data["entity_kind"].astype(str)
                for slot, entity_slot in enumerate(entity_slots):
                    if slot == target or entity_kinds[entity_slot] != "player":
                        continue
                    ids = render_ids[:, slot]
                    valid_ids = (ids > 0) & (ids != np.iinfo(np.uint16).max)
                    slot_masks = (masks == ids[:, None, None]) & valid_ids[:, None, None]
                    player_masks |= slot_masks
                    coverage = slot_masks.reshape(t, -1).sum(1)
                    if len(coverage) > 1:
                        frame = int(coverage[1:].argmax()) + 1
                        if int(coverage[frame]) > identity_pixels:
                            identity_pixels = int(coverage[frame])
                            identity_frame = frame
                            identity_slot = slot
                            identity_player_mask = slot_masks[frame]
            weights = np.stack([cv2.resize((m != 0).astype(np.float32), self.latent_size[::-1],
                                            interpolation=cv2.INTER_AREA) for m in masks])
            actions = incoming_actions(data["action_continuous"], start, t)
            active = data["entity_valid"][start:end, entity_slots].astype(bool)
        kinds = manifest["agent_kinds"]
        types = np.broadcast_to([RESIDENT_TYPES[kinds[f"agent{i}"]] for i in range(a)], (t, a)).copy()
        cues = np.zeros((t, a, 4), np.float32)
        cues[1:, :, 3] = health[1:] - health[:-1]
        event_path = path / manifest.get("event_file", "events.jsonl")
        for line in event_path.read_text().splitlines():
            event = json.loads(line)
            frame = int(event.get("observation_frame", -1)) - start
            if not 1 <= frame < t:
                continue
            source, recipient = event.get("source", event.get("actor")), event.get("target")
            kind = event.get("event")
            for slot in range(a):
                if source == f"agent{slot}" and kind in {"block_dug", "block_placed"}:
                    cues[frame, slot, 0] += 1
                if kind == "damage":
                    if source == f"agent{slot}":
                        cues[frame, slot, 1] += 1
                    if recipient == f"agent{slot}":
                        cues[frame, slot, 2] += 1
        skins = []
        references = []
        for slot in range(a):
            views = []
            reference_views = []
            for view in ("front", "back", "left", "right"):
                image_path = path / "players" / f"agent{slot}" / f"{view}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise ValueError(f"missing appearance view: {image_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGBA)
                reference = image.astype(np.float32) / 255.
                # Preserve the native 256x128 matte and aspect ratio. Clearing
                # transparent RGB avoids hidden PNG colors leaking through
                # interpolation in the reference encoder.
                reference[..., :3] *= reference[..., 3:4]
                reference_views.append(np.moveaxis(reference, -1, 0))
                views.append(np.moveaxis(cv2.resize(image, (64, 64)).astype(np.float32) / 255., -1, 0))
            skins.append(np.stack(views))
            references.append(np.stack(reference_views))
        videos = manifest.get("agent_video_files") or [f"rgb_agent{i}.mp4" for i in range(a)]
        decoded = _read_video_frames(path / videos[target], range(start, end), self.image_size)
        rgb = np.stack([decoded[i] for i in range(start, end)])
        condition = {
            "voxel_classes": np.stack([c[0] for c in crops]),
            "voxel_known": np.stack([c[1] for c in crops]),
            "raster_camera": raster_camera(camera[:, target], direction[:, target], fov[:, target], anchors),
            "target_agent": np.asarray(target, np.int64), "player_position": position,
            "camera_relative": camera - position, "camera_direction": direction, "fov_x": fov,
            "hp": health, "yaw_pitch": angles, "held_item": held, "resident_type": types,
            "event_cues": cues, "action": actions, "player_valid": active,
            "player_skin": np.stack(skins), "player_appearance_valid": np.ones((a, 4), bool),
            "player_reference": np.stack(references),
            "condition_mask": np.arange(t) == 0, "action_prefix_mask": np.arange(t) == 0,
        }
        # Raw collections retain source-resolution uint16 masks while RGB can
        # be decoded at the configured training resolution. Resize IDs/masks
        # with nearest-neighbor sampling so pixel losses and prefix masking
        # always align with the RGB tensor without inventing mixed IDs.
        if tuple(masks.shape[-2:]) != tuple(self.image_size):
            pixel_region = np.stack([
                cv2.resize((mask != 0).astype(np.uint8), self.image_size[::-1],
                           interpolation=cv2.INTER_NEAREST).astype(bool)
                for mask in masks
            ])
            player_region = np.stack([
                cv2.resize(mask.astype(np.uint8), self.image_size[::-1],
                           interpolation=cv2.INTER_NEAREST).astype(bool)
                for mask in player_masks
            ])
            identity_player_mask = cv2.resize(
                identity_player_mask.astype(np.uint8), self.image_size[::-1],
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            pixel_region = masks != 0
            player_region = player_masks
        pixel_region_mask = pixel_region[:, None]
        player_region_mask = player_region[:, None]
        return {"rgb": torch.from_numpy(rgb),
                "region_weight": torch.from_numpy(
                    1 + self.entity_region_upweight * weights[:, None]
                ),
                "pixel_region_mask": torch.from_numpy(pixel_region_mask),
                "player_region_mask": torch.from_numpy(player_region_mask),
                "player_identity_mask": torch.from_numpy(identity_player_mask[None]),
                "player_identity_frame": torch.tensor(identity_frame, dtype=torch.int64),
                "player_identity_slot": torch.tensor(identity_slot, dtype=torch.int64),
                "player_identity_valid": torch.tensor(identity_pixels >= 96),
                "conditions": {k: torch.from_numpy(v) for k, v in condition.items()}}


def collate_renderer(samples):
    """Flatten selected target views, then pad resident-condition slots."""
    samples = [view for sample in samples for view in (sample if isinstance(sample, list) else [sample])]
    max_agents = max(s["conditions"]["player_position"].shape[1] for s in samples)
    temporal_agent = {"player_position", "camera_relative", "camera_direction", "fov_x", "hp",
                      "yaw_pitch", "held_item", "resident_type", "event_cues", "action", "player_valid"}
    result = {}
    for key in samples[0]["conditions"]:
        values = []
        for sample in samples:
            value = sample["conditions"][key]
            axis = 1 if key in temporal_agent else 0 if key in {
                "player_skin", "player_reference", "player_appearance_valid"
            } else None
            if axis is not None and value.shape[axis] < max_agents:
                shape = list(value.shape)
                shape[axis] = max_agents - shape[axis]
                value = torch.cat((value, value.new_zeros(shape)), dim=axis)
            values.append(value)
        result[key] = torch.stack(values)
    return {"conditions": result, "rgb": torch.stack([s["rgb"] for s in samples]),
            "region_weight": torch.stack([s["region_weight"] for s in samples]),
            "pixel_region_mask": torch.stack([s["pixel_region_mask"] for s in samples]),
            "player_region_mask": torch.stack([s["player_region_mask"] for s in samples]),
            "player_identity_mask": torch.stack([s["player_identity_mask"] for s in samples]),
            "player_identity_frame": torch.stack([s["player_identity_frame"] for s in samples]),
            "player_identity_slot": torch.stack([s["player_identity_slot"] for s in samples]),
            "player_identity_valid": torch.stack([s["player_identity_valid"] for s in samples])}
