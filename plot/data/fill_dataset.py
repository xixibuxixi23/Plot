from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from plot.geometry import TILE_SIZE, coverage_mask, crop_raw_49_to_tile_48


DEFAULT_VERTICAL_FOV_RADIANS = np.deg2rad(72.0)


def _default_fov(manifest: dict, agents: int) -> tuple[np.ndarray, np.ndarray]:
    height = float(manifest.get("height", 720))
    width = float(manifest.get("width", 1280))
    fov_y = np.full(agents, DEFAULT_VERTICAL_FOV_RADIANS, dtype=np.float32)
    fov_x_value = 2.0 * np.arctan((width / height) * np.tan(DEFAULT_VERTICAL_FOV_RADIANS / 2.0))
    return np.full(agents, fov_x_value, dtype=np.float32), fov_y


@dataclass(frozen=True)
class BlockVocabulary:
    class_to_raw: tuple[int, ...]

    @classmethod
    def load(cls, path: str | Path) -> "BlockVocabulary":
        value = json.loads(Path(path).read_text())
        return cls(tuple(int(v) for v in value["class_to_raw"]))

    @property
    def size(self) -> int:
        return len(self.class_to_raw)

    def encode(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys = np.asarray(self.class_to_raw, dtype=np.int64)
        order = np.argsort(keys)
        sorted_keys = keys[order]
        flat = np.asarray(raw, dtype=np.int64).reshape(-1)
        positions = np.searchsorted(sorted_keys, flat)
        clipped = np.minimum(positions, max(0, len(sorted_keys) - 1))
        valid = (positions < len(sorted_keys)) & (sorted_keys[clipped] == flat)
        encoded = np.zeros_like(flat, dtype=np.int64)
        encoded[valid] = order[clipped[valid]]
        return encoded.reshape(raw.shape), valid.reshape(raw.shape)

    def decode(self, classes: np.ndarray) -> np.ndarray:
        table = np.asarray(self.class_to_raw, dtype=np.int32)
        return table[np.asarray(classes)]


def _read_video_frame(path: Path, index: int, image_size: tuple[int, int]) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise RuntimeError(f"could not decode frame {index} from {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, image_size[::-1], interpolation=cv2.INTER_AREA)
    return np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)


def _read_video_frames(path: Path, indices, image_size: tuple[int, int]) -> dict[int, np.ndarray]:
    """Decode ordered sparse frames with one video open and forward scan."""
    wanted = sorted(set(int(index) for index in indices))
    if not wanted:
        return {}
    capture = cv2.VideoCapture(str(path))
    frames = {}
    try:
        # Seeking once to the first requested frame avoids decoding the whole
        # episode for late contiguous M3/M4 windows.
        cursor = wanted[0]
        capture.set(cv2.CAP_PROP_POS_FRAMES, cursor)
        for target in wanted:
            while cursor <= target:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"could not decode frame {target} from {path}")
                if cursor == target:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, image_size[::-1], interpolation=cv2.INTER_AREA)
                    frames[target] = np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)
                cursor += 1
    finally:
        capture.release()
    return frames


def _read_cached_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"could not decode cached image {path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, image_size[::-1], interpolation=cv2.INTER_AREA)
    return np.moveaxis(frame.astype(np.float32) / 255.0, -1, 0)


def _shift_xy_without_wrap(value: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    """Translate the first two axes, padding instead of wrapping boundary cells."""
    result = np.zeros_like(value)
    source_x = slice(max(0, -shift_x), min(value.shape[0], value.shape[0] - shift_x))
    source_y = slice(max(0, -shift_y), min(value.shape[1], value.shape[1] - shift_y))
    target_x = slice(max(0, shift_x), min(value.shape[0], value.shape[0] + shift_x))
    target_y = slice(max(0, shift_y), min(value.shape[1], value.shape[1] + shift_y))
    result[target_x, target_y] = value[source_x, source_y]
    return result


def _canonical_rotate_tile(value: np.ndarray, quarter_turns: int) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a lower-heavy even tile while keeping player cell index 24 fixed."""
    rotated = np.rot90(value, k=-quarter_turns, axes=(0, 1)).copy()
    marker = np.zeros(value.shape[:2], dtype=bool)
    marker[TILE_SIZE // 2, TILE_SIZE // 2] = True
    marker = np.rot90(marker, k=-quarter_turns, axes=(0, 1))
    location = np.argwhere(marker)[0]
    shift = np.asarray((TILE_SIZE // 2, TILE_SIZE // 2)) - location
    return _shift_xy_without_wrap(rotated, int(shift[0]), int(shift[1])), shift


def select_frontier_observation(
    centers: np.ndarray,
    start: int,
    target_agent: int,
    sample_slot: int,
    samples_per_agent: int,
) -> int:
    """Select a state whose current tile contains voxels unseen by earlier tiles."""
    if sample_slot <= 0:
        return int(start)
    seen = {tuple(int(v) for v in row) for row in centers[start].reshape(-1, 3)}
    candidates: list[int] = []
    for observation in range(start + 1, len(centers)):
        query = tuple(int(v) for v in centers[observation, target_agent])
        if query not in seen:
            previous = np.asarray(sorted(seen), dtype=np.int64)
            if not coverage_mask(np.asarray(query), previous).all():
                candidates.append(observation)
                if samples_per_agent == 2:
                    return observation
        seen.update(tuple(int(v) for v in row) for row in centers[observation].reshape(-1, 3))
    if not candidates:
        return int(start)
    fraction = (sample_slot - 1) / max(1, samples_per_agent - 2)
    return candidates[round(fraction * (len(candidates) - 1))]


def use_image_condition(
    sample_slot: int,
    samples_per_agent: int,
    frontier_image_probability: float,
) -> bool:
    """Deterministically allocate optional-image frontier samples."""
    if sample_slot == 0:
        return True
    frontier_count = max(1, samples_per_agent - 1)
    image_count = round(frontier_count * frontier_image_probability)
    return sample_slot > frontier_count - image_count


class TextAgentFillDataset(Dataset):
    """Derive regular M1 tiles from continuous TextAgent episodes.

    Sample zero for every resident is an all-UNFILLED initialization tile. Extra
    samples approximate discovery by taking a later tile and marking the union
    of earlier resident windows as known. Ground-truth centers are adapter-only
    metadata and are never returned as neural conditioning.
    """

    def __init__(
        self,
        root: str | Path,
        vocabulary: str | Path | BlockVocabulary,
        *,
        split: str | None = "train",
        samples_per_agent: int = 2,
        image_size: tuple[int, int] = (180, 320),
        max_agents: int = 4,
        num_views: int | None = None,
        initial_only: bool = False,
        episode_index: str | Path | None = None,
        canonical_yaw: bool = False,
        frontier_sampling: bool = False,
        frontier_image_probability: float = 1.0,
    ):
        self.root = Path(root)
        self.vocabulary = (
            vocabulary if isinstance(vocabulary, BlockVocabulary) else BlockVocabulary.load(vocabulary)
        )
        self.samples_per_agent = int(samples_per_agent)
        self.image_size = image_size
        self.max_agents = int(max_agents)
        self.num_views = int(num_views or max_agents)
        self.initial_only = bool(initial_only)
        self.canonical_yaw = bool(canonical_yaw)
        self.frontier_sampling = bool(frontier_sampling)
        self.frontier_image_probability = float(frontier_image_probability)
        if not 0.0 <= self.frontier_image_probability <= 1.0:
            raise ValueError("frontier_image_probability must be in [0, 1]")
        self.episode_index = Path(episode_index) if episode_index is not None else None
        if self.num_views > self.max_agents:
            raise ValueError("num_views cannot exceed max_agents")
        self.episodes: list[tuple[Path, dict]] = []
        if self.episode_index is not None:
            episode_records = self._indexed_episodes(split)
        else:
            episode_records = (
                (manifest_path.parent, json.loads(manifest_path.read_text()))
                for manifest_path in self._manifest_paths(split)
            )
        for episode_path, manifest in episode_records:
            if split is not None and manifest.get("split") != split:
                continue
            if int(manifest.get("num_agents", 0)) < self.num_views:
                continue
            validation_path = episode_path / "validation.json"
            if self.episode_index is None and validation_path.exists():
                if not json.loads(validation_path.read_text()).get("usable"):
                    continue
            data_path = episode_path / manifest.get("training_data_file", "data.npz")
            if data_path.exists():
                self.episodes.append((episode_path, manifest))
        if not self.episodes:
            raise ValueError(f"no usable episodes found below {self.root}")
        self.index = []
        for episode_index, (_, manifest) in enumerate(self.episodes):
            for agent in range(int(manifest["num_agents"])):
                for sample_slot in range(1 if self.initial_only else self.samples_per_agent):
                    self.index.append((episode_index, agent, sample_slot))

    def _manifest_paths(self, split: str | None) -> list[Path]:
        """Enumerate accepted episodes without walking large rejected trees."""
        ledgers = sorted(self.root.glob("plan_results_queue_*.jsonl"))
        if not ledgers:
            return sorted(self.root.rglob("manifest.json"))
        root = self.root.resolve()
        manifests: set[Path] = set()
        for ledger in ledgers:
            with ledger.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if not row.get("success") or not row.get("validation", {}).get("usable"):
                        continue
                    if split is not None and row.get("split") != split:
                        continue
                    episode = Path(row["output_dir"]).resolve()
                    if root not in episode.parents:
                        raise ValueError(f"ledger output is outside dataset root: {episode}")
                    manifest = episode / "manifest.json"
                    if manifest.exists():
                        manifests.add(manifest)
        return sorted(manifests)

    def _indexed_episodes(self, split: str | None) -> list[tuple[Path, dict]]:
        payload = json.loads(self.episode_index.read_text())
        indexed_root = Path(payload["dataset_root"]).resolve()
        if indexed_root != self.root.resolve():
            raise ValueError(f"episode index belongs to {indexed_root}, not {self.root.resolve()}")
        splits = payload["splits"]
        records = (
            splits.get(split, []) if split is not None
            else [record for values in splits.values() for record in values]
        )
        return [(self.root / record["path"], record["manifest"]) for record in records]

    def __len__(self) -> int:
        return len(self.index)

    @staticmethod
    def _model_start(data: np.lib.npyio.NpzFile, manifest: dict) -> int:
        explicit = manifest.get("model_start_observation")
        if explicit is not None:
            return int(explicit)
        sources = np.asarray(data["action_source"]).astype(str)
        excluded = np.zeros(sources.shape, dtype=bool)
        for prefix in ("environment_", "teacher_calibrate_"):
            excluded |= np.char.startswith(sources, prefix)
        rows = np.flatnonzero(np.all(~excluded, axis=1))
        return int(rows[0]) if len(rows) else 0

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        episode_index, target_agent, sample_slot = self.index[item]
        path, manifest = self.episodes[episode_index]
        cache_path = path / "m1_initial.npz"
        use_cache = self.initial_only and sample_slot == 0 and cache_path.exists()
        source_path = cache_path if use_cache else path / manifest.get("training_data_file", "data.npz")
        with np.load(source_path, allow_pickle=False) as data:
            if use_cache:
                centers = np.asarray(data["voxel_center"], dtype=np.int64)[None]
                observation = 0
                previous = np.empty((0, 3), dtype=np.int64)
                raw = np.asarray(data["voxel_tiles"][target_agent])
                cam_pos = np.asarray(data["cam_pos"], dtype=np.float32)
                cam_dir = np.asarray(data["cam_dir"], dtype=np.float32)
                if "fov_x" in data and "fov_y" in data:
                    fov_x = np.asarray(data["fov_x"], dtype=np.float32)
                    fov_y = np.asarray(data["fov_y"], dtype=np.float32)
                else:
                    fov_x, fov_y = _default_fov(manifest, len(cam_pos))
            else:
                centers = np.asarray(data["obs_voxel_center"], dtype=np.int64)
                start = min(self._model_start(data, manifest), len(centers) - 1)
                if sample_slot == 0:
                    observation = start
                    previous = np.empty((0, 3), dtype=np.int64)
                else:
                    if self.frontier_sampling:
                        observation = select_frontier_observation(
                            centers, start, target_agent, sample_slot, self.samples_per_agent
                        )
                    else:
                        fraction = sample_slot / max(1, self.samples_per_agent - 1)
                        observation = start + int(fraction * max(0, len(centers) - 1 - start))
                    previous = centers[start:observation].reshape(-1, 3)
                raw = np.asarray(data["obs_voxel_mt"][observation, target_agent])
                cam_pos = np.asarray(data["cam_pos"][observation], dtype=np.float32)
                cam_dir = np.asarray(data["cam_dir"][observation], dtype=np.float32)
                fov_x = np.asarray(data["fov_x"][observation], dtype=np.float32)
                fov_y = np.asarray(data["fov_y"][observation], dtype=np.float32)
            tile = raw if use_cache else crop_raw_49_to_tile_48(raw)
            target, target_valid = self.vocabulary.encode(tile[..., 0])
            known = coverage_mask(centers[observation, target_agent], previous)
            fill = ~known & target_valid
            context = np.where(known & target_valid, target, 0)

            # The target resident is view zero. Camera targets use the same
            # gauge as PERSIST: tile center - 0.5, normalized by 48.
            available = int(manifest["num_agents"])
            order = [target_agent] + [i for i in range(available) if i != target_agent]
            order = order[:self.num_views]
            reference = centers[observation, target_agent].astype(np.float32) - 0.5
            camera_position = (
                cam_pos[order] - reference
            ) / float(TILE_SIZE)
            camera_direction = cam_dir[order]
            norms = np.linalg.norm(camera_direction, axis=-1, keepdims=True)
            camera_valid = np.isfinite(camera_position).all(-1) & np.isfinite(camera_direction).all(-1)
            camera_valid &= norms[:, 0] > 1e-6
            camera_direction = camera_direction / np.clip(norms, 1e-6, None)

            if self.canonical_yaw:
                yaw = np.arctan2(camera_direction[0, 1], camera_direction[0, 0])
                quarter_turns = int(np.rint(yaw / (0.5 * np.pi)))
                angle = -quarter_turns * 0.5 * np.pi
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.asarray(
                    ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                    dtype=np.float32,
                )
                camera_position = camera_position @ rotation.T
                camera_direction = camera_direction @ rotation.T
                target, shift = _canonical_rotate_tile(target, quarter_turns)
                target_valid, _ = _canonical_rotate_tile(target_valid, quarter_turns)
                known, _ = _canonical_rotate_tile(known, quarter_turns)
                fill, _ = _canonical_rotate_tile(fill, quarter_turns)
                context, _ = _canonical_rotate_tile(context, quarter_turns)
                camera_position[:, :2] += shift[None] / float(TILE_SIZE)
                fill &= target_valid

        image_condition = use_image_condition(
            sample_slot, self.samples_per_agent, self.frontier_image_probability
        )
        videos = manifest.get("agent_video_files") or [
            f"rgb_agent{i}.mp4" for i in range(int(manifest["num_agents"]))
        ]
        order = order[:len(videos)]
        images = np.zeros((self.num_views, 3, *self.image_size), dtype=np.float32)
        if image_condition:
            if use_cache:
                decoded = np.stack([
                    _read_cached_image(path / f"m1_rgb_agent{i}.jpg", self.image_size)
                    for i in order
                ])
            else:
                decoded = np.stack([
                    _read_video_frame(path / videos[i], observation, self.image_size)
                    for i in order
                ])
            images[:len(decoded)] = decoded
        agent_mask = np.zeros(self.num_views, dtype=bool)
        agent_mask[:len(order)] = True
        return {
            "voxel_context": torch.from_numpy(context).long(),
            "known_mask": torch.from_numpy(known),
            "fill_mask": torch.from_numpy(fill),
            "target": torch.from_numpy(target).long(),
            "target_valid": torch.from_numpy(target_valid),
            "images": torch.from_numpy(images),
            "agent_mask": torch.from_numpy(agent_mask),
            "image_condition_mask": torch.tensor(image_condition, dtype=torch.bool),
            "camera_position": torch.from_numpy(camera_position),
            "camera_direction": torch.from_numpy(camera_direction),
            "camera_valid": torch.from_numpy(camera_valid),
            "fov_x": torch.from_numpy(fov_x[order]),
            "fov_y": torch.from_numpy(fov_y[order]),
        }
