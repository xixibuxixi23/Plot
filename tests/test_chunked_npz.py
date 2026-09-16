import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from plot.data.chunked_npz import (
    ChunkedNpzFile,
    open_npz,
    validate_chunked_npz,
    write_chunked_npz,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _make_archive(path, frames=19):
    rng = np.random.default_rng(7)
    values = {
        "instance_mask": rng.integers(0, 50, (frames, 3, 11, 13), dtype=np.uint16),
        "obs_voxel_mt": rng.integers(-1, 80, (frames, 3, 5, 4, 3, 2), dtype=np.int16),
        "cam_pos": rng.normal(size=(frames, 3, 3)).astype(np.float32),
        "entity_id": np.asarray(["agent0", "agent1", "agent2"]),
    }
    np.savez_compressed(path, **values)
    return values


def test_chunked_npz_exact_slices_and_immutable_source(tmp_path):
    source = tmp_path / "data.npz"
    expected = _make_archive(source)
    before = _sha256(source)
    for chunk_frames in (8, 16):
        cache = tmp_path / f"data.m3c{chunk_frames}.npz"
        write_chunked_npz(source, cache, chunk_frames=chunk_frames)
        assert _sha256(source) == before
        assert validate_chunked_npz(cache, expected_chunk_frames=chunk_frames)
        with open_npz(cache) as data:
            assert isinstance(data, ChunkedNpzFile)
            assert data.chunk_frames == chunk_frames
            np.testing.assert_array_equal(data["instance_mask"][3:18, 1],
                                          expected["instance_mask"][3:18, 1])
            np.testing.assert_array_equal(data["obs_voxel_mt"][7:19],
                                          expected["obs_voxel_mt"][7:19])
            np.testing.assert_array_equal(data["obs_voxel_mt"][-1],
                                          expected["obs_voxel_mt"][-1])
            np.testing.assert_array_equal(data["cam_pos"], expected["cam_pos"])
            np.testing.assert_array_equal(data["entity_id"], expected["entity_id"])
    with open_npz(source) as ordinary:
        np.testing.assert_array_equal(ordinary["instance_mask"], expected["instance_mask"])


def test_materializer_emits_relative_portable_index_and_resumes(tmp_path):
    source_root = tmp_path / "raw"
    episode = source_root / "train" / "episode_0001"
    episode.mkdir(parents=True)
    _make_archive(episode / "data.npz")
    index = tmp_path / "train_c65.pt"
    torch.save({
        "context_frames": 65,
        "split": "train",
        "item_vocabulary": {"": 0},
        "episodes": [{
            "path": str(episode.resolve()),
            "manifest": {"training_data_file": "data.npz", "num_agents": 3},
        }],
        "windows": [(0, 0, 0)],
    }, index)
    cache_root = tmp_path / "cache"
    command = [
        sys.executable,
        "scripts/materialize_m3_chunk_cache.py",
        "--window-index", str(index),
        "--source-root", str(source_root),
        "--output-root", str(cache_root),
        "--chunk-frames", "8",
        "--workers", "2",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    subprocess.run(command, check=True, capture_output=True, text=True)
    portable = torch.load(cache_root / "train_c65_chunk8.pt",
                          map_location="cpu", weights_only=False)
    row = portable["episodes"][0]
    assert row["path"] == "train/episode_0001"
    assert row["manifest"]["m3_chunk_cache_file"] == (
        "train/episode_0001/data.m3c8.npz"
    )
    assert validate_chunked_npz(
        cache_root / row["manifest"]["m3_chunk_cache_file"],
        expected_chunk_frames=8,
    )
