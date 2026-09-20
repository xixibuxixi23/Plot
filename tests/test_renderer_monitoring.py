import json
from pathlib import Path

import cv2
import torch

from plot.checkpoint_io import staged_torch_save
from plot.training.renderer_monitoring import (
    render_probe,
    select_renderer_probes,
    write_comparison_video,
)


def test_staged_checkpoint_copy_and_fuse_rename_fallback(tmp_path, monkeypatch):
    staging = tmp_path / "pfs"
    output = tmp_path / "oss"
    destination = output / "model.pt"
    original_replace = __import__("os").replace

    def reject_fuse_replace(source, target):
        if Path(target) == destination:
            raise OSError(16, "Device or resource busy")
        return original_replace(source, target)

    monkeypatch.setattr("plot.checkpoint_io.os.replace", reject_fuse_replace)
    staged_torch_save({"weight": torch.arange(32)}, destination, staging_dir=staging)
    result = torch.load(destination, map_location="cpu", weights_only=True)
    torch.testing.assert_close(result["weight"], torch.arange(32))
    assert not destination.with_name("model.pt.uploading").exists()


def test_probe_selection_requires_event_in_first_predicted_chunk(tmp_path):
    episodes = []
    windows = []
    for episode_id, (scenario, event_frame) in enumerate((("S01", 19), ("S08", 8))):
        path = tmp_path / scenario
        path.mkdir()
        event = "block_placed" if scenario == "S01" else "damage"
        (path / "events.jsonl").write_text(
            json.dumps(
                {
                    "event": event,
                    "observation_frame": event_frame,
                    "source": "agent0",
                    "target": "agent1",
                }
            )
            + "\n"
        )
        episodes.append((path, {"scenario_id": scenario}))
        windows.extend(((episode_id, 0, 0), (episode_id, 16, 0)))

    dataset = type("Dataset", (), {"episodes": episodes, "index": windows})()
    probes = select_renderer_probes(
        dataset,
        specs=(
            ("build", "S01", {"block_placed"}),
            ("combat", "S08", {"damage"}),
        ),
    )
    assert [(probe.name, probe.start, probe.event) for probe in probes] == [
        ("build", 16, "block_placed"),
        ("combat", 0, "damage"),
    ]


def test_comparison_video_is_written(tmp_path):
    truth = torch.zeros(2, 3, 16, 24)
    prediction = torch.ones_like(truth) * 0.25
    weight = torch.ones(2, 1, 2, 3)
    weight[:, :, 0, 0] = 5
    path = write_comparison_video(tmp_path / "probe.mp4", truth, prediction, weight)
    assert path.stat().st_size > 0
    capture = cv2.VideoCapture(str(path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 48
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 16
        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        assert codec.lower() in {"avc1", "h264"}
    finally:
        capture.release()


def test_probe_uses_real_cached_64_frame_rollout(tmp_path):
    from tests.test_renderer import conditions, tiny_model

    class FakeCodec:
        @staticmethod
        def encode(rgb):
            return torch.cat((rgb, torch.zeros(*rgb.shape[:2], 13, *rgb.shape[-2:])), dim=2)

        @staticmethod
        def decode(latent):
            return latent[:, :, :3].clamp(0, 1)

    sample = {
        "rgb": torch.rand(1, 65, 3, 4, 4),
        "conditions": conditions(65),
        "region_weight": torch.ones(1, 65, 1, 4, 4),
        "player_region_mask": torch.ones(1, 65, 1, 4, 4, dtype=torch.bool),
    }
    model = tiny_model().eval()
    metrics = render_probe(
        model, FakeCodec(), sample, tmp_path / "rollout.mp4", seed=7, denoising_steps=1
    )
    assert set(metrics) == {
        "l1", "psnr", "entity_l1", "health_l1", "player_l1",
        "player_detail_ratio", "player_pixels",
    }
    assert (tmp_path / "rollout.mp4").stat().st_size > 0
    assert model.core.kv_caches is None
