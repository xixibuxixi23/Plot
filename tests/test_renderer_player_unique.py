import os
from pathlib import Path
import subprocess

import pytest
import torch

from plot.training.renderer_diagnostics import pack_player_diagnostics, summarize_player_diagnostics
from plot.training.renderer_trainer import select_renderer_pixel_frames, renderer_pixel_losses, renderer_training_losses
from tests.test_renderer_player_pixels_only import ToyRenderer, FrozenCodec


def select(mask, count=2, seed=19):
    return select_renderer_pixel_frames(
        torch.ones_like(mask), frames_per_sample=count, player_region_mask=mask,
        selection_mode="player_unique", generator=torch.Generator().manual_seed(seed),
    )[0]


def test_unique_player_frames_are_visible_distinct_and_exclude_prefix():
    mask = torch.zeros(64, 9, 1, 8, 8)
    mask[:, 0] = 1
    mask[:, 2, :, :2] = 1
    mask[:, 7] = 1
    indices = select(mask)
    assert torch.equal(indices.sort(1).values, torch.tensor([[2, 7]]).expand(64, -1))
    assert torch.equal(indices, select(mask))


def test_one_visible_frame_is_not_duplicated_and_empty_clips_do_not_crash():
    mask = torch.zeros(64, 9, 1, 8, 8)
    mask[:, 5] = 1
    indices = select(mask)
    assert (indices[:, 0] == 5).all()
    assert (indices[:, 1] != 5).all()
    assert ((indices > 0) & (indices < 9)).all()
    empty = select(torch.zeros_like(mask), count=8)
    assert torch.equal(empty.sort(1).values, torch.arange(1, 9).expand(64, -1))
    with pytest.raises(ValueError, match="requires player_region_mask"):
        select_renderer_pixel_frames(mask, frames_per_sample=2, selection_mode="player_unique")


def test_pixel_noise_bins_follow_selected_frame_times_and_ignore_empty_masks():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent

    latent = torch.zeros(1, 9, 3, 8, 8)
    latent[:, 1] = .2
    latent[:, 3] = .8
    latent.requires_grad_()
    mask = torch.zeros(1, 9, 1, 8, 8)
    mask[:, [1, 3]] = 1
    times = torch.full((1, 8), .5)
    times[:, 0], times[:, 2] = .1, .9
    metrics = {}
    losses = renderer_pixel_losses(
        Codec(), latent, torch.zeros_like(latent), mask,
        player_region_mask=mask, frames_per_sample=2, frame_selection="player_unique",
        diagnostics=metrics, noise_time=times, generator=torch.Generator().manual_seed(6),
    )
    summary = summarize_player_diagnostics(pack_player_diagnostics(metrics, device="cpu"))
    assert summary["player_pixel_low_mean"] == pytest.approx(.2)
    assert summary["player_pixel_high_mean"] == pytest.approx(.8)
    assert summary["player_pixel_mid_count"] == 0
    assert "player_pixel_mid_mean" not in summary
    assert summary["player_pixel_supervised_fraction"] == 1
    assert not metrics["player_pixel_high_sum"].requires_grad
    losses["player_pixel_l1"].backward()
    assert latent.grad[:, 1].abs().sum() > 0
    assert latent.grad[:, 0].abs().sum() == 0


def test_noise_statistics_merge_sums_and_counts_not_rank_means():
    a = pack_player_diagnostics({"player_flow_low_sum": 2., "player_flow_low_count": 1,
                                "player_pixel_valid_count": 1, "player_pixel_selected_count": 2}, device="cpu")
    b = pack_player_diagnostics({"player_flow_low_sum": 18., "player_flow_low_count": 3,
                                "player_pixel_valid_count": 2, "player_pixel_selected_count": 2}, device="cpu")
    summary = summarize_player_diagnostics(a + b)
    assert summary["player_flow_low_mean"] == 5
    assert summary["player_flow_low_count"] == 4
    assert "player_flow_high_mean" not in summary
    assert summary["player_pixel_supervised_fraction"] == .75


def test_combined_diagnostics_do_not_change_loss_or_gradient():
    torch.manual_seed(73)
    model, codec = ToyRenderer(), FrozenCodec()
    clean, rgb = torch.randn(2, 9, 3, 8, 8), torch.rand(2, 9, 3, 8, 8)
    mask = torch.ones(2, 9, 1, 8, 8)
    results, gradients = [], []
    diagnostics = {}
    for collector in [None, diagnostics]:
        losses = renderer_training_losses(
            model, codec, clean, {}, rgb, mask, player_region_mask=mask,
            frames_per_sample=2, pixel_frame_selection="player_unique",
            flow_loss_weight=1., player_flow_loss_weight=1.,
            player_pixel_l1_weight=1., player_pixel_edge_weight=.25,
            entity_pixel_l1_weight=0., entity_pixel_edge_weight=0., health_pixel_l1_weight=0.,
            flow_diagnostics=collector, generator=torch.Generator().manual_seed(17),
        )
        expected = losses["flow_loss"] + losses["player_flow_loss"] + losses["player_pixel_l1"] + .25*losses["player_pixel_edge"]
        torch.testing.assert_close(losses["total_loss"], expected)
        results.append(losses["total_loss"].detach())
        gradients.append(torch.autograd.grad(losses["total_loss"], model.weight)[0])
    torch.testing.assert_close(*results)
    torch.testing.assert_close(*gradients)
    assert sum(int(diagnostics[f"player_flow_{k}_count"]) for k in ["low", "mid", "high"]) == 16
    assert sum(int(diagnostics[f"player_pixel_{k}_count"]) for k in ["low", "mid", "high"]) == 4


def test_rgb10x_recipe_resolves_to_combined_unique_loss():
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PLOT_DATASET_ROOT="/data", INIT_CHECKPOINT="/checkpoint.pt",
               FINAL_STEP="7565", OUTPUT_DIR="/output", CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
               PYTHON_BIN="/bin/echo", EVALUATE_BASELINE="0", BATCH_SIZE="4")
    result = subprocess.run(["bash", str(repo / "train_scripts/recipes/m3/train_m3_simple_fulltrain_player_rgb10x.sh")],
                            env=env, text=True, capture_output=True, check=True)
    tokens = result.stdout.split()
    def last_value(key):
        return tokens[max(i for i, v in enumerate(tokens) if v == key) + 1]
    assert last_value("--flow-loss-weight") == "1"
    assert last_value("--player-flow-loss-weight") == "1"
    assert last_value("--player-pixel-l1-weight") == "1"
    assert last_value("--player-pixel-edge-weight") == "0.25"
    assert last_value("--pixel-frame-selection") == "player_unique"
    assert last_value("--batch-size") == "4"
    assert "--nproc-per-node=8" in tokens
    assert "--log-player-noise-bins" in tokens
