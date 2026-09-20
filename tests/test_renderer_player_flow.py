from types import SimpleNamespace

import pytest
import torch

from plot.training.renderer_trainer import player_flow_diagnostics, renderer_training_losses


def test_single_clip_logger_ignores_framewise_diagnostic_tensors():
    from scripts.overfit_m3_single_clip import scalar_diagnostics

    result = scalar_diagnostics({
        "noise_time": torch.tensor(.5),
        "noise_time_per_frame": torch.full((1, 8), .5),
        "player_flow_high_count": torch.tensor(8),
        "player_flow_loss": torch.tensor(.2, requires_grad=True),
    })
    assert "noise_time_per_frame" not in result
    assert result["noise_time"] == .5
    assert result["player_flow_high_count"] == 8
    assert result["player_flow_loss"] == pytest.approx(.2)


def test_player_flow_is_area_normalized_excludes_prefix_and_background():
    error = torch.full((2, 8, 3, 4, 4), 7., requires_grad=True)
    mask = torch.zeros(2, 9, 1, 4, 4)
    mask[:, 0] = 1  # Observed prefix must not contribute.
    mask[0, 1:, :, :1, :1] = 1
    mask[1, 1:, :, :3, :3] = 1
    metrics = player_flow_diagnostics(error, mask, torch.full((2, 8), .8))
    torch.testing.assert_close(metrics["player_flow_loss"], torch.tensor(7.))
    assert metrics["player_flow_low_count"] == 0
    assert metrics["player_flow_high_count"] == 16
    assert metrics["player_flow_high_sum"] / 16 == 7
    metrics["player_flow_loss"].backward()
    assert error.grad[0, :, :, 1:, 1:].count_nonzero() == 0
    torch.testing.assert_close(error.grad[0].sum(), error.grad[1].sum())


def test_player_flow_preserves_subpixel_mask_and_handles_empty_sample():
    error = torch.ones(2, 8, 3, 2, 2, requires_grad=True)
    mask = torch.zeros(2, 9, 1, 16, 16)
    mask[0, 1:, :, 0, 0] = 1  # Area downsampling gives fractional coverage.
    metrics = player_flow_diagnostics(error, mask, torch.ones(2, 8))
    torch.testing.assert_close(metrics["player_flow_loss"], torch.tensor(1.))
    assert metrics["player_flow_high_count"] == 8
    empty = player_flow_diagnostics(error, torch.zeros_like(mask), torch.ones(2, 8))
    assert empty["player_flow_loss"] == 0
    empty["player_flow_loss"].backward()
    assert error.grad.count_nonzero() == 0


def test_independent_player_term_uses_same_forward_and_keeps_original_api():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.3))
            self.cfg = SimpleNamespace(context_frames=9, block_frames=8)
            self.core = SimpleNamespace(kv_caches=None, use_condition_mask=False)
            self.calls = 0

        def forward(self, noisy, time, conditions):
            self.calls += 1
            assert (time[:, 0] == 0).all()
            return noisy * self.weight

    model = Model()
    clean, rgb = torch.randn(1, 9, 3, 4, 4), torch.rand(1, 9, 3, 4, 4)
    mask = torch.ones(1, 9, 1, 4, 4)
    arguments = dict(player_region_mask=mask, entity_pixel_l1_weight=0.,
                     entity_pixel_edge_weight=0., health_pixel_l1_weight=0.)
    diagnostics = {}
    losses = renderer_training_losses(model, None, clean, {}, rgb, mask,
        player_flow_loss_weight=2., flow_diagnostics=diagnostics, **arguments)
    assert model.calls == 1
    torch.testing.assert_close(losses["total_loss"], losses["flow_loss"] * 3)
    losses["total_loss"].backward()
    assert torch.isfinite(model.weight.grad) and model.weight.grad != 0
    assert "noise_time" in diagnostics
    old = renderer_training_losses(model, None, clean, {}, rgb, mask, **arguments)
    torch.testing.assert_close(old["total_loss"], old["flow_loss"])
    assert old["player_flow_loss"] == 0
    arguments["player_region_mask"] = None
    with pytest.raises(ValueError, match="requires player_region_mask"):
        renderer_training_losses(model, None, clean, {}, rgb, mask,
                                 player_flow_loss_weight=1., **arguments)
