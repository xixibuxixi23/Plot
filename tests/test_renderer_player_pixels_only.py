"""Turning off flow must leave a genuine decoded-player-only gradient."""
from types import SimpleNamespace

import pytest
import torch

from plot.training.renderer_trainer import renderer_training_losses


class ToyRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.3))
        self.cfg = SimpleNamespace(context_frames=9, block_frames=8)
        self.core = SimpleNamespace(kv_caches=None, use_condition_mask=False)
        self.calls = 0

    def forward(self, noisy, time, conditions):
        self.calls += 1
        return noisy * self.weight


class FrozenCodec(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
        self.frames = None

    def decode_for_loss(self, latent, chunk_size=1):
        self.frames = latent.shape[:2]
        return (self.scale * latent).sigmoid()


@pytest.mark.parametrize("empty_player_mask", [False, True])
def test_pixel_only_gradient_has_no_flow_contribution(empty_player_mask):
    torch.manual_seed(73)
    model, codec = ToyRenderer(), FrozenCodec()
    clean = torch.randn(2, 9, 3, 8, 8)
    rgb = torch.rand_like(clean)
    entity_mask = torch.ones(2, 9, 1, 8, 8)
    player_mask = torch.zeros_like(entity_mask)
    player_mask[:, 0] = 1  # Prefix must never supply supervision.
    if not empty_player_mask:
        player_mask[:, 1:, :, 2:6, 2:6] = 1
    losses = renderer_training_losses(
        model, codec, clean, {}, rgb, entity_mask,
        player_region_mask=player_mask, frames_per_sample=2,
        pixel_frame_selection="player", flow_loss_weight=0.,
        player_flow_loss_weight=0., entity_pixel_l1_weight=0.,
        entity_pixel_edge_weight=0., health_pixel_l1_weight=0.,
        player_pixel_l1_weight=.1, player_pixel_edge_weight=.025,
        generator=torch.Generator().manual_seed(17),
    )
    expected = .1 * losses["player_pixel_l1"] + .025 * losses["player_pixel_edge"]
    torch.testing.assert_close(losses["total_loss"], expected)
    assert model.calls == 1
    assert codec.frames == (4, 1)
    assert losses["flow_loss"].item() > 0  # Diagnostic, not optimized.
    assert losses["entity_pixel_l1"].item() > 0
    actual_grad, = torch.autograd.grad(losses["total_loss"], model.weight, retain_graph=True)
    expected_grad, = torch.autograd.grad(expected, model.weight)
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad)
    if empty_player_mask:
        assert losses["total_loss"].item() == 0
        assert actual_grad.item() == 0
    else:
        assert actual_grad.abs().item() > 0
    assert codec.scale.grad is None
