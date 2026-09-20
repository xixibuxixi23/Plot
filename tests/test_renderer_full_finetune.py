import copy

import pytest
import torch

from plot.training.renderer_optimizer import AppearanceAdamW, build_full_player_optimizer
from plot.training.renderer_monitoring import render_probe
from tests.test_renderer import conditions, tiny_model


def test_channel_learning_rates_match_independent_adamw_and_resume():
    torch.manual_seed(17)
    weight = torch.nn.Parameter(torch.randn(3, 7, 2, 2, dtype=torch.float64))
    base = torch.nn.Parameter(weight[:, :-2].detach().clone())
    tail = torch.nn.Parameter(weight[:, -2:].detach().clone())
    optimizer = AppearanceAdamW([{
        "params": [weight], "lr": 1e-3, "appearance_lr": 5e-3,
        "appearance_input_channels": 2,
    }], weight_decay=0.1)
    expected = torch.optim.AdamW([
        {"params": [base], "lr": 1e-3}, {"params": [tail], "lr": 5e-3},
    ], weight_decay=0.1)
    for step in range(4):
        gradient = torch.randn_like(weight)
        weight.grad = gradient
        base.grad, tail.grad = gradient[:, :-2].clone(), gradient[:, -2:].clone()
        optimizer.step()
        expected.step()
        torch.testing.assert_close(weight[:, :-2], base, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(weight[:, -2:], tail, rtol=1e-10, atol=1e-10)
        if step == 1:
            state = copy.deepcopy(optimizer.state_dict())
            optimizer = AppearanceAdamW([{"params": [weight]}])
            optimizer.load_state_dict(state)


def test_full_optimizer_covers_all_parameters_without_freezing():
    model = tiny_model(simple_conditioning=True)
    optimizer = build_full_player_optimizer(model, base_lr=1e-5, appearance_lr=5e-5)
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in parameters}) == len(parameters)
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters()}
    before = model.core.blocks[0].s_attn.to_qkv.weight.detach().clone()
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    assert not torch.equal(before, model.core.blocks[0].s_attn.to_qkv.weight)
    next(model.parameters()).requires_grad_(False)
    with pytest.raises(ValueError, match="all M3 parameters trainable"):
        build_full_player_optimizer(model, base_lr=1e-5, appearance_lr=5e-5)


def test_short_rollout_encodes_only_observed_prefix(tmp_path):
    class Codec:
        def encode(self, rgb):
            assert rgb.shape[1] == 1
            return torch.cat((rgb, torch.zeros(*rgb.shape[:2], 13, *rgb.shape[-2:])), dim=2)

        def decode(self, latent):
            assert latent.shape[1] == 8
            return latent[:, :, :3].clamp(0, 1)

    model = tiny_model(simple_conditioning=True).eval()
    sample = {"rgb": torch.rand(1, 9, 3, 4, 4), "conditions": conditions(9),
              "region_weight": torch.ones(1, 9, 1, 4, 4),
              "player_region_mask": torch.ones(1, 9, 1, 4, 4)}
    metrics = render_probe(model, Codec(), sample, tmp_path / "short.mp4",
                           seed=19, denoising_steps=2, precision="fp32")
    assert metrics["player_pixels"] == 8 * 4 * 4
    assert torch.isfinite(torch.tensor(metrics["player_l1"]))
    assert model.core.kv_caches is None
