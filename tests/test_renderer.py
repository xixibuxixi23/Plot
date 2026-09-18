from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from plot.data.fill_dataset import BlockVocabulary
from plot.data.appearance_counterfactual_dataset import AppearanceCounterfactualRendererDataset
from plot.data.renderer_dataset import incoming_actions, merge_observation_crops, raster_camera
from plot.models.renderer import Renderer, RendererArgs
from plot.training.renderer_trainer import (
    _sample_blockwise_train_time, renderer_counterfactual_player_loss,
)
from plot.models.renderer_backbone.player_spatial_condition import ViewAwarePlayerAppearance
from plot.pipelines.renderer_pipeline import RendererMemoryBlock
from plot.training.renderer_trainer import (
    RendererRollout,
    _sample_blockwise_train_time,
    renderer_flow_loss,
    renderer_pixel_losses,
    renderer_training_losses,
    select_renderer_pixel_frames,
    slice_conditions,
)
from train_scripts.train_renderer import (
    effective_auxiliary_loss_weights,
    load_renderer_resume,
    use_counterfactual_step,
)


torch.set_num_threads(2)


def tiny_model(*, simple_conditioning=False, qk_rms_norm=False):
    torch.manual_seed(42)
    model = Renderer(RendererArgs(3, 8, input_h=4, input_w=4, hidden_size=32, depth=2,
                                 num_heads=4, voxel_channels=4, condition_dim=16, actor_channels=4,
                                 context_frames=65, cache_frames=16,
                                 qk_rms_norm=qk_rms_norm,
                                 gradient_checkpointing=False, gpu_rasterizer=False,
                                 simple_conditioning=simple_conditioning,
                                 unified_player_reference=simple_conditioning,
                                 player_reference_grid_size=(4, 2) if simple_conditioning else (8, 4),
                                 player_reference_position_encoding=simple_conditioning))
    # Zero initialization would make future-invariance tests vacuous.
    with torch.no_grad():
        model.core.final_layer.linear.weight.normal_(std=.03)
        for block in model.core.blocks:
            block.s_adaLN_modulation[-1].weight.normal_(std=.03)
            block.t_adaLN_modulation[-1].weight.normal_(std=.03)
        model.core.extra_condition_embedder.weight.normal_(std=.03)
        if model.core.actor_condition_embedder is not None:
            model.core.actor_condition_embedder.weight.normal_(std=.03)
    return model


def test_2daction_backbone_expands_joint_input_without_changing_old_path():
    source = tiny_model(qk_rms_norm=True)
    target = tiny_model(simple_conditioning=True, qk_rms_norm=True)
    source_state = {f"core.{key}": value.clone() for key, value in source.core.state_dict().items()}
    old_projection = source.core.x_embedder.proj.weight.detach().clone()
    report = target.load_2daction_backbone(source_state)

    new_projection = target.core.x_embedder.proj.weight.detach()
    old_channels = old_projection.shape[1]
    torch.testing.assert_close(new_projection[:, :old_channels], old_projection)
    torch.testing.assert_close(
        new_projection[:, old_channels:], torch.zeros_like(new_projection[:, old_channels:])
    )
    assert report["partially_loaded"] == [{
        "key": "x_embedder.proj.weight",
        "source_shape": list(old_projection.shape),
        "target_shape": list(new_projection.shape),
        "new_input_channels_initialized_to_zero": new_projection.shape[1] - old_channels,
    }]
    assert "x_embedder.proj.weight" not in report["missing"]
    assert not any(key.startswith("blocks.") for key in report["missing"])
    torch.testing.assert_close(
        target.core.blocks[0].s_attn.q_rms_norm.gamma,
        source.core.blocks[0].s_attn.q_rms_norm.gamma,
    )


def conditions(t=17):
    b, a, h, w = 1, 2, 4, 4
    return {
        "target_agent": torch.tensor([0]),
        "player_position": torch.tensor([[[[0., 0., 0.], [0., 3., 0.]]]]).expand(b, t, a, 3).clone(),
        "camera_relative": torch.tensor([0., 0., 1.5]).expand(b, t, a, 3).clone(),
        "camera_direction": torch.tensor([0., 1., 0.]).expand(b, t, a, 3).clone(),
        "fov_x": torch.full((b,t,a), 1.2), "hp": torch.full((b,t,a), 20.),
        "yaw_pitch": torch.zeros(b,t,a,2), "held_item": torch.zeros(b,t,a,dtype=torch.long),
        "resident_type": torch.zeros(b,t,a,dtype=torch.long),
        "event_cues": torch.zeros(b,t,a,4), "action": torch.zeros(b,t,a,23),
        "player_valid": torch.ones(b,t,a,dtype=torch.bool),
        "player_skin": torch.rand(b,a,4,4,16,16),
        "player_appearance_valid": torch.ones(b,a,4,dtype=torch.bool),
        "condition_mask": (torch.arange(t)[None] == 0),
        "action_prefix_mask": (torch.arange(t)[None] == 0),
        "raster_features": torch.randn(b,t,h,w,192,4),
        "raster_depth": torch.ones(b,t,h,w,192),
    }


@pytest.mark.parametrize("simple_conditioning", [False, True])
def test_attention_is_bidirectional_within_blocks_and_causal_between_blocks(simple_conditioning):
    model = tiny_model(simple_conditioning=simple_conditioning).eval()
    cond = conditions(17)
    x, time = torch.randn(1,17,16,4,4), torch.rand(1,17)
    with torch.no_grad():
        first = model(x, time, cond)
        # The second block must not affect the observed prefix or first block.
        x[:,9:] += 100
        second = model(x, time, cond)
    torch.testing.assert_close(first[:,:9], second[:,:9])
    assert not torch.allclose(first[:,9:], second[:,9:])

    within = x.clone()
    within[:,8] += 100
    with torch.no_grad():
        third = model(within, time, cond)
    torch.testing.assert_close(second[:,:1], third[:,:1])
    assert not torch.allclose(second[:,1:8], third[:,1:8])


def test_training_time_is_clean_for_prefix_and_shared_within_each_block():
    time = _sample_blockwise_train_time(
        2, 17, 8, torch.device("cpu"), torch.Generator().manual_seed(7),
    )
    torch.testing.assert_close(time[:, 0], torch.zeros(2))
    torch.testing.assert_close(time[:, 1:9], time[:, 1:2].expand(-1, 8))
    torch.testing.assert_close(time[:, 9:17], time[:, 9:10].expand(-1, 8))


def test_counterfactual_groups_can_share_random_training_times():
    time = _sample_blockwise_train_time(
        4, 17, 8, torch.device("cpu"),
        torch.Generator().manual_seed(7), counterfactual_group_size=2,
        counterfactual_pure_noise=False,
    )
    torch.testing.assert_close(time[:, 0], torch.zeros(4))
    torch.testing.assert_close(time[0], time[1])
    torch.testing.assert_close(time[2], time[3])
    assert bool(((time[:, 1:] > 0) & (time[:, 1:] < 1)).all())


@pytest.mark.parametrize("simple_conditioning", [False, True])
def test_training_backward_and_supervision_only_masks(simple_conditioning):
    model = tiny_model(simple_conditioning=simple_conditioning).train()
    cond = conditions(65)
    clean = torch.randn(1,65,16,4,4)
    loss = renderer_flow_loss(model, clean, cond, region_weight=torch.ones(1,65,1,4,4))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.resident_encoder.target_mlp[0].weight.grad.abs().sum() > 0
    assert model.core.blocks[0].t_attn.to_qkv.weight.grad is not None
    cond["instance_mask"] = torch.zeros(1,65,4,4)
    with pytest.raises(ValueError, match="not neural inputs"):
        model(clean, torch.zeros(1,17), cond)


def test_deep_condition_reinjection_is_exact_warm_start_and_trainable():
    baseline = tiny_model().eval()
    deep = Renderer(replace(
        baseline.cfg,
        deep_condition_reinjection=True,
        view_aware_appearance=True,
    )).eval()
    incompatible = deep.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith((
            "core.hud_condition_embedder.",
            "core.condition_reinjectors.",
            "core.appearance_condition_embedder.",
            "core.appearance_reinjectors.",
        ))
        for key in incompatible.missing_keys
    )
    cond = conditions(9)
    cond["hp"][:, 4:, 0] = 12
    cond["event_cues"][:, 4, 0, 3] = -8
    encoded = deep.encode_conditions(cond)
    hud = encoded["hud_condition"]
    assert hud.shape == (1, 9, 3, 4, 4)
    assert torch.count_nonzero(hud[..., :3, :]) == 0
    torch.testing.assert_close(hud[:, 4:, 1, 3, 1], torch.full((1, 5), 0.6))
    torch.testing.assert_close(hud[:, 4, 2, 3, 1], torch.tensor([-.4]))

    x, time = torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9)
    with torch.no_grad():
        expected = baseline(x, time, cond)
        actual = deep(x, time, cond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    deep.train()
    deep(x, time, cond).square().mean().backward()
    assert all(
        adapter[-1].weight.grad is not None
        and adapter[-1].weight.grad.abs().sum() > 0
        for adapter in deep.core.condition_reinjectors
    )
    assert all(
        adapter.to_output.weight.grad is not None
        and adapter.to_output.weight.grad.abs().sum() > 0
        for adapter in deep.core.appearance_reinjectors
    )


def test_detail_preserving_appearance_is_an_exact_trainable_warm_start():
    base = tiny_model()
    regular = Renderer(replace(
        base.cfg,
        view_aware_appearance=True,
    )).eval()
    regular.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        # Represent a checkpoint whose original appearance adapters have
        # already learned nonzero behavior.
        for adapter in regular.core.appearance_reinjectors:
            adapter.to_output.weight.normal_(std=.01)

    detail = Renderer(replace(
        regular.cfg,
        detail_preserving_appearance=True,
    )).eval()
    incompatible = detail.load_state_dict(regular.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith((
            "core.appearance_detail_embedder.",
            "core.appearance_detail_reinjectors.",
        ))
        for key in incompatible.missing_keys
    )

    cond = conditions(9)
    x, time = torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9)
    with torch.no_grad():
        expected = regular(x, time, cond)
        actual = detail(x, time, cond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    detail.train()
    detail(x, time, cond).square().mean().backward()
    assert all(
        adapter.to_output.weight.grad is not None
        and adapter.to_output.weight.grad.abs().sum() > 0
        for adapter in detail.core.appearance_detail_reinjectors
    )


def test_entity_reference_attention_is_an_exact_trainable_warm_start():
    base = tiny_model()
    reference = Renderer(replace(
        base.cfg,
        view_aware_appearance=True,
        entity_reference_attention=True,
    )).eval()
    incompatible = reference.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(
        key.startswith((
            "appearance_spatial_encoder.",
            "core.appearance_condition_embedder.",
            "core.appearance_reinjectors.",
            "reference_encoder.",
            "core.entity_reference_adapters.",
        ))
        for key in incompatible.missing_keys
    )
    cond = conditions(9)
    cond["player_reference"] = torch.rand(1, 2, 4, 4, 32, 16)
    x, time = torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9)
    with torch.no_grad():
        expected = base(x, time, cond)
        actual = reference(x, time, cond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    reference.train()
    reference(x, time, cond).square().mean().backward()
    assert all(
        adapter.to_output.weight.grad is not None
        and adapter.to_output.weight.grad.abs().sum() > 0
        for adapter in reference.core.entity_reference_adapters
    )


def test_unified_player_reference_is_the_only_appearance_route_and_uses_all_views():
    base = tiny_model()
    unified = Renderer(replace(
        base.cfg,
        unified_player_reference=True,
    )).eval()
    incompatible = unified.load_state_dict(base.state_dict(), strict=False)
    assert not incompatible.missing_keys or all(
        key.startswith(("reference_encoder.", "core.unified_reference_adapter."))
        for key in incompatible.missing_keys
    )
    assert all(
        key.startswith((
            "resident_encoder.skin_view_encoder.",
            "resident_encoder.appearance_direction_embedding",
            "resident_encoder.skin_view_fusion.",
        ))
        for key in incompatible.unexpected_keys
    )
    assert unified.appearance_spatial_encoder is None
    assert unified.core.entity_reference_adapters is None
    assert unified.core.unified_reference_adapter is not None
    assert not hasattr(unified.resident_encoder, "skin_view_encoder")
    assert sum(
        type(module).__name__ == "UnifiedPlayerReferenceAdapter"
        for module in unified.modules()
    ) == 1

    cond = conditions(9)
    cond["player_reference"] = torch.rand(1, 2, 4, 4, 32, 16)
    encoded = unified.encode_conditions(cond)
    assert encoded["unified_reference_tokens"].shape == (1, 2, 4, 32, 256)

    high_detail = Renderer(replace(
        base.cfg, unified_player_reference=True, player_reference_grid_size=(16, 8),
        player_reference_position_encoding=True,
    )).eval()
    high_detail_tokens = high_detail.encode_conditions(cond)["unified_reference_tokens"]
    assert high_detail_tokens.shape == (1, 2, 4, 128, 256)
    plain_high_detail = Renderer(replace(
        high_detail.cfg, player_reference_position_encoding=False,
    )).eval()
    plain_high_detail.load_state_dict(high_detail.state_dict(), strict=True)
    plain_tokens = plain_high_detail.encode_conditions(cond)["unified_reference_tokens"]
    expected_position = high_detail.reference_encoder.position_embedding
    torch.testing.assert_close(
        high_detail_tokens[0, 0, 0] - plain_tokens[0, 0, 0],
        expected_position,
    )
    assert expected_position.shape == (128, 256)
    assert not torch.allclose(expected_position[0], expected_position[-1])
    assert "reference_encoder.position_embedding" not in high_detail.state_dict()
    assert "entity_reference_view_weights" not in encoded
    assert "appearance_spatial_condition" not in encoded

    # Changing the legacy pooled skin cannot affect either state or spatial
    # conditions when the explicit reference tensor is held fixed.
    changed_skin = dict(cond, player_skin=torch.rand_like(cond["player_skin"]))
    changed_encoded = unified.encode_conditions(changed_skin)
    torch.testing.assert_close(
        encoded["extra_condition"], changed_encoded["extra_condition"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        encoded["actor_spatial_condition"],
        changed_encoded["actor_spatial_condition"],
        rtol=0,
        atol=0,
    )

    x, time = torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9)
    with torch.no_grad():
        zero_outputs = []
        for view in range(4):
            altered = cond["player_reference"].clone()
            altered[:, 1, view] += 2
            zero_outputs.append(unified(x, time, dict(cond, player_reference=altered)))
    for output in zero_outputs[1:]:
        torch.testing.assert_close(output, zero_outputs[0], rtol=0, atol=0)

    # Once the single zero-initialized output projection learns, every source
    # view remains independently available (there is no four-select-one step).
    with torch.no_grad():
        unified.core.unified_reference_adapter.to_output.weight.normal_(std=.02)
    with torch.no_grad():
        learned_outputs = []
        for view in range(4):
            altered = cond["player_reference"].clone()
            altered[:, 1, view] += 2
            learned_outputs.append(unified(x, time, dict(cond, player_reference=altered)))
    for output in learned_outputs[1:]:
        assert not torch.allclose(output, learned_outputs[0])

    unified.train()
    unified(x, time, cond).square().mean().backward()
    grad = unified.core.unified_reference_adapter.to_output.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_simple_m3_jointly_embeds_video_and_raster_with_compact_appearance_path():
    base = tiny_model()
    model = Renderer(replace(
        base.cfg,
        simple_conditioning=True,
        unified_player_reference=True,
        player_reference_grid_size=(4, 2),
    )).train()

    assert model.core.x_embedder.proj.in_channels > model.core.in_channels
    assert model.core.actor_condition_embedder is None
    assert model.core.unified_reference_adapter is None
    assert model.roi_appearance_projector is not None
    assert model.core.condition_reinjectors is None
    assert model.core.appearance_reinjectors is None
    assert model.core.entity_reference_adapters is None
    assert model.core.unified_reference_reinject_blocks == ()
    assert model.core.condition_mask_embedder is None
    assert not any("condition_mask_embedder" in name for name in model.state_dict())

    cond = conditions(9)
    encoded = model.encode_conditions(cond)
    assert encoded["actor_spatial_condition"].shape == (1, 9, 4, 4, 4)
    assert encoded["roi_appearance_condition"].shape == (1, 9, 32, 4, 4)
    assert "unified_reference_tokens" not in encoded
    assert "unified_reference_view_weights" not in encoded
    assert "unified_reference_local_coordinates" not in encoded
    assert "condition_mask" not in encoded

    # The second resident's action remains attached to its projected region.
    changed = {name: value.clone() if torch.is_tensor(value) else value
               for name, value in cond.items()}
    changed["action"][:, :, 1, 0] = 1
    changed_spatial = model.encode_conditions(changed)["actor_spatial_condition"]
    assert not torch.allclose(encoded["actor_spatial_condition"], changed_spatial)

    with torch.no_grad():
        model.core.final_layer.linear.weight.normal_(std=.03)
        for block in model.core.blocks:
            block.s_adaLN_modulation[-1].weight.normal_(std=.03)
            block.t_adaLN_modulation[-1].weight.normal_(std=.03)
    output = model(torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9), cond)
    output.square().mean().backward()
    condition_grad = model.core.x_embedder.proj.weight.grad[:, model.core.in_channels:]
    assert condition_grad.abs().sum() > 0
    assert model.resident_encoder.actor_mlp[0].weight.grad.abs().sum() > 0
    assert model.reference_encoder.encoder[0].weight.grad.abs().sum() > 0
    assert model.roi_appearance_projector.value.weight.grad.abs().sum() > 0

    with pytest.raises(ValueError, match="one joint spatial input"):
        Renderer(replace(model.cfg, unified_reference_reinject_blocks=(0,)))


def test_simple_m3_keeps_four_roi_appearance_views_in_separate_channels():
    base = tiny_model()
    model = Renderer(replace(
        base.cfg,
        simple_conditioning=True,
        unified_player_reference=True,
        player_reference_grid_size=(4, 2),
    )).eval()
    cond = conditions(3)
    with torch.no_grad():
        baseline = model.encode_conditions(cond)["roi_appearance_condition"]
        channels_per_view = baseline.shape[2] // 4
        for view in range(4):
            changed = {
                name: value.clone() if torch.is_tensor(value) else value
                for name, value in cond.items()
            }
            changed["player_skin"][:, 1, view] += 2
            projected = model.encode_conditions(changed)["roi_appearance_condition"]
            delta = (projected - baseline).abs()
            start, end = view * channels_per_view, (view + 1) * channels_per_view
            assert delta[:, :, start:end].sum() > 0
            assert delta[:, :, :start].sum() == 0
            assert delta[:, :, end:].sum() == 0


def test_geometry_aware_unified_reference_uses_view_and_local_coordinates():
    base = tiny_model()
    plain = Renderer(replace(
        base.cfg,
        unified_player_reference=True,
    )).eval()
    plain.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        plain.core.unified_reference_adapter.to_output.weight.normal_(std=.02)

    geometry = Renderer(replace(
        plain.cfg,
        geometry_aware_player_reference=True,
    )).eval()
    incompatible = geometry.load_state_dict(plain.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys == [
        "core.unified_reference_adapter.geometry_log_scale",
        "core.unified_reference_adapter.geometry_view_logit",
    ]

    cond = conditions(9)
    cond["player_reference"] = torch.rand(1, 2, 4, 4, 32, 16)
    encoded = geometry.encode_conditions(cond)
    assert encoded["unified_reference_view_weights"].shape == (1, 9, 2, 4)
    assert encoded["unified_reference_local_coordinates"].shape == (1, 9, 2, 4, 4, 2)
    assert "core.unified_reference_adapter.reference_coordinates" not in geometry.state_dict()

    x, time = torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9)
    with torch.no_grad():
        plain_output = plain(x, time, cond)
        geometry_output = geometry(x, time, cond)
    assert not torch.allclose(plain_output, geometry_output)

    geometry.train()
    geometry.zero_grad(set_to_none=True)
    geometry(x, time, cond).square().mean().backward()
    adapter = geometry.core.unified_reference_adapter
    scale_grad = adapter.geometry_log_scale.grad
    view_grad = adapter.geometry_view_logit.grad
    assert scale_grad is not None and scale_grad.abs().sum() > 0
    assert view_grad is not None and view_grad.abs().sum() > 0


def test_unified_reference_reinjection_is_shared_gated_and_exact_at_warm_start():
    base = tiny_model()
    single = Renderer(replace(base.cfg, unified_player_reference=True)).eval()
    single.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        single.core.unified_reference_adapter.to_output.weight.normal_(std=.02)
        single.core.unified_reference_adapter.to_output.bias.normal_(std=.02)

    repeated = Renderer(replace(
        base.cfg,
        unified_player_reference=True,
        unified_reference_reinject_blocks=(0, 1),
    )).eval()
    incompatible = repeated.load_state_dict(single.state_dict(), strict=False)
    assert incompatible.unexpected_keys == []
    assert incompatible.missing_keys == [
        "core.unified_reference_adapter.reinjection_gates"
    ]
    assert sum(
        type(module).__name__ == "UnifiedPlayerReferenceAdapter"
        for module in repeated.modules()
    ) == 1

    cond = conditions(9)
    x, time = torch.randn(1, 9, 16, 4, 4), torch.rand(1, 9)
    with torch.no_grad():
        baseline = single(x, time, cond)
        exact_warm_start = repeated(x, time, cond)
    torch.testing.assert_close(baseline, exact_warm_start, rtol=0, atol=0)

    with torch.no_grad():
        repeated.core.unified_reference_adapter.reinjection_gates.fill_(0.5)
        reinjected = repeated(x, time, cond)
    assert not torch.allclose(baseline, reinjected)

    repeated.train()
    repeated.zero_grad(set_to_none=True)
    with torch.no_grad():
        repeated.core.unified_reference_adapter.reinjection_gates.zero_()
    repeated(x, time, cond).square().mean().backward()
    gate_grad = repeated.core.unified_reference_adapter.reinjection_gates.grad
    assert gate_grad is not None and gate_grad.abs().sum() > 0

    with pytest.raises(ValueError, match="valid DiT block indices"):
        Renderer(replace(
            base.cfg,
            unified_player_reference=True,
            unified_reference_reinject_blocks=(2,),
        ))


def test_view_aware_appearance_preserves_reference_pixels_and_selects_back_view():
    renderer = ViewAwarePlayerAppearance(height=8, width=8)
    b, t, a = 1, 1, 2
    skins = torch.zeros(b, a, 4, 4, 16, 16)
    colors = torch.tensor([
        [1., 0., 0.],  # front
        [0., 1., 0.],  # back
        [0., 0., 1.],  # left
        [1., 1., 0.],  # right
    ])
    skins[:, 1, :, :3] = colors[:, :, None, None]
    skins[:, 1, :, 3] = 1
    position = torch.tensor([[[[0., 0., 0.], [0., 3., 0.]]]])
    camera_position = position.clone()
    camera_position[..., 2] += 1.5
    result = renderer({
        "player_skin": skins,
        "player_appearance_valid": torch.ones(b, a, 4, dtype=torch.bool),
        "player_position": position,
        "camera_position": camera_position,
        "camera_direction": torch.tensor([[[[0., 1., 0.], [0., 1., 0.]]]]),
        "camera": torch.full((b, t, a, 1), 1.2),
        "yaw_pitch": torch.zeros(b, t, a, 2),
        "player_valid": torch.ones(b, t, a, dtype=torch.bool),
    }, torch.tensor([0]))
    assert result.shape == (b, t, 11, 8, 8)
    occupied = result[0, 0, 3] > .5
    assert occupied.any()
    occupancy = result[0, 0, 3][occupied]
    assert (result[0, 0, 1][occupied] / occupancy).mean() > .99
    assert result[0, 0, 0][occupied].mean() < .01
    assert (result[0, 0, 8][occupied] / occupancy).mean() > .99


def test_world_translation_does_not_change_resident_features_or_projection():
    model = tiny_model().eval()
    cond = conditions(1)
    with torch.no_grad():
        original = model.encode_conditions(cond)
        cond["player_position"] += torch.tensor([100.,-20.,7.])
        translated = model.encode_conditions(cond)
    torch.testing.assert_close(original["extra_condition"], translated["extra_condition"])
    torch.testing.assert_close(original["actor_spatial_condition"], translated["actor_spatial_condition"])
    cam, anchor = np.array([2.,3.,1.5]), np.array([0,0,0])
    first = raster_camera(cam, [0,1,0], 1.2, anchor)
    second = raster_camera(cam + 100, [0,1,0], 1.2, anchor + 100)
    np.testing.assert_allclose(first, second)
    # Camera origin transformed by W2C maps to zero.
    from plot.models.renderer_backbone.camera_util import rotation_6d_to_matrix
    rotation = rotation_6d_to_matrix(torch.from_numpy(first[:6])).numpy()
    np.testing.assert_allclose(rotation @ ((cam - (anchor-.5))/48) + first[6:9], 0, atol=1e-7)


def test_two_blocks_commit_only_clean_frames_and_slide_cache():
    model = tiny_model().eval()
    cond = conditions(25)
    runner = RendererRollout(model, denoising_steps=2)
    runner.start(torch.randn(1,1,16,4,4), slice_conditions(cond,0,1))
    assert runner.next_frame == 1
    for start in (1,9,17):
        before = [c["k"].clone() for c in model.core.kv_caches]
        candidate_condition = slice_conditions(cond,start,start+8)
        with torch.no_grad():
            model(torch.randn(1,8,16,4,4), torch.ones(1,8), candidate_condition,
                  global_start_idx=start, cache_write=False)
        for cache, old in zip(model.core.kv_caches, before):
            torch.testing.assert_close(cache["k"], old)
        result = runner.generate(torch.randn(1,8,16,4,4), candidate_condition)
        assert result.shape == (1,8,16,4,4)
        assert len(runner.last_policy_layers) == 2
        assert all(layer.shape[:2] == (1, 8) for layer in runner.last_policy_layers)
        assert torch.isfinite(result).all()
        for cache in model.core.kv_caches:
            assert cache["global_end_index"].item() == start+8
            assert cache["local_end_index"].item() == min(16,start+8)


def test_rollout_can_start_at_absolute_episode_frame():
    model = tiny_model().eval()
    cond = conditions(9)
    runner = RendererRollout(model, denoising_steps=1)
    runner.start(torch.randn(1, 1, 16, 4, 4), slice_conditions(cond, 0, 1),
                 global_start_idx=19)
    assert runner.next_frame == 20
    assert {int(c["global_end_index"]) for c in model.core.kv_caches} == {20}


@pytest.mark.parametrize("simple_conditioning", [False, True])
def test_rollout_generates_64_frames_as_eight_cached_chunks(simple_conditioning):
    model = tiny_model(simple_conditioning=simple_conditioning).eval()
    cond = conditions(65)
    runner = RendererRollout(model, denoising_steps=1)
    runner.start(torch.randn(1, 1, 16, 4, 4), slice_conditions(cond, 0, 1))
    result = runner.generate_64(
        torch.randn(1, 64, 16, 4, 4), slice_conditions(cond, 1, 65)
    )
    assert result.shape == (1, 64, 16, 4, 4)
    assert runner.next_frame == 65
    assert {int(c["global_end_index"]) for c in model.core.kv_caches} == {65}


def test_simple_m3_needs_no_condition_mask_in_training_or_rollout():
    model = tiny_model(simple_conditioning=True)
    cond = conditions(65)
    cond.pop("condition_mask")
    clean = torch.randn(1, 65, 16, 4, 4)
    seen = []

    def check_core_input(_module, args):
        _, time, encoded = args
        assert "condition_mask" not in encoded
        seen.append((time.detach().clone(), encoded["action_prefix_mask"].clone()))

    hook = model.core.register_forward_pre_hook(check_core_input)
    try:
        model.train()
        model.core.gradient_checkpointing = True
        loss = renderer_flow_loss(model, clean, cond)
        loss.backward()
        assert torch.isfinite(loss)
        assert model.reference_encoder.encoder[0].weight.grad.abs().sum() > 0
        train_time, prefix = seen.pop()
        assert train_time[0, 0] == 0 and prefix.sum() == 1 and prefix[0, 0]

        model.eval()
        with torch.no_grad():
            expected = model(clean[:, :9], train_time[:, :9], slice_conditions(cond, 0, 9))
            # Shared/legacy callers may still supply this metadata; it has no
            # influence on M3-Simple's neural input.
            tagged = dict(slice_conditions(cond, 0, 9),
                          condition_mask=torch.ones(1, 9, dtype=torch.bool))
            actual = model(clean[:, :9], train_time[:, :9], tagged)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        seen.clear()

        runner = RendererRollout(model, denoising_steps=2)
        runner.start(clean[:, :1], slice_conditions(cond, 0, 1))
        output = runner.generate_64(torch.randn_like(clean[:, 1:]), slice_conditions(cond, 1, 65))
        assert output.shape == clean[:, 1:].shape and torch.isfinite(output).all()
        assert seen[0][0].eq(0).all() and seen[0][1].all()
        for block in range(8):
            denoise0, denoise1, commit = seen[1 + block * 3:1 + (block + 1) * 3]
            assert denoise0[0].eq(1).all()
            assert denoise1[0].eq(.5).all()
            assert commit[0].eq(0).all()
            assert not any(prefix.any() for _, prefix in (denoise0, denoise1, commit))
    finally:
        hook.remove()
        model.clear_cache()


def test_simple_m3_clean_history_cache_matches_full_forward():
    model = tiny_model(simple_conditioning=True).eval()
    cond = conditions(17)
    cond.pop("condition_mask")
    x = torch.randn(1, 17, 16, 4, 4)
    time = torch.zeros(1, 17)
    time[:, 9:] = .6
    with torch.no_grad():
        expected = model(x, time, cond)[:, 9:]
        runner = RendererRollout(model)
        try:
            runner.start(x[:, :1], slice_conditions(cond, 0, 1))
            runner._commit(x[:, 1:9], slice_conditions(cond, 1, 9), 1)
            actual = model(x[:, 9:], time[:, 9:], slice_conditions(cond, 9, 17),
                           global_start_idx=9, cache_write=False)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        finally:
            model.clear_cache()


def test_crops_keep_unknown_distinct_and_actions_are_incoming():
    raw = np.zeros((1,49,49,49,2),np.int16)
    raw[0,24,24,24,0] = 7
    blocks, known = merge_observation_crops(raw, [[0,0,0]], [2,0,0], BlockVocabulary((0,7)))
    assert blocks[22,24,24] == 1 and known[22,24,24]
    assert not known[-1].any() and known[0].all()
    actions = np.arange(12*23).reshape(12,1,23)
    aligned = incoming_actions(actions, 3, 9)
    assert not aligned[0].any()
    np.testing.assert_array_equal(aligned[1], actions[3])
    np.testing.assert_array_equal(aligned[-1], actions[10])


def test_target_crop_wins_a_one_voxel_same_frame_edit_boundary():
    raw = np.zeros((2, 49, 49, 49, 2), np.int16)
    raw[0, 24, 24, 24, 0] = 7
    centers = np.zeros((2, 3), np.int64)
    first, _ = merge_observation_crops(raw, centers, [0, 0, 0],
                                       BlockVocabulary((0, 7)), preferred_index=0)
    second, _ = merge_observation_crops(raw, centers, [0, 0, 0],
                                        BlockVocabulary((0, 7)), preferred_index=1)
    assert first[24, 24, 24] == 1
    assert second[24, 24, 24] == 0


def test_memory_snapshots_do_not_alias_later_writes():
    class Memory:
        def __init__(self):
            self.blocks = np.zeros((48,48,48),np.int32)
        def read_tile(self, anchor):
            return self.blocks, np.ones_like(self.blocks,bool)
    memory = Memory()
    block = RendererMemoryBlock([0,0,0])
    for step in range(8):
        memory.blocks[24,24,24] = step
        block.append(memory, transition_index=step, camera_world=[0,0,1.5],
                     camera_direction=[0,1,0], fov_x=1.2)
    assert block.conditions()["voxel_classes"][0,:,24,24,24].tolist() == list(range(8))


def test_continuous_dataset_preserves_state_time_and_splits(tmp_path, monkeypatch):
    import json
    import cv2
    from plot.data import renderer_dataset as module
    t, a = 17, 2
    (tmp_path / "manifest.json").write_text(json.dumps({
        "split": "train", "num_agents": a, "model_start_observation": 0,
        "agent_kinds": {"agent0": "human_like", "agent1": "npc_skeleton"}}))
    (tmp_path / "validation.json").write_text('{"usable":true}')
    (tmp_path / "training_metadata.json").write_text(json.dumps({"item_vocabulary": {"":0,"sword":1}}))
    (tmp_path / "events.jsonl").write_text(json.dumps({"event":"damage", "source":"agent1",
                                                        "target":"agent0", "observation_frame":9})+'\n')
    raw = np.zeros((t,a,49,49,49,2),np.int16)
    raw[9:,:,24,24,24,0] = 7
    health = np.full((t,a),20.,np.float32)
    health[9:,0] = 18
    np.savez(tmp_path / "data.npz", cam_pos=np.tile([0,0,1.5],(t,a,1)),
             cam_dir=np.tile([0,1,0],(t,a,1)), player_pos=np.zeros((t,a,3)),
             fov_x=np.full((t,a),1.2), player_yaw=np.zeros((t,a)), player_pitch=np.zeros((t,a)),
             obs_voxel_center=np.zeros((t,a,3),np.int64), obs_voxel_mt=raw,
             action_continuous=np.arange((t-1)*a*23).reshape(t-1,a,23),
             wielded_item_id=np.tile([0,1],(t-1,1)),
             player_health=health, player_health_valid=np.ones((t,a),bool),
             termination_flag=np.zeros((t-1,a),bool), entity_id=np.array(['agent0','agent1']),
             entity_kind=np.array(['player','player']),
             entity_render_object_id=np.tile([10,11],(t,1)).astype(np.uint16),
             entity_weapon_name=np.tile(['','sword'],(t,1)), entity_valid=np.ones((t,a),bool),
             instance_mask=np.stack((
                 np.full((t,8,8),11,np.uint16),
                 np.full((t,8,8),10,np.uint16),
             ),axis=1))
    for slot in range(a):
        root = tmp_path / 'players' / f'agent{slot}'
        root.mkdir(parents=True)
        for view in ('front','back','left','right'):
            reference = np.full((8,4,4),255,np.uint8)
            reference[0, 0] = [127, 63, 255, 0]
            assert cv2.imwrite(str(root / f'{view}.png'), reference)
    monkeypatch.setattr(module, '_read_video_frames', lambda path, indices, size:
                        {index: np.full((3,*size),index/255.,np.float32) for index in indices})
    ds = module.TextAgentRendererDataset(tmp_path,BlockVocabulary((0,7)),context_frames=17,
                                         image_size=(4,4),latent_size=(4,4))
    sample = ds[0]
    cond = sample['conditions']
    assert cond['voxel_classes'][:9,24,24,24].sum() == 0
    assert cond['voxel_classes'][9:,24,24,24].tolist() == [1]*8
    assert cond['event_cues'][9,0].tolist() == [0.,0.,1.,-2.]
    assert cond['event_cues'][9,1,1] == 1
    assert cond['resident_type'][:,1].tolist() == [3]*17
    assert sample['region_weight'].min() == 5
    assert sample['pixel_region_mask'].dtype == torch.bool
    assert sample['pixel_region_mask'].shape == (17, 1, 4, 4)
    assert sample['player_region_mask'].all()
    assert sample['conditions']['player_reference'].shape == (2, 4, 4, 8, 4)
    assert sample['conditions']['player_reference'][..., 0, 0].count_nonzero() == 0
    assert not {'instance_mask','crop_anchor','behavior_text'} & cond.keys()
    paired = module.TextAgentRendererDataset(
        tmp_path, BlockVocabulary((0,7)), context_frames=17,
        image_size=(4,4), latent_size=(4,4), targets_per_window=2,
    )[0]
    assert len(paired) == 2
    assert {int(view['conditions']['target_agent']) for view in paired} == {0, 1}
    flat = module.collate_renderer([paired])
    assert flat['rgb'].shape[:2] == (2, 17)
    assert flat['pixel_region_mask'].shape == (2, 17, 1, 4, 4)
    assert flat['player_region_mask'].shape == (2, 17, 1, 4, 4)
    assert flat['conditions']['player_reference'].shape == (2, 2, 4, 4, 8, 4)

    uniform = module.TextAgentRendererDataset(
        tmp_path, BlockVocabulary((0,7)), context_frames=17,
        image_size=(4,4), latent_size=(4,4), entity_region_upweight=0,
    )[0]
    assert uniform['region_weight'].min() == uniform['region_weight'].max() == 1
    focus_index = tmp_path / 'health_focus.pt'
    torch.save({'context_frames': 17, 'split': 'train',
                'rows': torch.tensor([[0, 0, 0]])}, focus_index)
    focused = module.TextAgentRendererDataset(
        tmp_path, BlockVocabulary((0,7)), context_frames=17,
        image_size=(4,4), latent_size=(4,4),
        health_focus_index=focus_index, health_focus_oversample=3,
    )
    assert len(focused) == 4  # two ordinary targets plus two extra copies of target zero
    from plot.data.chunked_npz import write_chunked_npz
    cache_root = tmp_path / "cache"
    cache_file = cache_root / "episode" / "data.m3c8.npz"
    write_chunked_npz(tmp_path / "data.npz", cache_file, chunk_frames=8)
    cached_manifest = json.loads((tmp_path / "manifest.json").read_text())
    cached_manifest["m3_chunk_cache_file"] = "episode/data.m3c8.npz"
    window_index = tmp_path / "portable_chunk_index.pt"
    torch.save({
        "context_frames": 17,
        "split": "train",
        "item_vocabulary": {"": 0, "sword": 1},
        "episodes": [{"path": ".", "manifest": cached_manifest}],
        "windows": [(0, 0, 0)],
    }, window_index)
    cached = module.TextAgentRendererDataset(
        tmp_path, BlockVocabulary((0,7)), context_frames=17,
        image_size=(4,4), latent_size=(4,4), window_index=window_index,
        chunk_cache_root=cache_root,
    )[0]
    assert torch.equal(cached["region_weight"], sample["region_weight"])
    assert torch.equal(cached["pixel_region_mask"], sample["pixel_region_mask"])
    assert torch.equal(
        cached["conditions"]["voxel_classes"], sample["conditions"]["voxel_classes"]
    )
    with pytest.raises(ValueError,match='no accepted'):
        module.TextAgentRendererDataset(tmp_path,BlockVocabulary((0,7)),split='test',context_frames=17)


def test_counterfactual_groups_use_identical_pure_noise_and_supervise_target_difference():
    time = _sample_blockwise_train_time(
        4, 17, 8, torch.device("cpu"),
        torch.Generator().manual_seed(3), counterfactual_group_size=2,
    )
    torch.testing.assert_close(time[:, 0], torch.zeros(4))
    torch.testing.assert_close(time[:, 1:], torch.ones(4, 16))
    clean = torch.zeros(2, 17, 2, 2, 2)
    clean[1, 1:] = 1

    class CaptureNoisyInput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(context_frames=17, block_frames=8)
            self.core = SimpleNamespace(kv_caches=None)

        def forward(self, noisy, _time, _conditions):
            self.noisy = noisy.detach().clone()
            return torch.zeros_like(noisy)

    capture = CaptureNoisyInput()
    renderer_flow_loss(
        capture, clean, {}, counterfactual_group_size=2,
        generator=torch.Generator().manual_seed(11),
    )
    torch.testing.assert_close(capture.noisy[0, 1:], capture.noisy[1, 1:])

    mask = torch.ones(2, 17, 1, 4, 4, dtype=torch.bool)
    ignored_reference = torch.zeros_like(clean)
    assert renderer_counterfactual_player_loss(
        ignored_reference, clean, mask, 2
    ) > 0
    assert renderer_counterfactual_player_loss(clean, clean, mask, 2) == 0


def test_mixed_counterfactual_schedule_is_stable_and_close_to_requested_ratio():
    first = [use_counterfactual_step(step, 0.2, 17) for step in range(1, 10_001)]
    resumed = [use_counterfactual_step(step, 0.2, 17) for step in range(5_001, 10_001)]
    assert first[5_000:] == resumed
    assert 0.18 < sum(first) / len(first) < 0.22
    assert not any(use_counterfactual_step(step, 0.0, 17) for step in range(10))
    assert all(use_counterfactual_step(step, 1.0, 17) for step in range(10))


def test_resume_can_append_item_embedding_and_adam_rows():
    old = Renderer(RendererArgs(
        3, 8, input_h=4, input_w=4, hidden_size=32, depth=2,
        num_heads=4, voxel_channels=4, condition_dim=16, actor_channels=4,
        context_frames=65, cache_frames=16, gradient_checkpointing=False,
        gpu_rasterizer=False,
    ))
    old_optimizer = torch.optim.AdamW(old.parameters(), lr=1e-4)
    for parameter in old.parameters():
        parameter.grad = torch.ones_like(parameter)
    old_optimizer.step()
    checkpoint = {
        "model": old.state_dict(),
        "optimizer": old_optimizer.state_dict(),
        "config": {"item_vocabulary": {f"item_{index}": index for index in range(1, 8)}},
        "step": 12,
    }

    torch.manual_seed(123)
    extended = Renderer(RendererArgs(
        3, 9, input_h=4, input_w=4, hidden_size=32, depth=2,
        num_heads=4, voxel_channels=4, condition_dim=16, actor_channels=4,
        context_frames=65, cache_frames=16, gradient_checkpointing=False,
        gpu_rasterizer=False,
    ))
    new_row_before_resume = extended.resident_encoder.item_embedder.weight[8].detach().clone()
    extended_optimizer = torch.optim.AdamW(extended.parameters(), lr=1e-4)
    report = load_renderer_resume(
        extended,
        extended_optimizer,
        checkpoint,
        {**checkpoint["config"]["item_vocabulary"], "new_item": 8},
        allow_item_vocabulary_extension=True,
    )

    assert report["new_items"] == {"new_item": 8}
    torch.testing.assert_close(
        extended.resident_encoder.item_embedder.weight[:8],
        old.resident_encoder.item_embedder.weight,
    )
    torch.testing.assert_close(
        extended.resident_encoder.item_embedder.weight[8], new_row_before_resume,
    )
    state = extended_optimizer.state[extended.resident_encoder.item_embedder.weight]
    assert state["exp_avg"].shape == (9, 64)
    assert state["exp_avg_sq"].shape == (9, 64)
    assert torch.count_nonzero(state["exp_avg"][8]) == 0
    assert torch.count_nonzero(state["exp_avg_sq"][8]) == 0
    for parameter in extended.parameters():
        parameter.grad = torch.ones_like(parameter)
    extended_optimizer.step()


def test_resume_rejects_existing_item_id_changes():
    model = tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {"item_vocabulary": {"pick": 1}},
    }
    with pytest.raises(RuntimeError, match="changed existing checkpoint IDs"):
        load_renderer_resume(model, optimizer, checkpoint, {"pick": 2})


def test_counterfactual_dataset_canonicalizes_voxels_but_keeps_references():
    class Base:
        def _read_target(self, episode, start, target):
            return {"conditions": {
                "voxel_classes": torch.full((2,), episode),
                "voxel_known": torch.ones(2, dtype=torch.bool),
                "player_position": torch.tensor([
                    [[0., 0., 0.], [1., 2., 3.]],
                    [[0., 1., 0.], [1., 3., 3.]],
                ]) + torch.tensor([float(episode), -2. * episode, 3. * episode]),
                "player_valid": torch.ones(2, 2, dtype=torch.bool),
                "camera_direction": torch.zeros(2, 2, 3) + episode * 1e-7,
                "raster_camera": torch.zeros(2, 10) + episode * 1e-7,
                "player_reference": torch.full((2, 4, 4, 2, 2), float(episode)),
            }}

    dataset = AppearanceCounterfactualRendererDataset.__new__(
        AppearanceCounterfactualRendererDataset
    )
    dataset.base = Base()
    dataset.index = [((0, 0, 0), (1, 0, 0))]
    samples = dataset[0]
    torch.testing.assert_close(
        samples[0]["conditions"]["voxel_classes"],
        samples[1]["conditions"]["voxel_classes"],
    )
    torch.testing.assert_close(
        samples[0]["conditions"]["player_position"],
        samples[1]["conditions"]["player_position"],
    )
    torch.testing.assert_close(
        samples[0]["conditions"]["raster_camera"],
        samples[1]["conditions"]["raster_camera"],
    )
    assert not torch.equal(
        samples[0]["conditions"]["player_reference"],
        samples[1]["conditions"]["player_reference"],
    )


def test_counterfactual_dataset_rejects_relative_geometry_changes():
    class Base:
        def _read_target(self, episode, start, target):
            position = torch.tensor([[[0., 0., 0.], [1., 2., 3.]]])
            if episode:
                position[:, 1, 0] += 1
            return {"conditions": {
                "voxel_classes": torch.zeros(2, dtype=torch.long),
                "voxel_known": torch.ones(2, dtype=torch.bool),
                "player_position": position,
                "player_valid": torch.ones(1, 2, dtype=torch.bool),
                "player_reference": torch.full((2, 4, 4, 2, 2), float(episode)),
            }}

    dataset = AppearanceCounterfactualRendererDataset.__new__(
        AppearanceCounterfactualRendererDataset
    )
    dataset.base = Base()
    dataset.index = [((0, 0, 0), (1, 0, 0))]
    with pytest.raises(ValueError, match="relative player geometry"):
        dataset[0]

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rasterization check")
def test_gpu_voxel_projection_backward():
    model = Renderer(replace(tiny_model().cfg, gpu_rasterizer=True, depth=1)).cuda().train()
    # Exercise gradients into the actual voxel projection rather than a zero output head.
    with torch.no_grad():
        model.core.final_layer.linear.weight.normal_(std=.03)
    cond = conditions(1)
    cond.pop('raster_features')
    cond.pop('raster_depth')
    cond.update(voxel_classes=torch.zeros(1,1,48,48,48,dtype=torch.long),
                voxel_known=torch.ones(1,1,48,48,48,dtype=torch.bool),
                raster_camera=torch.from_numpy(raster_camera([0,0,1.5],[0,1,0],1.2,[0,0,0]))[None,None])
    cond = {k:v.cuda() for k,v in cond.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        result = model(torch.randn(1,1,16,4,4,device='cuda'),torch.ones(1,1,device='cuda'),cond)
        loss = (result-1).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert model.voxel_embedder.weight.grad.abs().sum() > 0


def test_frozen_codec_rgb_latent_layout():
    from dataclasses import asdict
    from plot.models.renderer_codec import RendererCodec
    from plot.models.renderer_backbone.vae_pixel import ViTVae, ViTVaeArgs
    cfg = ViTVaeArgs(input_height=40,input_width=40,enc_dim=32,enc_depth=1,enc_heads=4,
                     dec_dim=32,dec_depth=1,dec_heads=4)
    weights = ViTVae(**asdict(cfg)).state_dict()
    codec = RendererCodec(weights,cfg).train()
    latent = codec.encode(torch.rand(1,9,3,40,40))
    rgb = codec.decode(latent)
    assert latent.shape == (1,9,16,4,4) and rgb.shape == (1,9,3,40,40)
    assert 0 <= rgb.min() <= rgb.max() <= 1
    assert not latent.requires_grad and not codec.vae.training
    predicted = latent[:, :1].detach().requires_grad_()
    codec.decode_for_loss(predicted).mean().backward()
    assert predicted.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in codec.parameters())


def test_normalized_codec_simple_flow_backward_and_rollout():
    from dataclasses import asdict
    from plot.models.renderer_codec import RendererCodec
    from plot.models.latent_normalization import pixel_vae_latent_stats
    from plot.models.renderer_backbone.vae_pixel import ViTVae, ViTVaeArgs

    cfg = ViTVaeArgs(input_height=40, input_width=40, enc_dim=32, enc_depth=1,
                     enc_heads=4, dec_dim=32, dec_depth=1, dec_heads=4)
    codec = RendererCodec(
        ViTVae(**asdict(cfg)).state_dict(), cfg,
        latent_normalization=pixel_vae_latent_stats(),
    )
    model = tiny_model(simple_conditioning=True).train()
    cond = conditions(65)
    cond.pop("condition_mask")
    clean = codec.encode(torch.rand(1, 65, 3, 40, 40))
    loss = renderer_flow_loss(model, clean, cond)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.core.final_layer.linear.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in codec.parameters())
    model.eval()
    with torch.no_grad():
        rollout = RendererRollout(model, denoising_steps=2)
        rollout.start(clean[:, :1], slice_conditions(cond, 0, 1))
        prediction = rollout.generate_64(
            torch.randn_like(clean[:, 1:]), slice_conditions(cond, 1, 65)
        )
        rgb = codec.decode(prediction)
    assert rgb.shape == (1, 64, 3, 40, 40)
    assert torch.isfinite(rgb).all()


def test_full_resolution_entity_and_health_losses_are_differentiable():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent[:, :, :3]

    predicted = torch.zeros(1, 3, 3, 4, 4, requires_grad=True)
    target = torch.zeros_like(predicted)
    target[:, 1:, :, :2, :2] = 1
    entity = torch.zeros(1, 3, 1, 4, 4, dtype=torch.bool)
    entity[:, 1:, :, :2, :2] = True
    player = torch.zeros_like(entity)
    player[:, 2:, :, :2, :2] = True
    losses = renderer_pixel_losses(
        Codec(), predicted, target, entity, player_region_mask=player,
        frames_per_sample=2, generator=torch.Generator().manual_seed(0),
        health_box=(0, 0, 0.5, 0.5),
    )
    assert losses['entity_pixel_l1'] == 1
    assert losses['health_pixel_l1'] == 1
    assert losses['entity_pixel_edge'] > 0
    assert losses['player_pixel_l1'] == 1
    assert losses['player_pixel_edge'] > 0
    sum(losses.values()).backward()
    assert predicted.grad.abs().sum() > 0


def test_pixel_frame_selection_includes_minimum_target_health():
    mask = torch.zeros(2, 5, 1, 4, 4, dtype=torch.bool)
    mask[:, 2, :, :3, :3] = True
    hp = torch.full((2, 5, 2), 20.0)
    hp[0, 3:, 1] = 12
    target = torch.tensor([1, 0])
    indices, target_hp = select_renderer_pixel_frames(
        mask, frames_per_sample=2, hp=hp, target_agent=target,
        generator=torch.Generator().manual_seed(3),
    )
    assert indices.shape == (2, 2)
    assert indices[0, 1].item() in {3, 4}
    assert 1 <= indices[1, 1].item() < 5
    torch.testing.assert_close(target_hp[0], hp[0, :, 1])


def test_combined_renderer_loss_can_disable_pixel_decoder():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            raise AssertionError("zero pixel weights must skip decoding")

    model = tiny_model().train()
    clean = torch.randn(1, 65, 16, 4, 4)
    terms = renderer_training_losses(
        model, Codec(), clean, conditions(65), torch.rand(1, 65, 3, 4, 4),
        torch.ones(1, 65, 1, 4, 4, dtype=torch.bool),
        entity_pixel_l1_weight=0, entity_pixel_edge_weight=0,
        health_pixel_l1_weight=0,
    )
    torch.testing.assert_close(terms['total_loss'], terms['flow_loss'])
    assert terms['auxiliary_loss'] == 0


def test_renderer_loss_can_disable_flow_for_pixel_only_adaptation():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent[:, :, :3]

    model = tiny_model().train()
    clean = torch.randn(1, 65, 16, 4, 4)
    rgb = torch.rand(1, 65, 3, 4, 4)
    mask = torch.ones(1, 65, 1, 4, 4, dtype=torch.bool)
    terms = renderer_training_losses(
        model, Codec(), clean, conditions(65), rgb, mask,
        player_region_mask=mask,
        frames_per_sample=3,
        flow_loss_weight=0,
        entity_pixel_l1_weight=0,
        entity_pixel_edge_weight=0,
        player_pixel_l1_weight=2,
        player_pixel_edge_weight=0,
        health_pixel_l1_weight=0,
    )
    torch.testing.assert_close(
        terms['total_loss'], 2 * terms['player_pixel_l1']
    )
    assert terms['flow_loss'] > 0


def test_flow_loss_mode_disables_every_auxiliary_weight():
    configured = {
        "entity_pixel_l1_weight": 0.5,
        "entity_pixel_edge_weight": 0.2,
        "player_pixel_l1_weight": 8.0,
        "player_pixel_edge_weight": 2.0,
        "health_pixel_l1_weight": 1.0,
        "player_identity_loss_weight": 3.0,
        "counterfactual_player_difference_weight": 4.0,
    }
    combined = effective_auxiliary_loss_weights(
        SimpleNamespace(loss_mode="combined", **configured)
    )
    flow = effective_auxiliary_loss_weights(
        SimpleNamespace(loss_mode="flow", **configured)
    )
    assert combined == configured
    assert set(flow) == set(configured)
    assert all(weight == 0 for weight in flow.values())


def test_kv_cache_uses_requested_inference_dtype():
    model = tiny_model().eval()
    model.init_kv_cache(1, dtype=torch.bfloat16)
    assert all(cache["k"].dtype == torch.bfloat16 for cache in model.core.kv_caches)
    assert all(cache["v"].dtype == torch.bfloat16 for cache in model.core.kv_caches)
    model.clear_cache()
