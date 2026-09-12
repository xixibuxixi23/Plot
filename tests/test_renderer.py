from dataclasses import replace
import numpy as np
import pytest
import torch

from plot.data.fill_dataset import BlockVocabulary
from plot.data.renderer_dataset import incoming_actions, merge_observation_crops, raster_camera
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_backbone.player_spatial_condition import ViewAwarePlayerAppearance
from plot.pipelines.renderer_pipeline import RendererMemoryBlock
from plot.training.renderer_trainer import (
    RendererRollout,
    renderer_flow_loss,
    renderer_pixel_losses,
    renderer_training_losses,
    select_renderer_pixel_frames,
    slice_conditions,
)


torch.set_num_threads(2)


def tiny_model():
    torch.manual_seed(42)
    model = Renderer(RendererArgs(3, 8, input_h=4, input_w=4, hidden_size=32, depth=2,
                                 num_heads=4, voxel_channels=4, condition_dim=16, actor_channels=4,
                                 context_frames=65, cache_frames=16,
                                 gradient_checkpointing=False, gpu_rasterizer=False))
    # Zero initialization would make future-invariance tests vacuous.
    with torch.no_grad():
        model.core.final_layer.linear.weight.normal_(std=.03)
        for block in model.core.blocks:
            block.s_adaLN_modulation[-1].weight.normal_(std=.03)
            block.t_adaLN_modulation[-1].weight.normal_(std=.03)
        model.core.extra_condition_embedder.weight.normal_(std=.03)
        model.core.actor_condition_embedder.weight.normal_(std=.03)
    return model


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


def test_future_latents_and_state_do_not_change_prefix():
    model = tiny_model().eval()
    cond = conditions(9)
    x, time = torch.randn(1,9,16,4,4), torch.rand(1,9)
    with torch.no_grad():
        first = model(x, time, cond)
        x[:,5:] += 100
        cond["hp"][:,5:] = 1
        cond["raster_features"][:,5:] *= 10
        second = model(x, time, cond)
    torch.testing.assert_close(first[:,:5], second[:,:5])
    assert not torch.allclose(first[:,5:], second[:,5:])


def test_training_backward_and_supervision_only_masks():
    model = tiny_model().train()
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


def test_rollout_generates_64_frames_as_eight_cached_chunks():
    model = tiny_model().eval()
    cond = conditions(65)
    runner = RendererRollout(model, denoising_steps=1)
    runner.start(torch.randn(1, 1, 16, 4, 4), slice_conditions(cond, 0, 1))
    result = runner.generate_64(
        torch.randn(1, 64, 16, 4, 4), slice_conditions(cond, 1, 65)
    )
    assert result.shape == (1, 64, 16, 4, 4)
    assert runner.next_frame == 65
    assert {int(c["global_end_index"]) for c in model.core.kv_caches} == {65}


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
             entity_weapon_name=np.tile(['','sword'],(t,1)), entity_valid=np.ones((t,a),bool),
             instance_mask=np.full((t,a,4,4),65535,np.uint16))
    for slot in range(a):
        root = tmp_path / 'players' / f'agent{slot}'
        root.mkdir(parents=True)
        for view in ('front','back','left','right'):
            assert cv2.imwrite(str(root / f'{view}.png'), np.full((4,4,4),255,np.uint8))
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
    with pytest.raises(ValueError,match='no accepted'):
        module.TextAgentRendererDataset(tmp_path,BlockVocabulary((0,7)),split='test',context_frames=17)


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


def test_full_resolution_entity_and_health_losses_are_differentiable():
    class Codec:
        def decode_for_loss(self, latent, chunk_size=1):
            return latent[:, :, :3]

    predicted = torch.zeros(1, 3, 3, 4, 4, requires_grad=True)
    target = torch.zeros_like(predicted)
    target[:, 1:, :, :2, :2] = 1
    entity = torch.zeros(1, 3, 1, 4, 4, dtype=torch.bool)
    entity[:, 1:, :, :2, :2] = True
    losses = renderer_pixel_losses(
        Codec(), predicted, target, entity,
        frames_per_sample=2, generator=torch.Generator().manual_seed(0),
        health_box=(0, 0, 0.5, 0.5),
    )
    assert losses['entity_pixel_l1'] == 1
    assert losses['health_pixel_l1'] == 1
    assert losses['entity_pixel_edge'] > 0
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


def test_kv_cache_uses_requested_inference_dtype():
    model = tiny_model().eval()
    model.init_kv_cache(1, dtype=torch.bfloat16)
    assert all(cache["k"].dtype == torch.bfloat16 for cache in model.core.kv_caches)
    assert all(cache["v"].dtype == torch.bfloat16 for cache in model.core.kv_caches)
    model.clear_cache()
