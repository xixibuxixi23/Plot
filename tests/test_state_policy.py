import json

import numpy as np
import pytest
import torch

from plot.models.state_policy import StateInhabitantPolicy, StatePolicyArgs
from plot.data.state_policy_dataset import (assemble_state_inputs, collate_state_policy,
                                            read_state_sample, read_npz_slice)
from plot.data.fill_dataset import BlockVocabulary


def sample():
    position = np.zeros((8, 3, 3), np.float32)
    position[:, 1, 0] = 3
    position[:, 2, 0] = 6
    inputs = assemble_state_inputs(
        blocks=np.zeros((48,48,48), np.int64), known=np.ones((48,48,48), bool),
        anchor=np.zeros(3), position=position, angles=np.zeros((8,3,2)), hp=np.ones((8,3))*20,
        camera_relative=np.zeros((8,3,3)), camera_direction=np.ones((8,3,3)),
        event_cues=np.zeros((8,3,4)), held_item=np.zeros((8,3), np.int64),
        resident_type=np.zeros((8,3), np.int64), resident_valid=np.ones((8,3), bool),
        incoming_actions=np.zeros((8,23)), target=0)
    return dict(inputs=inputs, target_actions=torch.zeros(8,23), valid_mask=torch.ones(8,dtype=torch.bool))


def tiny(profile='zombie_melee'):
    return StateInhabitantPolicy(StatePolicyArgs(3, 4, profile, hidden=32, heads=4,
                resident_layers=1, temporal_layers=1, decoder_layers=1, dropout=0.))


def test_backward_and_resident_permutation():
    torch.manual_seed(11)
    model = tiny().eval()
    batch = collate_state_policy([sample()])
    first = model(batch['inputs'])
    perm = torch.tensor([2,0,1])
    changed = dict(batch['inputs'])
    for key in ('resident_state','held_item','resident_type','resident_valid'):
        changed[key] = changed[key][:,:,perm]
    changed['target_agent'] = torch.tensor([1])
    second = model(changed)
    for key in first: torch.testing.assert_close(first[key], second[key], atol=2e-6, rtol=2e-5)
    loss, _ = model.loss(first, batch['target_actions'])
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.decode(first).shape == (1,8,23)


def test_padding_and_peaceful_constraints():
    model = tiny('villager_peaceful').eval()
    batch = collate_state_policy([sample()])
    inputs = batch['inputs']
    inputs['history_valid'][:, :7] = False
    inputs['resident_valid'][:, :7] = False
    first = model(inputs)
    changed = {k: v.clone() for k,v in inputs.items()}
    changed['resident_state'][:, :7] = 100
    changed['incoming_actions'][:, :7] = 100
    second = model(changed)
    for key in first: torch.testing.assert_close(first[key], second[key])
    actions = model.decode(first)
    assert not actions[..., [8,9,10,*range(12,21)]].any()


def test_unknown_voxels_ignore_underlying_class():
    model = tiny().eval()
    inputs = collate_state_policy([sample()])['inputs']
    inputs['voxel_known'][:] = False
    first = model(inputs)
    inputs['voxel_classes'][:] = 2
    second = model(inputs)
    for key in first: torch.testing.assert_close(first[key], second[key])


def test_real_adapter_alignment_and_no_future_leakage(tmp_path):
    t,a = 18,2
    data = dict(player_pos=np.zeros((t,a,3),np.float32),
                player_yaw=np.zeros((t,a)), player_pitch=np.zeros((t,a)),
                player_health=np.ones((t,a))*20, cam_pos=np.zeros((t,a,3)),
                cam_dir=np.ones((t,a,3)), wielded_item_id=np.zeros((t-1,a),np.int64),
                fov_x=np.ones((t,a),np.float32)*1.2,
                entity_id=np.array(['agent0','agent1']), entity_valid=np.ones((t,a),bool),
                action_continuous=np.zeros((t-1,a,23),np.float32),
                obs_voxel_center=np.zeros((t,a,3),np.int16),
                obs_voxel_mt=np.zeros((t,a,49,49,49,2),np.int16))
    data['action_continuous'][:,:,0] = np.arange(t-1)[:,None]
    (tmp_path/'manifest.json').write_text(json.dumps({'agent_kinds': {'agent0':'human_like','agent1':'npc_zombie'}}))
    (tmp_path/'training_metadata.json').write_text(json.dumps({'item_vocabulary': {'':0}}))
    np.savez(tmp_path/'data.npz', **data)
    kwargs = dict(episode=tmp_path, anchor=8, target=1, vocabulary=BlockVocabulary((0,1)),
                  model_start=0, item_vocabulary={'':0})
    first = read_state_sample(**kwargs)
    torch.testing.assert_close(first['inputs']['incoming_actions'][:,0], torch.arange(8).float())
    torch.testing.assert_close(first['target_actions'][:,0], torch.arange(8,16).float())
    for key in ('player_pos','player_health','obs_voxel_mt'): data[key][9:] = 1
    data['action_continuous'][8:] = 100
    np.savez(tmp_path/'data.npz', **data)
    second = read_state_sample(**kwargs)
    for key in first['inputs']: torch.testing.assert_close(first['inputs'][key], second['inputs'][key])
    assert not torch.equal(first['target_actions'], second['target_actions'])
    enriched = read_state_sample(**kwargs, include_m3_fields=True)
    assert enriched['inputs']['raster_camera'].shape == (10,)
    torch.testing.assert_close(enriched['inputs']['resident_actions'][:,1,0],torch.arange(8).float())
    for key in second['inputs']:torch.testing.assert_close(enriched['inputs'][key],second['inputs'][key])
    np.testing.assert_array_equal(read_npz_slice(tmp_path/'data.npz', 'obs_voxel_mt', 8,9), data['obs_voxel_mt'][8:9])
    initial = read_state_sample(**dict(kwargs, anchor=0))
    assert initial['inputs']['history_valid'].tolist() == [False]*7+[True]
    # Full-data cache must reproduce direct reads for multiple anchors/targets.
    from types import SimpleNamespace
    from dataset_toolkits.prepare_state_policy import cache_episode
    from plot.data.state_policy_dataset import load_state_cache
    np.savez_compressed(tmp_path/'data.npz', **data)
    records, labels, _, _ = cache_episode(('train', tmp_path, 0, [(8,1),(0,1),(8,0)],
        'test', SimpleNamespace(profile='zombie_melee',output_dir=tmp_path),
        BlockVocabulary((0,1)), {'':0}))
    for row, label in zip(records, labels):
        cached = load_state_cache(tmp_path/row['cache'])
        direct = read_state_sample(**dict(kwargs, anchor=row['anchor'],target=row['agent_slot']))
        for key in direct['inputs']: torch.testing.assert_close(cached['inputs'][key], direct['inputs'][key])
        torch.testing.assert_close(cached['target_actions'], direct['target_actions'])
        np.testing.assert_array_equal(label, direct['target_actions'].numpy())


def test_checkpoint_roundtrip(tmp_path):
    model = tiny().eval()
    inputs = collate_state_policy([sample()])['inputs']
    torch.save(model.state_dict(), tmp_path/'model.pt')
    restored = tiny().eval()
    restored.load_state_dict(torch.load(tmp_path/'model.pt', weights_only=True))
    for key, value in model(inputs).items(): torch.testing.assert_close(value, restored(inputs)[key])


def test_live_capture_is_immutable_and_bootstraps():
    from plot.pipelines.state_policy_pipeline import StatePolicySession
    from plot.transition_state import CharRow
    class Memory:
        def read_tile(self, anchor):
            return np.zeros((48,48,48),np.int64), np.ones((48,48,48),bool)
    row = CharRow('npc',0,(0.,0.,0.),0.,0.,20.,0,2,(0.,0.,1.),(1.,0.,0.))
    session = StatePolicySession(tiny(), ['npc'], 'npc', {0:0})
    session.append({'npc':row})
    captured = session.capture(Memory())
    assert captured['history_valid'].tolist() == [False]*7+[True]
    row.hp = 1.
    session.append({'npc':row}, incoming_actions=np.zeros((1,23)))
    assert captured['resident_state'][-1,0,7] == 1.
    assert session.act(captured).shape == (8,23)


def test_attack_positive_weight_only_changes_positive_attack_supervision():
    model = tiny()
    logits = {'keys':torch.zeros(1,8,10,requires_grad=True),
              'hotbar':torch.zeros(1,8,10), 'mouse_x':torch.zeros(1,8,17),
              'mouse_y':torch.zeros(1,8,17)}
    actions = torch.zeros(1,8,23)
    regular,_ = model.loss(logits, actions)
    weighted,_ = model.loss(logits, actions, attack_positive_weight=8.)
    torch.testing.assert_close(regular, weighted)
    actions[:,0,8] = 1
    regular,_ = model.loss(logits, actions)
    weighted,_ = model.loss(logits, actions, attack_positive_weight=8.)
    torch.testing.assert_close(weighted-regular, torch.tensor(7*np.log(2)/80,dtype=torch.float32))
    gradient_regular = torch.autograd.grad(regular, logits['keys'], retain_graph=True)[0]
    gradient_weighted = torch.autograd.grad(weighted, logits['keys'])[0]
    torch.testing.assert_close(gradient_weighted[0,0,7], gradient_regular[0,0,7]*8)
    mask = torch.ones_like(gradient_regular,dtype=torch.bool); mask[0,0,7] = False
    torch.testing.assert_close(gradient_weighted[mask], gradient_regular[mask])


def test_language_text_controls_logits_and_masks_padding():
    torch.manual_seed(123)
    model = StateInhabitantPolicy(StatePolicyArgs(3,4,'language_builder',hidden=32,heads=4,
        resident_layers=1,temporal_layers=1,decoder_layers=1,dropout=0.,text_hidden_size=12)).eval()
    inputs = collate_state_policy([sample()])['inputs']
    for name in ('shared','current'):
        inputs[name+'_text'] = torch.randn(1,5,12)
        inputs[name+'_text_mask'] = torch.tensor([[True,True,True,False,False]])
    first = model(inputs)
    changed = {key:value.clone() for key,value in inputs.items()}
    changed['current_text'][:,3:] = 1000
    for key,value in first.items(): torch.testing.assert_close(value, model(changed)[key])
    changed['current_text'][:,:3] *= -1
    assert any(not torch.allclose(value, model(changed)[key]) for key,value in first.items())
    loss,_ = model.loss(first, torch.zeros(1,8,23))
    loss.backward()
    assert model.text_projection.weight.grad.abs().sum() > 0
    for name in ('shared','current'): changed[name+'_text_mask'].zero_()
    blank = model(changed)
    assert all(torch.isfinite(value).all() for value in blank.values())
    changed['current_text'] *= 100
    for key,value in blank.items(): torch.testing.assert_close(value, model(changed)[key])
