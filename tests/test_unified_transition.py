"""CPU regressions for the opt-in shared-trunk M2 architecture."""
from dataclasses import asdict
import io
import importlib.util
import json
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from plot.data.transition_dataset import collate_transition
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.kinematics import KinematicsConfig
from plot.pipelines.closed_loop_pipeline import decode_transition
from plot.pipelines.transition_pipeline import TransitionCommitter
from plot.training.transition_trainer import transition_loss
from plot.transition_state import CharRow
from plot.world_memory import WorldMemory
from test_transition import sample, add_event_context

torch.set_num_threads(2)


def network(**kwargs):
    torch.manual_seed(17)
    cfg = dict(num_block_classes=3, num_items=10, width=32, depth=2, heads=4,
               unified_interactions=True, air_class=0, event_queries=8)
    cfg.update(kwargs)
    return TransitionNetwork(TransitionArgs(**cfg))


def batch(a=2):
    value = collate_transition([sample(a)])
    value['inputs'].pop('previous_rgb')
    value.pop('metadata')
    return value


def test_single_trunk_no_player_or_attack_transformer_and_no_rgb_required():
    net = network()
    assert not hasattr(net, 'player_blocks') and not hasattr(net, 'attack_decoder')
    assert isinstance(net.player_adapter, torch.nn.Identity)
    values = batch()
    out = net(values['inputs'])
    assert out['event_operation_logits'].shape == (1,2,8,3)
    assert out['event_address_logits'].shape == (1,2,8,2200)
    assert out['pose'].shape == (1,8,2,5)
    torch.testing.assert_close(out['player_hidden'],out['hidden'],rtol=0,atol=0)
    assert 'attack_time_logits' not in out
    loss, metrics = transition_loss(net, out, **values, objective='unified')
    loss.backward()
    assert np.isfinite(metrics['loss'])
    for head in (net.pose_head,net.camera_head,net.held_head,net.event_time_head,
                 net.event_operation_head,net.event_pointer_query,net.block_head,net.damage_head):
        assert head.weight.grad.abs().sum() > 0
    assert net.visual[0].weight.grad is None


@pytest.mark.parametrize('term', ['position','event_block','event_damage'])
def test_motion_edits_and_attacks_all_backpropagate_to_same_trunk(term):
    # Isolate supervision while retaining a full common forward graph.
    net = network()
    if term == 'position':
        # The production residual head starts at zero to preserve the kinematic
        # proposal; trunk gradients from pose begin after its first update.
        with torch.no_grad():
            net.pose_head.weight.normal_(std=.02)
    values = batch(); out = net(values['inputs'])
    if term == 'position':
        loss = out['pose'].square().mean()
    else:
        addresses = torch.full((1,2,8),2199,dtype=torch.long)
        addresses[0,0,0] = 0 if term == 'event_block' else 2198
        block, damage = net.event_payloads(out, addresses)
        loss = block[0,0,0].square().mean() if term == 'event_block' else damage[0,0,0].square()
    loss.backward()
    assert net.blocks[0].time.in_proj_weight.grad.abs().sum() > 0
    assert net.blocks[0].memory.in_proj_weight.grad.abs().sum() > 0


def test_padding_unknown_self_targets_and_player_forward_share_exact_path():
    net = network().eval()
    two, four = sample(2), sample(4)
    two['inputs']['voxel_known'][:,5,5,5] = False
    single = collate_transition([two])['inputs']
    mixed = collate_transition([two,four])['inputs']
    with torch.no_grad():
        a, b = net(single), net(mixed)
        c = net.forward_player(single)
    torch.testing.assert_close(a['pose'][0], b['pose'][0,:,:2], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(a['pose'], c['pose'])
    torch.testing.assert_close(a['event_address_logits'][0,:,:,:2197],
                               b['event_address_logits'][0,:2,:,:2197],atol=1e-5,rtol=1e-5)
    logits = b['event_address_logits']
    assert torch.isneginf(logits[0,0,:,2197]).all()
    assert torch.isneginf(logits[0,:2,:,2199:2201]).all()
    assert torch.isneginf(logits[0,:2,:,5*169+5*13+5]).all()
    decoded = net.decode_interactions(b)
    assert (decoded['address'][0,:,2:] == 2201).all()
    loss, _ = transition_loss(net,b,mixed,collate_transition([two,four])['targets'],objective='unified')
    assert torch.isfinite(loss)


def test_state_only_mode_ignores_rgb_and_camera_still_conditions_predictions():
    net = network().eval(); values = batch()['inputs']
    with torch.no_grad():
        first = net(values)
        values['previous_rgb'] = torch.full((1,2,3,32,32),float('nan'))
        second = net(values)
        values['camera_direction'] = -values['camera_direction']
        third = net(values)
    torch.testing.assert_close(first['event_address_logits'],second['event_address_logits'])
    assert not torch.allclose(first['hidden'],third['hidden'])


def test_unified_rgb_ablation_and_four_frame_history():
    net = network(unified_use_rgb=True,event_context_frames=4).eval()
    values = collate_transition([add_event_context(sample())])
    values.pop('metadata')
    with torch.no_grad():
        first = net(values['inputs'])
        values['inputs']['event_context_actions'][:,-1,:,9] = 1
        second = net(values['inputs'])
    assert not torch.allclose(first['event_count_logits'],second['event_count_logits'])
    loss,_ = transition_loss(net,second,**values,objective='unified')
    assert torch.isfinite(loss)


def test_incomplete_windows_not_count_negatives_and_overflow_reported():
    values = batch(); values['targets']['address_valid'][:,3,0] = False
    net = network(); out = net(values['inputs'])
    loss,metrics = transition_loss(net,out,**values,objective='unified')
    assert metrics['incomplete_event_windows'] == 1
    assert metrics['supervised_event_windows'] == 1
    loss.backward()
    values = batch(); net = network(event_queries=1)
    _,metrics = transition_loss(net,net(values['inputs']),**values,objective='unified')
    assert metrics['overflow_event_windows'] == 1


def test_all_unknown_voxels_and_one_resident_have_safe_null_candidate():
    values = batch(1); values['inputs']['voxel_known'].zero_()
    values['targets']['address'].fill_(2198)
    values['targets']['block_valid'].zero_();values['targets']['damage_valid'].zero_()
    net = network(); out = net(values['inputs'])
    loss,metrics = transition_loss(net,out,**values,objective='unified')
    loss.backward()
    assert np.isfinite(metrics['loss'])
    assert torch.isfinite(out['hidden']).all()
    assert (net.decode_interactions(out)['address'] == 2198).all()


def force_slots(net, out):
    # Actor 0: place at t0, attack actor 1 at t1, remove at t2. Actor 1: no writes.
    for key in ('event_count_logits','event_time_logits','event_operation_logits'):
        out[key] = torch.full_like(out[key],-40.)
    out['event_count_logits'][0,0,3]=40.;out['event_count_logits'][0,1,0]=40.
    out['event_time_logits'][0,0,0,0]=40.;out['event_time_logits'][0,0,1,1]=40.
    out['event_time_logits'][0,0,2,2]=40.
    out['event_operation_logits'][0,0,0,1]=40.;out['event_operation_logits'][0,0,1,2]=40.
    out['event_operation_logits'][0,0,2,0]=40.
    logits = out['event_address_logits']
    out['event_address_logits'] = torch.where(torch.isfinite(logits),-40.,logits)
    out['event_address_logits'][0,0,0,1098]=40.
    out['event_address_logits'][0,0,1,2198]=40.
    out['event_address_logits'][0,0,2,1098]=40.
    with torch.no_grad():
        net.block_head.weight.zero_();net.block_head.bias.copy_(torch.tensor([20.,10.,-10.]))
        net.damage_head.weight.zero_();net.damage_head.bias.fill_(1.)
    return out


def test_unified_decode_and_commit_both_voxel_and_hp_exactly_once():
    net = network().eval(); values = batch()
    out = force_slots(net,net(values['inputs']))
    decoded = net.decode_interactions(out)
    assert decoded['address'][0,:3,0].tolist() == [1098,2198,1098]
    assert decoded['block_payload'][0,0,0] == 1  # place must not emit air despite its higher logit
    assert decoded['block_payload'][0,2,0] == 0  # remove always air
    assert decoded['hp_payload'][0,1,0] < 0
    assert decoded['hp_payload'].count_nonzero() == 1
    memory = WorldMemory();memory.commit_points(np.array([[0,0,0]]),np.array([0]))
    rows = [CharRow(f'agent{i}',i,(0,0,0),0,0,20,0,0,(0,0,1.5),(0,1,0)) for i in range(2)]
    committer = TransitionCommitter(memory,rows)
    arrays = {k:v[0].detach().numpy() for k,v in decoded.items()}
    snapshots = committer.commit(resident_ids=['agent0','agent1'],anchors=np.zeros((2,3)),**arrays)
    assert len(committer.ledger) == 3
    assert snapshots[0]['agent1'].hp == 20
    assert snapshots[-1]['agent1'].hp == pytest.approx(20+float(decoded['hp_payload'][0,1,0]))
    assert memory.read_region([0,0,0],(1,1,1))[0].item() == 0


def test_same_frame_slots_resolve_once_and_operation_cannot_attack_a_voxel():
    net = network().eval(); out = force_slots(net,net(batch()['inputs']))
    out['event_time_logits'][0,0,1,:] = -40.;out['event_time_logits'][0,0,1,0] = 40.
    decoded = net.decode_interactions(out)
    assert (decoded['address'][0,:,0] != 2199).sum() == 2
    out['event_address_logits'][0,0,1,0] = 100.  # illegal voxel candidate for attack
    decoded = net.decode_interactions(out)
    assert decoded['address'][0,0,0] != 0


def test_closed_loop_dispatch_and_checkpoint_roundtrip():
    net = network().eval(); values = batch()['inputs']
    out = decode_transition(net,values)
    assert set(out) == {'pose','velocity','address','block_payload','hp_payload','held_item',
                        'camera_relative','camera_direction'}
    buffer = io.BytesIO();torch.save(dict(config=asdict(net.cfg),model=net.state_dict()),buffer)
    buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    saved['config']['kinematics'] = KinematicsConfig(**saved['config']['kinematics'])
    other = TransitionNetwork(TransitionArgs(**saved['config'])).eval()
    other.load_state_dict(saved['model'],strict=True)
    with torch.no_grad():
        torch.testing.assert_close(net(values)['pose'],other(values)['pose'])


def test_incompatible_flags_and_no_hp_objective_fail_explicitly():
    with pytest.raises(ValueError,match='replaces'):
        network(player_branch=True)
    with pytest.raises(ValueError,match='air_class'):
        network(air_class=None)
    net=network();values=batch()
    with pytest.raises(ValueError,match='no_hp'):
        transition_loss(net,net(values['inputs']),**values,objective='no_hp')
    with pytest.raises(ValueError,match='separate stream'):
        net.decode_attacks({})


def test_training_entry_synthetic_cpu_step_save_resume_and_evaluate(tmp_path,monkeypatch):
    from plot.data.transition_stream import TRANSITION_CACHE_REVISION
    source = Path(__file__).resolve().parents[1]/'train_scripts/train_m2_unified.py'
    spec = importlib.util.spec_from_file_location('train_m2_unified_test',source)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data = tmp_path/'data';data.mkdir()
    index = tmp_path/'index.json';vocab = tmp_path/'vocab.json';items_path = tmp_path/'items.json'
    items = {'':0,**{f'item{i}':i for i in range(1,10)}}
    items_path.write_text(json.dumps(items))
    index.write_text(json.dumps(dict(dataset_root=str(data),splits={
        'train':[dict(path='train/fixture_seed1')],
        'val_id':[dict(path='val_id/fixture_seed2')]})))
    vocab.write_text(json.dumps(dict(class_to_raw=[126,244,245])))
    cache = tmp_path/'cache';cache.mkdir()
    identity = dict(schema='m2-unified-cache-v1',items=items,dataset_root=str(data.resolve()),
                    index_sha256=hashlib.sha256(index.read_bytes()).hexdigest(),
                    vocabulary_sha256=hashlib.sha256(vocab.read_bytes()).hexdigest())
    (cache/'unified_cache_manifest.json').write_text(json.dumps(identity))
    for split,seed in (('train',1),('val_id',2)):
        folder=cache/split;folder.mkdir()
        example=sample();example['metadata'].update(episode=f'fixture_seed{seed}',start=0)
        torch.save(dict(samples=[example],transition_cache_revision=TRANSITION_CACHE_REVISION),
                   folder/f'fixture_seed{seed}.pt')
    out=tmp_path/'run1'
    argv=['train_m2_unified','--dataset-root',str(data),'--index',str(index),
          '--vocabulary',str(vocab),'--items',str(items_path),'--cache-root',str(cache),
          '--output-dir',str(out),'--steps','1','--batch-size','1','--workers','0',
          '--width','16','--depth','1','--heads','4','--event-context-frames','0',
          '--device','cpu','--save-every','1','--eval-batches','1']
    monkeypatch.setattr(sys,'argv',argv);module.main()
    assert (out/'checkpoint.pt').is_file() and (out/'COMPLETED.json').is_file()
    rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert {row['split'] for row in rows} == {'train','val_subset'}
    assert rows[0]['attack_labels'] == 1 and rows[0]['block_labels'] == 1
    argv[argv.index('--output-dir')+1]=str(tmp_path/'run2')
    argv[argv.index('--steps')+1]='2'
    argv+=['--resume',str(out/'checkpoint.pt')]
    monkeypatch.setattr(sys,'argv',argv);module.main()
    saved=torch.load(tmp_path/'run2/checkpoint.pt',map_location='cpu',weights_only=False)
    assert saved['step']==2 and saved['config']['model']['unified_interactions']
