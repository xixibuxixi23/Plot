import numpy as np
import torch

from plot.data.fill_dataset import BlockVocabulary
from plot.data.edit_process import derive_edit_process
from plot.data.transition_dataset import build_write_labels,collate_transition,event_position
from plot.models.transition import TransitionArgs,TransitionNetwork
from plot.pipelines.transition_pipeline import TransitionCommitter,held_item_trajectory
from plot.transition_state import CharRow
from plot.world_memory import WorldMemory
from plot.training.transition_trainer import transition_loss


torch.set_num_threads(2)


def test_edit_process_accumulates_across_windows_and_keeps_delayed_completion():
    actions=np.zeros((12,2,23),np.float32)
    actions[3:6,0,8]=1
    pointed_type=np.ones((13,2),np.int64)
    under=np.zeros((13,2,3),np.int32);above=np.zeros_like(under)
    under[:,0]=[4,7,2]
    dt=np.full((13,2),.1)
    event={'event':'block_dug','actor':'agent0','position':{'x':4,'y':2,'z':7}}
    trace=derive_edit_process(actions,pointed_type,under,above,dt,{6:[event]},['agent0','agent1'])
    np.testing.assert_array_equal(trace['age_frames'][3:7,0],[1,2,3,3])
    np.testing.assert_allclose(trace['progress'][3:7,0],[1/3,2/3,1.,1.])
    assert trace['completed'][6,0] and trace['kind'][6,0]==1
    assert not trace['target_valid'][:3,0].any()


def test_edit_process_resets_when_target_changes_and_does_not_label_failure_progress():
    actions=np.zeros((5,1,23),np.float32);actions[:,0,8]=1
    pointed_type=np.ones((6,1),np.int64)
    under=np.zeros((6,1,3),np.int32);under[2:,0]=[1,0,0]
    trace=derive_edit_process(actions,pointed_type,under,np.zeros_like(under),np.ones((6,1))*.1,{},['agent0'])
    np.testing.assert_array_equal(trace['age_frames'][:,0],[1,2,1,2,3])
    assert not trace['progress_valid'].any()


def sample(a=2):
    inputs={
        'active':torch.ones(a,dtype=torch.bool),'voxels':torch.zeros(a,13,13,13,dtype=torch.long),
        'voxel_known':torch.ones(a,13,13,13,dtype=torch.bool),
        'voxel_relative_xyz':torch.randn(a,2197,3), 'initial_pose':torch.randn(a,5),
        'initial_hp':torch.ones(a)*20,'held_item':torch.zeros(a,dtype=torch.long),
        'resident_type':torch.zeros(a,dtype=torch.long),'camera_relative':torch.rand(a,3),
        'camera_direction':torch.tensor([0.,1.,0.]).expand(a,3).clone(),
        'actions':torch.zeros(8,a,23),'previous_rgb':torch.rand(a,3,32,32),
        'hotbar':torch.arange(9)[None].expand(a,-1).clone(),'selected_slot':torch.zeros(a,dtype=torch.long),
    }
    targets={'address':torch.full((8,a),2197+a,dtype=torch.long),'address_valid':torch.ones(8,a,dtype=torch.bool),
             'block':torch.zeros(8,a,dtype=torch.long),'block_valid':torch.zeros(8,a,dtype=torch.bool),
             'damage':torch.zeros(8,a),'damage_valid':torch.zeros(8,a,dtype=torch.bool),
             'pose':torch.randn(8,a,5),'hp':torch.ones(8,a)*20,
             'camera_relative':torch.rand(8,a,3),'camera_direction':torch.tensor([0.,1.,0.]).expand(8,a,3).clone(),
             'state_valid':torch.ones(8,a,dtype=torch.bool),'held_item':torch.zeros(8,a,dtype=torch.long)}
    targets['address'][0,0]=2198;targets['damage'][0,0]=-1;targets['damage_valid'][0,0]=True
    targets['address'][1,0]=0;targets['block'][1,0]=1;targets['block_valid'][1,0]=True
    return {'inputs':inputs,'targets':targets,'metadata':{'fixed_hotbar':True}}


def add_edit_process_targets(value):
    shape=value['targets']['address'].shape
    value['targets'].update(
        edit_kind=torch.zeros(shape,dtype=torch.long),
        edit_age_frames=torch.zeros(shape,dtype=torch.int16),
        edit_age_seconds=torch.zeros(shape),edit_progress=torch.zeros(shape),
        edit_progress_valid=torch.zeros(shape,dtype=torch.bool),
        edit_completed=torch.zeros(shape,dtype=torch.bool),
        edit_target=torch.zeros(shape,dtype=torch.long),
        edit_target_valid=torch.zeros(shape,dtype=torch.bool))
    value['targets']['edit_kind'][1:3,0]=1
    value['targets']['edit_age_frames'][1:3,0]=torch.tensor([1,2])
    value['targets']['edit_progress'][1:3,0]=torch.tensor([.5,1.])
    value['targets']['edit_progress_valid'][1:3,0]=True
    value['targets']['edit_target'][1:3,0]=10
    value['targets']['edit_target_valid'][1:3,0]=True
    return value


def model():
    torch.manual_seed(8)
    return TransitionNetwork(TransitionArgs(3,10,width=32,depth=2,heads=4))


def test_two_and_four_players_batch_padding_has_no_effect():
    net=model().eval();two=sample(2);four=sample(4)
    single=collate_transition([two]);mixed=collate_transition([two,four])
    assert mixed['targets']['address'][0,-1,0]==2201
    assert not mixed['inputs']['active'][0,2:].any()
    with torch.no_grad():
        first=net(single['inputs']);second=net(mixed['inputs'])
    torch.testing.assert_close(first['pose'][0],second['pose'][0,:,:2],atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(first['address_logits'][0,:,:,:2197],second['address_logits'][0,:,:2,:2197],atol=1e-5,rtol=1e-5)
    loss,metrics=transition_loss(net,net(mixed['inputs']),mixed['inputs'],mixed['targets'])
    loss.backward()
    assert np.isfinite(metrics['loss'])
    assert net.pointer_query.weight.grad.abs().sum()>0
    assert net.damage_head.weight.grad.abs().sum()>0
    assert net.block_head.weight.grad.abs().sum()>0


def test_permutation_equivariance_and_world_translation():
    net=model().eval();batch=collate_transition([sample(4)])['inputs'];perm=torch.tensor([2,0,3,1])
    reordered={k:v.index_select(2 if k=='actions' else 1,perm) for k,v in batch.items()}
    with torch.no_grad():
        out=net(batch);other=net(reordered)
    torch.testing.assert_close(out['pose'][:,:,perm],other['pose'],atol=1e-5,rtol=1e-5)
    columns=torch.cat((torch.arange(2197),2197+perm,torch.tensor([2201])))
    torch.testing.assert_close(out['address_logits'][:,:,perm][:,:,:,columns],other['address_logits'],atol=1e-5,rtol=1e-5)
    batch['initial_pose'][...,:3]+=torch.tensor([100.,-20.,7.])
    with torch.no_grad():translated=net(batch)
    torch.testing.assert_close(translated['address_logits'],out['address_logits'],atol=1e-5,rtol=1e-5)


def test_future_actions_can_affect_earlier_queries_and_no_cache():
    net=model().eval();inputs=collate_transition([sample()])['inputs']
    with torch.no_grad():
        first=net(inputs)['address_logits']
        inputs['actions'][:,-1,:,0]=1
        second=net(inputs)['address_logits']
    assert not torch.allclose(first[:,0],second[:,0])
    assert not hasattr(net,'kv_caches')


def test_damage_uses_observed_hp_and_ambiguous_payloads_are_masked():
    raw=np.zeros((9,3,49,49,49,2),np.int16);centers=np.zeros((9,3,3),np.int64)
    hp=np.full((9,3),20.,np.float32);hp[1:,1]=19
    single={'event':'damage','source':'agent0','target':'agent1','damage':2}
    labels,stats=build_write_labels({0:[single]},0,['agent0','agent1','agent2'],np.zeros((3,3)),hp,raw,centers,BlockVocabulary((0,1)))
    assert labels['damage'][0,0]==-1 and labels['damage_valid'][0,0]
    assert stats['nominal_damage_differs_from_hp']==1
    other=dict(single,source='agent2')
    labels,stats=build_write_labels({0:[single,other]},0,['agent0','agent1','agent2'],np.zeros((3,3)),hp,raw,centers,BlockVocabulary((0,1)))
    assert labels['address_valid'][0,[0,2]].all()
    assert not labels['damage_valid'][0].any()


def test_multiwrite_and_outside_cube_are_not_null_negatives():
    raw=np.zeros((9,2,49,49,49,2),np.int16);centers=np.zeros((9,2,3),np.int64)
    hp=np.full((9,2),20.)
    event={'event':'block_dug','actor':'agent0','position':{'x':20,'y':0,'z':0}}
    labels,_=build_write_labels({0:[event],1:[event,event]},0,['agent0','agent1'],np.zeros((2,3)),hp,raw,centers,BlockVocabulary((0,1)))
    assert not labels['address_valid'][:2,0].any()


def test_commit_last_voxel_write_and_sum_damage_once():
    memory=WorldMemory();rows=[CharRow(f'agent{i}',i,(0,0,0),0,0,20,0,0,(0,0,1.5),(0,1,0)) for i in range(3)]
    memory.commit_points(np.asarray([[0,0,0]]),np.asarray([0]))
    runner=TransitionCommitter(memory,rows);address=np.full((8,3),2200,np.int64)
    address[0,:2]=1098;address[1,:2]=2199
    blocks=np.zeros((8,3),np.int64);blocks[0,:2]=[1,2]
    damage=np.zeros((8,3));damage[1,:2]=[-2,-3]
    captured=[]
    snapshots=runner.commit(resident_ids=[f'agent{i}' for i in range(3)],anchors=np.zeros((3,3)),
                            pose=np.zeros((8,3,5)),address=address,block_payload=blocks,hp_payload=damage,
                            held_item=np.zeros((8,3)),camera_relative=np.zeros((8,3,3)),
                            camera_direction=np.tile([0,1,0],(8,3,1)),
                            after_step=lambda t,m,c,e:captured.append((t,c['agent2'].hp)))
    assert memory.read_region([0,0,0],(1,1,1))[0].item()==2
    assert snapshots[0]['agent2'].hp==20 and snapshots[1]['agent2'].hp==15
    assert len(runner.ledger)==4 and captured==[(0,20)]+[(i,15) for i in range(1,8)]
    assert memory.read_region([-6,-6,-6],(13,13,13))[0].shape==(13,13,13)


def test_fixed_loadout_adapter_and_drop_validity():
    inputs=collate_transition([sample()])['inputs']
    inputs['actions'][:,0,:,14]=1  # slot_3
    inputs['actions'][:,3,:,10]=1  # drop
    items,valid=held_item_trajectory(inputs)
    assert (items==2).all() and valid[:,:3].all() and not valid[:,3:].any()


def test_textagent_event_coordinates_and_yaw_use_the_same_enu_convention():
    from plot.kinematics import propose_trajectory,KinematicsConfig
    np.testing.assert_array_equal(event_position({'position':{'x':-129,'y':16,'z':56}}),[-129,56,16])
    initial=torch.zeros(1,1,5);initial[...,3]=-torch.pi/2
    actions=torch.zeros(1,8,1,23);actions[...,0]=1
    proposal=propose_trajectory(initial,actions,KinematicsConfig(distance_per_step=1.))
    torch.testing.assert_close(proposal[0,0,0,:3],torch.tensor([-1.,0.,0.]),atol=1e-6,rtol=1e-6)


def test_edit_objective_ignores_hp_and_motion_supervision():
    net=model();batch=collate_transition([sample()])
    # Construction-only labels; the resident attack is a null in this fixture.
    batch['targets']['address'][0,0,0]=2199
    batch['targets']['damage_valid'].zero_()
    out=net(batch['inputs'])
    first,metrics=transition_loss(net,out,**{k:batch[k] for k in ('inputs','targets')},objective='edit')
    batch['targets']['hp'].add_(10)
    batch['targets']['pose'].add_(100)
    second,_=transition_loss(net,out,**{k:batch[k] for k in ('inputs','targets')},objective='edit')
    torch.testing.assert_close(first,second)
    first.backward()
    assert net.damage_head.weight.grad is None
    assert net.block_head.weight.grad.abs().sum()>0
    assert net.pointer_query.weight.grad.abs().sum()>0
    assert metrics['block_labels']==1


def test_no_hp_objective_trains_geometry_and_inventory_without_health():
    net=model();batch=collate_transition([sample()]);out=net(batch['inputs'])
    args={k:batch[k] for k in ('inputs','targets')}
    loss,_=transition_loss(net,out,**args,objective='no_hp')
    batch['targets']['hp'].add_(10);batch['targets']['damage'].sub_(7)
    other,_=transition_loss(net,out,**args,objective='no_hp')
    torch.testing.assert_close(loss,other)
    loss.backward()
    for head in (net.pose_head,net.camera_head,net.held_head,net.block_head,
                 net.event_count_head[-1],net.event_time_head,net.event_pointer_query):
        assert head.weight.grad.abs().sum()>0
    assert net.occurrence_head.weight.grad is None
    assert net.damage_head.weight.grad is None and net.hp_aux.weight.grad is None


def test_player_objective_only_supervises_pose_camera_and_inventory():
    net=model();batch=collate_transition([sample()]);out=net(batch['inputs'])
    args={k:batch[k] for k in ('inputs','targets')}
    loss,_=transition_loss(net,out,**args,objective='player')
    batch['targets']['hp'].add_(10);batch['targets']['damage'].sub_(7)
    batch['targets']['address'].fill_(0);batch['targets']['block'].add_(1)
    other,_=transition_loss(net,out,**args,objective='player')
    torch.testing.assert_close(loss,other)
    loss.backward()
    for head in (net.player_adapter[-1],net.pose_head,net.camera_head,net.held_head):
        assert head.weight.grad.abs().sum()>0
    for head in (net.event_count_head[-1],net.event_time_head,net.event_pointer_query,
                 net.block_head,net.damage_head,net.hp_aux):
        assert head.weight.grad is None


def test_lightweight_player_branch_supports_mixed_resident_slots():
    cfg=TransitionArgs(3,10,width=32,depth=2,heads=4,player_branch=True,player_depth=2)
    net=TransitionNetwork(cfg);batch=collate_transition([sample()]);output=net.forward_player(batch['inputs'])
    assert output['pose'].shape==(1,8,2,5)
    assert output['camera_relative'].shape==(1,8,2,3)
    loss=(output['pose'].square().mean()+output['camera_relative'].square().mean()
          +output['held_logits'].square().mean())
    loss.backward()
    assert net.player_blocks[0].time.in_proj_weight.grad.abs().sum()>0
    assert net.event_time_head.weight.grad is None


def test_attack_branch_uses_ordered_slots_and_masks_self_targets():
    cfg=TransitionArgs(3,10,width=32,depth=2,heads=4,player_branch=True,player_depth=2,
                       attack_branch=True,attack_queries=2)
    net=TransitionNetwork(cfg);batch=collate_transition([sample(3)]);output=net.forward_player(batch['inputs'])
    assert output['attack_count_logits'].shape==(1,3,3)
    assert output['attack_time_logits'].shape==(1,3,2,8)
    assert output['attack_target_logits'].shape==(1,3,2,3)
    assert output['attack_damage'].shape==(1,3,2)
    diagonal=output['attack_target_logits'][0,torch.arange(3),:,torch.arange(3)]
    assert torch.isneginf(diagonal).all()


def test_ordered_event_slots_predict_count_and_one_time_per_slot():
    net=model().eval();batch=collate_transition([sample()]);out=net(batch['inputs'])
    assert out['event_count_logits'].shape==(1,2,5)
    assert out['event_time_logits'].shape==(1,2,4,8)
    assert out['event_address_logits'].shape==(1,2,4,2199)
    # A categorical time head assigns each slot to exactly one frame.
    selected=out['event_time_logits'].argmax(-1)
    assert selected.shape==(1,2,4)


def test_no_hp_objective_supervises_postprocessed_edit_process():
    net=model();batch=collate_transition([add_edit_process_targets(sample())]);out=net(batch['inputs'])
    loss,metrics=transition_loss(net,out,**{k:batch[k] for k in ('inputs','targets')},objective='no_hp')
    loss.backward()
    assert all(name in metrics for name in ('edit_kind','edit_age','edit_progress','edit_target'))
    for head in (net.edit_kind_head,net.edit_age_head,net.edit_progress_head):
        assert head.weight.grad.abs().sum()>0


def test_payload_uses_consistent_updated_view_when_first_resident_is_stale():
    raw=np.full((9,2,49,49,49,2),126,np.int16);centers=np.zeros((9,2,3),np.int64)
    raw[1,1,24,24,24,0]=244
    hp=np.full((9,2),20.)
    e={'event':'block_placed','actor':'agent0','position':{'x':0,'y':0,'z':0}}
    labels,_=build_write_labels({0:[e]},0,['agent0','agent1'],np.zeros((2,3),dtype=int),hp,raw,centers,BlockVocabulary((126,244)),strict_block_payload=True)
    assert labels['block_valid'][0,0] and labels['block'][0,0]==1
    raw[1,0,24,24,24,0]=245
    labels,_=build_write_labels({0:[e]},0,['agent0','agent1'],np.zeros((2,3),dtype=int),hp,raw,centers,BlockVocabulary((126,244,245)),strict_block_payload=True)
    assert not labels['block_valid'][0,0]


def test_stream_shards_cover_each_episode_once_across_ranks_and_workers(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    import plot.data.transition_stream as module
    records=[{'path':f'train/episode_{i}'} for i in range(71)]
    index=tmp_path/'index.json';index.write_text(json.dumps({'dataset_root':'/old/machine/path','splits':{'train':records}}))
    vocab=tmp_path/'vocab.json';vocab.write_text(json.dumps({'class_to_raw':[126,244]}))
    cache=tmp_path/'cache'/'train';cache.mkdir(parents=True)
    for i,r in enumerate(records):torch.save({'samples':[{'id':i}]},cache/(r['path'].split('/')[-1]+'.pt'))
    all_ids=[]
    for rank in range(8):
        for worker in range(4):
            monkeypatch.setattr(module,'get_worker_info',lambda w=worker:SimpleNamespace(id=w,num_workers=4))
            stream=module.TransitionStream(index,vocab,tmp_path/'cache','train',rank=rank,world_size=8,
                                           fixed_order=True,dataset_root=tmp_path)
            first=[s['id'] for s in stream];second=[s['id'] for s in stream]
            assert first==second
            all_ids.extend(first)
    assert sorted(all_ids)==list(range(71))


def test_sparse_video_reader_matches_single_frame_reader(tmp_path):
    import cv2
    from plot.data.fill_dataset import _read_video_frame,_read_video_frames
    path=tmp_path/'tiny.mp4';writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),10,(16,12))
    for value in range(6):writer.write(np.full((12,16,3),value*30,np.uint8))
    writer.release()
    frames=_read_video_frames(path,[0,3,5],(8,6))
    for index in frames:
        np.testing.assert_allclose(frames[index],_read_video_frame(path,index,(8,6)),atol=2/255)
