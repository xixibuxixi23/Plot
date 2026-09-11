"""Lightweight player-dynamics windows without video or voxel decompression."""
from __future__ import annotations

import json
from pathlib import Path
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset,get_worker_info

from .fill_dataset import TextAgentFillDataset
from .transition_dataset import TYPES


FIELDS=('player_pos','player_yaw','player_pitch','player_health','player_health_valid',
        'cam_pos','cam_dir','entity_id','entity_weapon_name','action_continuous',
        'action_source','termination_flag','truncation_flag','active_agent_mask','selected_slot','inventory_item_ids')


def episode_player_windows(path,items,split,stride=8):
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('split')!=split or not json.loads((path/'validation.json').read_text()).get('usable'):
        raise ValueError('episode must be usable and match split')
    with np.load(path/manifest.get('training_data_file','data.npz'),allow_pickle=False) as stored:
        data={key:stored[key] for key in FIELDS}
    actions=data['action_continuous'];n=len(actions);agents=int(manifest['num_agents'])
    rows=[list(data['entity_id'].astype(str)).index(f'agent{index}') for index in range(agents)]
    names=data['entity_weapon_name'][:,rows].astype(str)
    names=np.asarray([[value.split(' ')[0] for value in row] for row in names])
    held=np.asarray([[items.get(value,0) for value in row] for row in names],np.int64)
    held_valid=np.asarray([[value in items for value in row] for row in names],bool)
    pose=np.concatenate((data['player_pos'],np.deg2rad(np.stack((data['player_yaw'],data['player_pitch']),-1))),-1).astype(np.float32)
    metadata=json.loads((path/'training_metadata.json').read_text());raw_items={int(v):k for k,v in metadata['item_vocabulary'].items()}
    hotbar=data['inventory_item_ids'];mapped=np.asarray(
        [[[items.get(raw_items.get(int(value),''),0) for value in row] for row in frame] for frame in hotbar],np.int64)
    kinds=np.asarray([TYPES[manifest['agent_kinds'][f'agent{i}']] for i in range(agents)],np.int64)
    first=TextAgentFillDataset._model_start(data,manifest)
    agent_names=[f'agent{i}' for i in range(agents)]
    damage_by_step=defaultdict(list)
    event_file=path/manifest.get('event_file','events.jsonl')
    for line in event_file.read_text().splitlines():
        event=json.loads(line)
        if event.get('initialization_event') or event.get('event')!='damage':continue
        transition=event.get('transition_index')
        if transition is None or event.get('observation_frame')!=int(transition)+1:continue
        target=event.get('target')
        if target not in agent_names:continue
        source=event.get('source')
        if source in agent_names:source_slot=agent_names.index(source)
        elif agents==2:source_slot=1-agent_names.index(target)
        else:source_slot=None
        damage_by_step[int(transition)].append((source_slot,agent_names.index(target),event))
    for start in range(first,n-7,stride):
        if np.asarray(data['termination_flag'][start:start+7]).any() or np.asarray(data['truncation_flag'][start:start+7]).any():continue
        active=data['active_agent_mask'][start:start+8].astype(bool)
        state_valid=active&data['player_health_valid'][start:start+8]&np.isfinite(pose[start:start+8]).all(-1)
        state_valid&=np.isfinite(data['cam_pos'][start:start+8]).all(-1)&np.isfinite(data['cam_dir'][start:start+8]).all(-1)
        input_active=active[0]&data['player_health_valid'][start]&np.isfinite(pose[start]).all(-1)
        input_active&=np.isfinite(data['cam_pos'][start]).all(-1)&np.isfinite(data['cam_dir'][start]).all(-1)
        state_valid&=input_active[None]
        if not input_active.any() or not state_valid.any():continue
        inputs={'initial_pose':pose[start],'initial_hp':data['player_health'][start].astype(np.float32),
                'held_item':held[start],'resident_type':kinds,
                'camera_relative':(data['cam_pos'][start]-pose[start,:,:3]).astype(np.float32),
                'camera_direction':data['cam_dir'][start].astype(np.float32),
                'actions':actions[start:start+8].astype(np.float32),'active':input_active,
                'hotbar':mapped[start],'selected_slot':data['selected_slot'][start].astype(np.int64)}
        targets={'pose':pose[start+1:start+9],
                 'camera_relative':(data['cam_pos'][start+1:start+9]-pose[start+1:start+9,:,:3]).astype(np.float32),
                 'camera_direction':data['cam_dir'][start+1:start+9].astype(np.float32),
                 'held_item':held[start+1:start+9],'held_item_valid':held_valid[start+1:start+9],
                 'state_valid':state_valid}
        # Successful engine callbacks are packed chronologically into at most
        # two slots per source.  Actual HP delta is a separate masked payload:
        # a callback remains a positive attack even when invulnerability or a
        # terminal state makes the observed delta unusable.
        attack_count=np.zeros(agents,np.int64);attack_count_valid=input_active.copy()
        attack_time=np.zeros((agents,2),np.int64);attack_target=np.zeros((agents,2),np.int64)
        attack_damage=np.zeros((agents,2),np.float32)
        attack_slot_valid=np.zeros((agents,2),bool);attack_damage_valid=np.zeros((agents,2),bool)
        packed=[[] for _ in range(agents)]
        for transition in range(start,start+8):
            records=damage_by_step.get(transition,[])
            by_target=defaultdict(list)
            for record in records:by_target[record[1]].append(record)
            for source,target,event in records:
                if source is None:
                    attack_count_valid[:]=False;continue
                packed[source].append((transition-start,target,event,len(by_target[target])==1))
        for source,records in enumerate(packed):
            attack_count[source]=min(2,len(records))
            if len(records)>2:attack_count_valid[source]=False
            for slot,(when,target,event,unambiguous) in enumerate(records[:2]):
                attack_time[source,slot]=when;attack_target[source,slot]=target
                attack_slot_valid[source,slot]=True
                delta=float(data['player_health'][start+when+1,target]-data['player_health'][start+when,target])
                if unambiguous and np.isfinite(delta) and delta<0:
                    attack_damage[source,slot]=-delta;attack_damage_valid[source,slot]=True
        targets.update(attack_count=attack_count,attack_count_valid=attack_count_valid,
                       attack_time=attack_time,attack_target=attack_target,
                       attack_damage=attack_damage,attack_slot_valid=attack_slot_valid,
                       attack_damage_valid=attack_damage_valid)
        yield {'inputs':{key:torch.from_numpy(np.asarray(value)) for key,value in inputs.items()},
               'targets':{key:torch.from_numpy(np.asarray(value)) for key,value in targets.items()},
               'metadata':{'episode':str(path),'start':start,'scenario':manifest['scenario_id']}}


class PlayerTransitionStream(IterableDataset):
    def __init__(self,index,split,items,seed=7,stride=8,rank=0,world_size=1,fixed_order=False):
        manifest=json.loads(Path(index).read_text());self.root=Path(manifest['dataset_root'])
        self.records=manifest['splits'][split];self.split=split;self.items=dict(items);self.seed=seed
        self.stride=stride;self.rank=rank;self.world_size=world_size;self.fixed_order=fixed_order;self.epoch=0

    def __iter__(self):
        worker=get_worker_info();wid=worker.id if worker else 0;nworkers=worker.num_workers if worker else 1
        rng=random.Random(self.seed+(0 if self.fixed_order else self.epoch));order=list(range(len(self.records)));rng.shuffle(order);self.epoch+=1
        for index in order[self.rank*nworkers+wid::self.world_size*nworkers]:
            yield from episode_player_windows(self.root/self.records[index]['path'],self.items,self.split,self.stride)


def collate_player_transition(samples):
    agents=max(sample['inputs']['active'].numel() for sample in samples);output={'inputs':{},'targets':{},'metadata':[s['metadata'] for s in samples]}
    for group in ('inputs','targets'):
        for key in samples[0][group]:
            values=[]
            for sample in samples:
                value=sample[group][key];old=sample['inputs']['active'].numel()
                axis=(0 if key.startswith('attack_') else (1 if group=='targets' or key=='actions' else 0))
                if old<agents:
                    shape=list(value.shape);shape[axis]=agents-old
                    value=torch.cat((value,value.new_zeros(shape)),axis)
                values.append(value)
            output[group][key]=torch.stack(values)
    return output
