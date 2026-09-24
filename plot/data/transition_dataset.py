"""Small cached M2 windows with explicit masks for unrepresentable events."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .fill_dataset import TextAgentFillDataset, _read_video_frames


VOXEL_COUNT = 13**3
EVENT_CONTEXT_FRAMES = 4
PAYLOAD_CONFIRMATION_STATES = 8
BLOCK_EVENTS = {'block_placed','block_dug','scaffold_placed','scaffold_dug'}
TYPES = {'human_like':0,'npc_villager':1,'npc_zombie':2,'npc_skeleton':3}


def event_position(event):
    p = event.get('position')
    if not isinstance(p,dict) or not all(k in p for k in ('x','y','z')):
        return None
    # Lua events use engine XYZ; dense arrays are already ENU.
    return np.asarray([p['x'],p['z'],p['y']],np.int64)


def source_slot(event, agents):
    source = event.get('source') or event.get('actor')
    return agents.index(source) if source in agents else None


def aligned_events(events, actions):
    result = defaultdict(list)
    for e in events:
        if e.get('initialization_event') or e.get('event') not in BLOCK_EVENTS | {'damage'}:
            continue
        t = e.get('transition_index')
        if t is None or e.get('observation_frame') != int(t)+1 or not 0 <= int(t) < actions:
            raise ValueError('effective event has missing/inconsistent transition alignment')
        result[int(t)].append(e)
    return result


def _confirmed_block_payload(events_by_step, event_step, event, xyz, raw, centers, stats):
    """Return the first committed block state before another edit at this voxel."""
    placing=event['event'] in {'block_placed','scaffold_placed'}
    next_edit=min((step for step,records in events_by_step.items() if step>event_step
                   and any((position:=event_position(record)) is not None
                           and np.array_equal(position,xyz) for record in records)),
                  default=len(raw))
    final_state=min(len(raw)-1,event_step+PAYLOAD_CONFIRMATION_STATES,next_edit)
    for state in range(event_step+1,final_state+1):
        observed=set()
        for resident in range(centers.shape[1]):
            source_local=xyz-(centers[state,resident]-24)
            if np.all(source_local>=0) and np.all(source_local<49):
                observed.add(int(raw[(state,resident,*source_local,0)]))
        compatible={raw_id for raw_id in observed
                    if (placing and raw_id!=126) or (not placing and raw_id==126)}
        if len(compatible)==1:
            delay=state-(event_step+1)
            if delay:
                stats['payload_confirmed_after_delay']+=1
                stats['payload_confirmation_delay_steps']+=delay
            return next(iter(compatible))
        if len(compatible)>1:
            stats['conflicting_payload_observations']+=1
        elif observed:
            stats['stale_payload_observations']+=1
    return None


def build_write_labels(events_by_step, start, agents, anchors, health, raw, centers, vocabulary, strict_block_payload=False):
    a = len(agents); null = VOXEL_COUNT+a
    address = np.full((8,a),null,np.int64)
    valid = np.ones((8,a),bool)
    block = np.zeros((8,a),np.int64); block_valid = np.zeros((8,a),bool)
    damage = np.zeros((8,a),np.float32); damage_valid = np.zeros((8,a),bool)
    stats = Counter()
    for k in range(8):
        events = events_by_step.get(start+k,[])
        by_source = defaultdict(list); by_target = defaultdict(list); by_voxel = Counter()
        for e in events:
            slot = source_slot(e,agents)
            if slot is None:
                stats['unattributed_events'] += 1
                # An unknown source could be any resident; do not turn it into null negatives.
                valid[k,:] = False
                continue
            by_source[slot].append(e)
            if e['event']=='damage': by_target[e.get('target')].append(e)
            elif (xyz := event_position(e)) is not None: by_voxel[tuple(xyz)] += 1
        for slot,records in by_source.items():
            if len(records)!=1:
                valid[k,slot]=False;stats['multi_write_queries']+=1;continue
            e=records[0]
            if e['event']=='damage':
                target=e.get('target')
                if target not in agents or target==agents[slot]:
                    valid[k,slot]=False;stats['unrepresented_resident_targets']+=1;continue
                j=agents.index(target);address[k,slot]=VOXEL_COUNT+j
                delta=float(health[start+k+1,j]-health[start+k,j])
                stats['damage_events']+=1
                if len(by_target[target])!=1:
                    stats['ambiguous_damage_payloads']+=1
                elif not np.isfinite(delta) or delta>=0:
                    valid[k,slot]=False;stats['damage_without_observed_hp_loss']+=1
                else:
                    # Observed net change under a single-source assumption, not nominal damage.
                    damage[k,slot]=delta;damage_valid[k,slot]=True
                    stats['single_source_hp_labels']+=1
                    if abs(delta+float(e.get('damage',0)))>1e-5:
                        stats['nominal_damage_differs_from_hp']+=1
            else:
                xyz=event_position(e)
                if xyz is None:
                    valid[k,slot]=False;stats['missing_voxel_position']+=1;continue
                local=xyz-(anchors[slot]-6)
                if np.any(local<0) or np.any(local>=13):
                    valid[k,slot]=False;stats['voxel_outside_13_cube']+=1;continue
                address[k,slot]=np.ravel_multi_index(tuple(local),(13,13,13))
                if by_voxel[tuple(xyz)]!=1:
                    stats['ambiguous_block_payloads']+=1;continue
                raw_id=_confirmed_block_payload(
                    events_by_step,start+k,e,xyz,raw,centers,stats)
                if raw_id is not None:
                    encoded,known=vocabulary.encode(np.asarray(raw_id))
                    block[k,slot]=int(encoded);block_valid[k,slot]=bool(known)
                else:stats['event_observation_payload_mismatch']+=1
                if not block_valid[k,slot]:stats['unknown_block_payloads']+=1
    stats['queries']=8*a;stats['null_labels']=int(((address==null)&valid).sum())
    return {'address':address,'address_valid':valid,'block':block,'block_valid':block_valid,
            'damage':damage,'damage_valid':damage_valid},stats


def extract_local(raw, centers, anchors, vocabulary):
    a=len(anchors); out=np.zeros((a,13,13,13),np.int64);known=np.zeros_like(out,bool)
    offsets=np.stack(np.meshgrid(*([np.arange(13)]*3),indexing='ij'),-1)
    for target in range(a):
        coordinates=anchors[target]-6+offsets
        for source in range(a):
            local=coordinates-(centers[source]-24)
            covered=((local>=0)&(local<49)).all(-1)
            ix=local[covered]
            values,valid=vocabulary.encode(raw[source,ix[:,0],ix[:,1],ix[:,2],0])
            if (known[target][covered]&valid&(out[target][covered]!=values)).any():
                raise ValueError('resident voxel observations disagree at the same world coordinates')
            previous=out[target][covered];previous[valid]=values[valid]
            out[target][covered]=previous;known[target][covered]|=valid
    return out,known,anchors[:,None,:]-6+offsets.reshape(1,-1,3)


def load_episode_arrays(path, manifest):
    fields=('obs_voxel_mt','obs_voxel_center','player_pos','player_yaw','player_pitch',
            'player_health','player_health_valid','cam_pos','cam_dir','entity_id','entity_weapon_name',
            'action_continuous','action_source','termination_flag','truncation_flag','active_agent_mask',
            'selected_slot','inventory_item_ids')
    with np.load(path/manifest.get('training_data_file','data.npz'),allow_pickle=False) as data:
        return {k:data[k] for k in fields}


def episode_windows(path, vocabulary, items, *, windows_per_episode=8, attack_only=False, split="train", stride=1, strict_block_payload=False):
    """Cache materialization, so training never repeatedly decompresses whole episodes."""
    path=Path(path);manifest=json.loads((path/'manifest.json').read_text())
    if manifest.get('split')!=split or not json.loads((path/'validation.json').read_text()).get('usable'):
        raise ValueError('episode must be usable and match the requested split')
    data=load_episode_arrays(path,manifest)
    a=int(manifest['num_agents']);agents=[f'agent{i}' for i in range(a)]
    health=data['player_health'];action=data['action_continuous'];n=len(action)
    if health.shape!=(n+1,a) or action.shape!=(n,a,23):raise ValueError('expected T+1 states and T x A x 23 actions')
    events=[json.loads(line) for line in (path/manifest.get('event_file','events.jsonl')).read_text().splitlines() if line.strip()]
    by_step=aligned_events(events,n)
    first=TextAgentFillDataset._model_start(data,manifest)
    eligible=[]
    for start in range(first,n-7):
        if np.asarray(data['termination_flag'][start:start+7]).any() or np.asarray(data['truncation_flag'][start:start+7]).any():continue
        if not data['player_health_valid'][start:start+9].all() or not np.isfinite(health[start:start+9]).all():continue
        if not data['active_agent_mask'][start:start+8].all():continue
        count=sum(len(by_step.get(t,[])) for t in range(start,start+8))
        if attack_only and not any(e['event']=='damage' for t in range(start,start+8) for e in by_step.get(t,[])):continue
        eligible.append((start,count))
    # Include attack-centred and ordinary windows; this is explicitly pilot sampling.
    positives=[s for s,c in eligible if c];negatives=[s for s,c in eligible if not c]
    def spread(values,count):
        return [values[i] for i in np.linspace(0,len(values)-1,min(count,len(values)),dtype=int)] if values else []
    limit=windows_per_episode or 8
    starts=sorted(set(spread(positives,limit if attack_only else max(1,limit//2))+
                      ([] if attack_only else spread(negatives,limit-max(1,limit//2)))))
    if windows_per_episode is None:
        starts=[s for s,_ in eligible if (s-first)%stride==0]
        # A strided stream must still cover the last valid transitions.  Without
        # this terminal window, up to stride-1 real events silently disappear.
        if eligible and eligible[-1][0] not in starts:
            starts.append(eligible[-1][0])
    metadata=json.loads((path/'training_metadata.json').read_text())
    raw_items={int(v):k for k,v in metadata['item_vocabulary'].items()}
    rows=[list(data['entity_id'].astype(str)).index(x) for x in agents]
    weapons=data['entity_weapon_name'][:,rows].astype(str)
    held_names=np.asarray([[name.split(' ')[0] for name in row] for row in weapons])
    held=np.asarray([[items.get(name,0) for name in row] for row in held_names],np.int64)
    held_valid=np.asarray([[name in items for name in row] for row in held_names],bool)
    pose=np.concatenate((data['player_pos'],np.deg2rad(np.stack((data['player_yaw'],data['player_pitch']),-1))),-1).astype(np.float32)
    videos=manifest.get('agent_video_files') or [f'rgb_agent{i}.mp4' for i in range(a)]
    rgb = [_read_video_frames(path/videos[i], starts, (64,64)) for i in range(a)]
    stats=Counter();samples=[]
    for start in starts:
        anchors=np.floor(data['player_pos'][start]+.5).astype(np.int64)
        try:
            voxels,known,xyz=extract_local(data['obs_voxel_mt'][start],data['obs_voxel_center'][start],anchors,vocabulary)
        except ValueError as error:
            if str(error)!='resident voxel observations disagree at the same world coordinates':raise
            stats['skipped_inconsistent_voxel_windows']+=1
            continue
        labels,counts=build_write_labels(by_step,start,agents,anchors,health,data['obs_voxel_mt'],data['obs_voxel_center'],vocabulary,strict_block_payload=strict_block_payload)
        stats.update(counts)
        # A missing voxel candidate is an invalid address label, not a null.
        for k,i in np.argwhere(labels['address']<VOXEL_COUNT):
            if not known[i].reshape(-1)[labels['address'][k,i]]:labels['address_valid'][k,i]=False
        hotbar=data['inventory_item_ids'][start]
        mapped=np.asarray([[items.get(raw_items.get(int(v),''),0) for v in row] for row in hotbar],np.int64)
        # The legacy integer item dictionary omits iron swords. Episode loadout
        # metadata is authoritative and available at initialization, unlike future items.
        for slot in range(a):
            weapon=(manifest.get('agent_weapons') or {}).get(agents[slot],{}).get('weapon_item')
            if weapon:
                if weapon not in items:raise ValueError(f'unknown loadout item: {weapon}')
                mapped[slot]=0;mapped[slot,0]=items[weapon]
        same_hotbar=bool((data['inventory_item_ids'][start:start+8]==hotbar).all())
        conditions={'voxels':voxels,'voxel_known':known,'voxel_relative_xyz':(xyz-pose[start,:,None,:3]).astype(np.float32),
                    'initial_pose':pose[start],'initial_hp':health[start].astype(np.float32),'held_item':held[start],
                    'resident_type':np.asarray([TYPES[manifest['agent_kinds'][x]] for x in agents],np.int64),
                    'camera_relative':(data['cam_pos'][start]-pose[start,:,:3]).astype(np.float32),
                    'camera_direction':data['cam_dir'][start].astype(np.float32),'actions':action[start:start+8].astype(np.float32),
                    'active':data['active_agent_mask'][start].astype(bool),
                    'previous_rgb':np.stack([rgb[i][start] for i in range(a)]),
                    'hotbar':mapped,'selected_slot':data['selected_slot'][start].astype(np.int64)}
        context_indices=np.arange(start-EVENT_CONTEXT_FRAMES,start)
        context_valid=context_indices>=0
        context_indices=np.clip(context_indices,0,n-1)
        conditions.update(
            event_context_actions=action[context_indices].astype(np.float32),
            event_context_pose=pose[context_indices].astype(np.float32),
            event_context_hp=health[context_indices].astype(np.float32),
            event_context_held_item=held[context_indices],
            event_context_camera_relative=(data['cam_pos'][context_indices]
                                           -pose[context_indices,:,:3]).astype(np.float32),
            event_context_camera_direction=data['cam_dir'][context_indices].astype(np.float32),
            event_context_valid=(context_valid[:,None]
                                 &data['active_agent_mask'][context_indices].astype(bool)))
        if not context_valid.all():
            for key in ('event_context_actions','event_context_pose','event_context_hp',
                        'event_context_held_item','event_context_camera_relative',
                        'event_context_camera_direction'):
                conditions[key][~context_valid]=0
        labels.update(pose=pose[start+1:start+9],hp=health[start+1:start+9].astype(np.float32),
                      camera_relative=(data['cam_pos'][start+1:start+9]-pose[start+1:start+9,:,:3]).astype(np.float32),
                      camera_direction=data['cam_dir'][start+1:start+9].astype(np.float32),
                      held_item=held[start+1:start+9],held_item_valid=held_valid[start+1:start+9],
                      state_valid=(health[start:start+8]>0))
        explained=np.zeros((8,a),np.float32)
        for k,source in np.argwhere(labels['damage_valid']):
            explained[k,labels['address'][k,source]-VOXEL_COUNT]+=labels['damage'][k,source]
        hp_residual=health[start+1:start+9]-health[start:start+8]-explained
        stats['unexplained_positive_hp_steps']+=int(((hp_residual>1e-5)&labels['state_valid']).sum())
        stats['unexplained_negative_hp_steps']+=int(((hp_residual< -1e-5)&labels['state_valid']).sum())
        stats['unknown_input_voxels']+=int((~known).sum());stats['input_voxels']+=known.size
        samples.append({'inputs':{k:torch.from_numpy(np.asarray(v)) for k,v in conditions.items()},
                        'targets':{k:torch.from_numpy(np.asarray(v)) for k,v in labels.items()},
                        'metadata':{'episode':str(path),'start':start,'anchors':anchors.tolist(),
                                    'fixed_hotbar':same_hotbar,'scenario':manifest['scenario_id']}})
    return samples,stats


class TransitionDataset(Dataset):
    def __init__(self,path):
        self.cache=torch.load(path,map_location='cpu',weights_only=True)
        self.samples=self.cache['samples']
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):return self.samples[index]


def collate_transition(samples):
    a=max(s['inputs']['active'].numel() for s in samples)
    output={'inputs':{},'targets':{},'metadata':[s['metadata'] for s in samples]}
    for group in ('inputs','targets'):
        for key in samples[0][group]:
            values=[]
            for s in samples:
                x=s[group][key].clone(); old=s['inputs']['active'].numel()
                if group=='targets' and key=='address':x[x==VOXEL_COUNT+old]=VOXEL_COUNT+a
                axis=1 if (group=='targets' or key=='actions'
                           or key.startswith('event_context_')) else 0
                if old<a:
                    shape=list(x.shape);shape[axis]=a-old
                    padding=x.new_zeros(shape)
                    if group=='targets' and key=='address':padding.fill_(VOXEL_COUNT+a)
                    x=torch.cat((x,padding),axis)
                values.append(x)
            output[group][key]=torch.stack(values)
    return output
