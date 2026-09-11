"""Build a bounded, auditable M2 pilot cache from accepted training episodes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from plot.data.fill_dataset import BlockVocabulary
from plot.data.transition_dataset import episode_windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--scenarios',nargs='+',default=['S02','S06','S08','S09','S10'])
    p.add_argument('--episodes-per-scenario',type=int,default=2)
    p.add_argument('--windows-per-episode',type=int,default=8)
    p.add_argument('--vocabulary',help='existing M1 vocabulary to extend while preserving class indices')
    args=p.parse_args()
    if min(args.episodes_per_scenario,args.windows_per_episode)<1:p.error('positive sample limits required')
    selected=[];counts=Counter()
    for path in sorted(Path(args.root).glob('*/manifest.json')):
        # Do not read unrelated manifests from a large collection.
        if not any(f'_{s}_' in path.parent.name and counts[s]<args.episodes_per_scenario for s in args.scenarios):continue
        m=json.loads(path.read_text());s=m.get('scenario_id')
        if s not in args.scenarios or counts[s]>=args.episodes_per_scenario:continue
        v=path.with_name('validation.json')
        if m.get('split')!='train' or not v.exists() or not json.loads(v.read_text()).get('usable'):continue
        selected.append((path.parent,m));counts[s]+=1
        if all(counts[s]>=args.episodes_per_scenario for s in args.scenarios):break
    if not selected:raise ValueError('no accepted episodes matched')
    raw_ids=set();item_names={''}
    for path,m in selected:
        print('indexing',path.name,flush=True)
        with np.load(path/m.get('training_data_file','data.npz'),allow_pickle=False) as d:
            raw_ids.update(int(v) for v in np.unique(d['obs_voxel_mt'][...,0]))
            rows=[list(d['entity_id'].astype(str)).index(f'agent{i}') for i in range(int(m['num_agents']))]
            item_names.update(x.split(' ')[0] for x in np.unique(d['entity_weapon_name'][:,rows].astype(str)))
        metadata=json.loads((path/'training_metadata.json').read_text())
        item_names.update(metadata['item_vocabulary'])
    initial=BlockVocabulary.load(args.vocabulary).class_to_raw if args.vocabulary else ()
    classes=initial+tuple(sorted(raw_ids-set(initial)))
    vocabulary=BlockVocabulary(classes);items={name:i for i,name in enumerate(sorted(item_names))}
    samples=[];stats=Counter();episodes=[]
    for path,m in selected:
        print('materializing',path.name,flush=True)
        windows,audit=episode_windows(path,vocabulary,items,windows_per_episode=args.windows_per_episode)
        samples.extend(windows);stats.update(audit)
        episodes.append({'path':str(path),'scenario':m['scenario_id'],'agents':m['num_agents'],
                         'kinds':m['agent_kinds'],'windows':len(windows),'audit':dict(audit)})
    if not samples:raise ValueError('selected episodes contain no valid M2 windows')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={'split':'train','purpose':'bounded pilot, not a dataset-wide estimate',
            'episodes':episodes,'samples':len(samples),'audit':dict(stats),'class_to_raw':classes,
            'items':items,'hp_label_assumption':'single-source observed net HP delta; multiple sources masked',
            'vocabulary_note':'extended pilot vocabulary; use this exact mapping for any M1/M3 integration'}
    torch.save({'samples':samples,'report':report},output)
    output.with_suffix('.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps({'samples':len(samples),'audit':dict(stats)},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
