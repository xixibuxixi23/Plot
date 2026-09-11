"""Accepted episodes, lazily materialized with reusable per-episode caches."""
import json
import random
from pathlib import Path
import torch
from torch.utils.data import IterableDataset, get_worker_info
from .fill_dataset import BlockVocabulary
from .transition_dataset import episode_windows


class TransitionStream(IterableDataset):
    def __init__(self, index, vocabulary, cache_root, split, seed=7, stride=8, rank=0,
                 world_size=1, fixed_order=False, legacy_cache=None, edit_process_root=None,
                 items=None, dataset_root=None):
        manifest=json.loads(Path(index).read_text())
        self.root=Path(dataset_root) if dataset_root is not None else Path(manifest['dataset_root'])
        self.records=manifest['splits'][split]
        self.vocabulary=BlockVocabulary.load(vocabulary)
        # The S01 default remains backward compatible. Larger collections pass a
        # train-split vocabulary explicitly so unknown items are not mapped to empty.
        default_items={name:i for i,name in enumerate(['','mcl_core:brick_block','mcl_core:cobble',
            'mcl_core:dirt','mcl_core:glass','mcl_core:stonebrick','mcl_core:tree','mcl_core:wood',
            'mcl_ocean:sea_lantern','mcl_tools:pick_iron','textagent_task:npc_axe','textagent_task:npc_sword'])}
        self.items=dict(items) if items is not None else default_items
        if self.items.get('')!=0 or sorted(self.items.values())!=list(range(len(self.items))):
            raise ValueError('item vocabulary must be contiguous and reserve index 0 for empty')
        self.legacy_cache=Path(legacy_cache)/split if legacy_cache else None
        self.edit_process_root=Path(edit_process_root)/split if edit_process_root else None
        self.rank=rank;self.world_size=world_size;self.fixed_order=fixed_order
        self.split=split;self.seed=seed;self.stride=stride;self.epoch=0
        self.cache_root=Path(cache_root)/split
        self.cache_root.mkdir(parents=True,exist_ok=True)

    def __iter__(self):
        worker=get_worker_info();wid=worker.id if worker else 0;nworkers=worker.num_workers if worker else 1
        rng=random.Random(self.seed+(0 if self.fixed_order else self.epoch));order=list(range(len(self.records)));rng.shuffle(order)
        self.epoch+=1
        for i in order[self.rank*nworkers+wid::self.world_size*nworkers]:
            path=self.root/self.records[i]['path'];cache=self.cache_root/(path.name+'.pt')
            if cache.exists():
                result=torch.load(cache,map_location='cpu',weights_only=True)
            elif self.legacy_cache and (self.legacy_cache/cache.name).exists():
                from .transition_dataset import build_write_labels, aligned_events
                import numpy as np
                result=torch.load(self.legacy_cache/cache.name,map_location='cpu',weights_only=True)
                manifest=json.loads((path/'manifest.json').read_text())
                with np.load(path/manifest.get('training_data_file','data.npz')) as d:
                    raw=d['obs_voxel_mt'];centers=d['obs_voxel_center'];health=d['player_health']
                events=aligned_events([json.loads(l) for l in (path/'events.jsonl').read_text().splitlines() if l.strip()],len(health)-1)
                audit={}
                for sample in result['samples']:
                    meta=sample['metadata'];agents=[f'agent{j}' for j in range(len(meta['anchors']))]
                    labels,counts=build_write_labels(events,meta['start'],agents,np.asarray(meta['anchors']),health,
                                                    raw,centers,self.vocabulary,strict_block_payload=True)
                    for key in ('block','block_valid'):sample['targets'][key]=torch.from_numpy(labels[key])
                    for key,value in counts.items():audit[key]=audit.get(key,0)+value
                result['audit']=audit;result['payload_revision']=2
                temporary=cache.with_suffix(f'.{self.rank}.{wid}.tmp');torch.save(result,temporary);temporary.replace(cache)
                cache.with_suffix('.json').write_text(json.dumps({'episode':str(path),'windows':len(result['samples']),'audit':audit,'migrated_from':str(self.legacy_cache/cache.name)}))
            else:
                samples,audit=episode_windows(path,self.vocabulary,self.items,windows_per_episode=None,
                                              split=self.split,stride=self.stride,strict_block_payload=True)
                result={'samples':samples,'audit':dict(audit),'episode':str(path)}
                temporary=cache.with_suffix(f'.{self.rank}.{wid}.tmp');torch.save(result,temporary);temporary.replace(cache)
                cache.with_suffix('.json').write_text(json.dumps({'episode':str(path),'windows':len(samples),'audit':dict(audit)}))
            samples=list(result['samples']);rng.shuffle(samples)
            if self.edit_process_root:
                import numpy as np
                sidecar=self.edit_process_root/(path.name+'.npz')
                if not sidecar.exists():raise FileNotFoundError(f'missing edit-process sidecar: {sidecar}')
                with np.load(sidecar) as stored:trace={key:stored[key] for key in stored.files}
                for sample in samples:
                    start=int(sample['metadata']['start']);anchors=np.asarray(sample['metadata']['anchors'])
                    absolute=trace['target'][start:start+8]
                    local=absolute-(anchors[None]-6)
                    inside=((local>=0)&(local<13)).all(-1)
                    address=np.zeros(inside.shape,np.int64)
                    address[inside]=np.ravel_multi_index(tuple(local[inside].T),(13,13,13))
                    known=sample['inputs']['voxel_known'].numpy().reshape(len(anchors),-1)
                    target_valid=trace['target_valid'][start:start+8]&inside
                    for step,slot in np.argwhere(target_valid):
                        target_valid[step,slot]&=known[slot,address[step,slot]]
                    for key in ('kind','age_frames','age_seconds','progress','progress_valid','completed'):
                        sample['targets']['edit_'+key]=torch.from_numpy(trace[key][start:start+8].copy())
                    sample['targets']['edit_target']=torch.from_numpy(address)
                    sample['targets']['edit_target_valid']=torch.from_numpy(target_valid.copy())
            yield from samples
