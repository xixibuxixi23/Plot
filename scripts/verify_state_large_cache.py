"""Read-only full-cache readiness/coverage gate before distributed training."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--profile',required=True)
a=p.parse_args();s=json.loads((a.cache/'summary.json').read_text())
assert s['profile']==a.profile and s['m3_state'] and s['selection']['all_data']
episodes=set()
for split in ('train','val_id'):
    rows=[json.loads(line) for line in (a.cache/f'{split}.jsonl').open()]
    assert len(rows)==s['splits'][split]['windows']>0
    labels=np.load(a.cache/f'{split}.actions.npy',mmap_mode='r')
    assert labels.shape==(len(rows),8,23)
    current={row['episode_id'] for row in rows}
    assert not episodes.intersection(current);episodes.update(current)
    assert len(current)==s['splits'][split]['episodes']
    for row in rows:
        assert row['profile']==a.profile
        if not (a.cache/row['cache']).is_file():raise FileNotFoundError(row['cache'])
    for index in sorted(set((0,len(rows)//2,len(rows)-1))):
        with np.load(a.cache/rows[index]['cache'],allow_pickle=False) as sample:
            assert sample['inputs/raster_camera'].shape==(10,)
            assert sample['inputs/resident_actions'].shape[0]==8
            np.testing.assert_array_equal(sample['target_actions'],labels[index])
print(json.dumps({'FULL_CACHE_VERIFIED':a.profile,'splits':s['splits']}),flush=True)
