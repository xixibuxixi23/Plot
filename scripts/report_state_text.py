"""Read-only report of the bounded state/text pilot."""
import json
from pathlib import Path

root = Path('/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d/runs/m4-state/text-pilot-20260922-v1')
def read(path): return json.loads(path.read_text())
summary = root/'cache/summary.json'
if summary.exists(): print(json.dumps({'data':read(summary)['splits']}))
baseline = root/'model/baseline.json'
if baseline.exists(): print(json.dumps({'baseline':read(baseline)}))
validation = root/'model/validation.jsonl'
if validation.exists():
    rows = [json.loads(line) for line in validation.read_text().splitlines() if line.strip()]
    for row in rows:
        print(json.dumps({'step':row['step'], 'train_loss':row['train']['loss'],
                          'val':{k:v for k,v in row['val'].items() if k!='horizon_loss'}}))
    if rows:
        final = rows[-1]
        print(json.dumps({'ablations':{key:value for key,value in final.items() if key.startswith('val_')}}))
        print(json.dumps({'best_loss_step':min(rows,key=lambda row:row['val']['loss'])['step']}))
complete = root/'model/COMPLETE.json'
print(json.dumps({'complete':read(complete) if complete.exists() else False}))
