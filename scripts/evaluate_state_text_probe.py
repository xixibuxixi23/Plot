"""Evaluate a completed checkpoint with genuinely mixed-episode text mismatch."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from plot.models.state_policy import StateInhabitantPolicy, StatePolicyArgs
from plot.data.state_policy_dataset import CachedStatePolicyDataset, collate_state_policy
from train_scripts.train_state_policy import evaluate

root = Path('/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d/runs/m4-state/text-pilot-20260922-v1')
checkpoint = torch.load(root/'model/latest.pt',map_location='cpu',weights_only=True)
model = StateInhabitantPolicy(StatePolicyArgs(**checkpoint['config'])).cuda()
model.load_state_dict(checkpoint['model'],strict=True)
dataset = CachedStatePolicyDataset(root/'cache/val_id.jsonl','language_builder')
def loader():
    return DataLoader(dataset,batch_size=8,shuffle=True,
        generator=torch.Generator().manual_seed(139),num_workers=4,collate_fn=collate_state_policy)
amp = lambda: torch.autocast('cuda',dtype=torch.bfloat16)
result = dict(step=checkpoint['step'],checkpoint='latest.pt',seed=139,validation_windows=len(dataset))
for name,mode in (('correct_text',None),('no_text','none'),('mismatched_text','shuffled')):
    result[name] = evaluate(model,loader(),torch.device('cuda'),amp,text_mode=mode)
    print(json.dumps({name:result[name]}),flush=True)
assert result['mismatched_text']['changed_current_text_fraction'] > .9, 'insufficient instruction mismatch'
output = root/'model/text-probe-mixed-episodes.json'
with output.open('x') as stream: json.dump(result,stream,indent=2)
print('TEXT_PROBE_COMPLETE',flush=True)
