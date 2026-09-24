"""One real cached sample: CUDA language forward/backward and old checkpoint reload."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file
from plot.models.state_policy import StateInhabitantPolicy, StatePolicyArgs
from plot.data.state_policy_dataset import load_state_cache, collate_state_policy
from plot.data.fill_dataset import BlockVocabulary

root = Path('/apdcephfs_bjzf/share_303702067/ruichzhang/code/multi3d')
cache = root/'runs/m4-state/text-pilot-20260922-v1/cache'
sample = load_state_cache(next(cache.glob('*.npz')))
text = load_file(str(root/'downloads/plot-checkpoints-main/m4_text/text_cache.safetensors'))
for name in ('shared','current'):
    index = int(sample['inputs'].pop(name+'_text_id'))
    sample['inputs'][name+'_text'] = text['encoder_hidden'][index].float()
    sample['inputs'][name+'_text_mask'] = text['attention_mask'][index].bool()
batch = collate_state_policy([sample])
batch['inputs'] = {key:value.cuda() for key,value in batch['inputs'].items()}
items = json.loads((cache/'item_vocabulary.json').read_text())
vocab = BlockVocabulary.load(root/'Plot/derived/common/block_vocabulary.json')
model = StateInhabitantPolicy(StatePolicyArgs(vocab.size,len(items),'language_builder',text_hidden_size=768)).cuda()
with torch.autocast('cuda',dtype=torch.bfloat16):
    logits = model(batch['inputs'])
    loss,_ = model.loss(logits,batch['target_actions'].cuda())
assert torch.isfinite(loss)
loss.backward()
assert model.text_projection.weight.grad.abs().sum() > 0
assert model.decode(logits).shape == (1,8,23)
print(json.dumps(dict(language_cuda_smoke='PASS',loss=float(loss),parameters=sum(p.numel() for p in model.parameters()))),flush=True)
old = torch.load(root/'runs/m4-state/full-10k-20260921-v3/models/villager_peaceful/latest.pt',map_location='cpu',weights_only=True)
restored = StateInhabitantPolicy(StatePolicyArgs(**old['config']))
restored.load_state_dict(old['model'],strict=True)
print('OLD_NPC_CHECKPOINT_STRICT_RELOAD=PASS',flush=True)
