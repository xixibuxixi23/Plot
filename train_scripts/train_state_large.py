"""Two-node M4 training with exact distributed validation and resumable checkpoints."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from plot.data.state_policy_dataset import CachedStatePolicyDataset,collate_state_policy
from plot.models.state_policy_large import LargeStatePolicy,LargeStatePolicyArgs
from plot.models.structured_action import StructuredActionHead


def move(batch,device):
    return {k:move(v,device) if isinstance(v,dict) else v.to(device,non_blocking=True) for k,v in batch.items()}


def init_distributed():
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    local=int(os.environ.get('LOCAL_RANK',0)); torch.cuda.set_device(local)
    if world>1: dist.init_process_group('nccl',timeout=timedelta(minutes=15))
    return rank,world,torch.device('cuda',local)


def baseline(cache):
    """Train-only smoothed per-horizon marginals; no geometry I/O or one huge one-hot array."""
    head=StructuredActionHead(1,8)
    labels=np.load(cache/'train.actions.npy',mmap_mode='r')
    counts={'keys':torch.zeros(8,10,dtype=torch.float64),
            'hotbar':torch.zeros(8,10,dtype=torch.float64),
            'mouse_x':torch.zeros(8,17,dtype=torch.float64),'mouse_y':torch.zeros(8,17,dtype=torch.float64)}
    for offset in range(0,len(labels),4096):
        target=head.targets(torch.tensor(np.array(labels[offset:offset+4096])))
        counts['keys']+=target['keys'].sum(0)
        for name,size in (('hotbar',10),('mouse_x',17),('mouse_y',17)):
            counts[name]+=torch.nn.functional.one_hot(target[name],size).sum(0)
    probability=(counts['keys']+.5)/(len(labels)+1)
    logits={'keys':torch.logit(probability).float()}
    for name,size in (('hotbar',10),('mouse_x',17),('mouse_y',17)):
        logits[name]=((counts[name]+.5)/(len(labels)+size*.5)).log().float()
    val=np.load(cache/'val_id.actions.npy',mmap_mode='r'); total=0.; component={k:0. for k in logits}
    for offset in range(0,len(val),4096):
        actions=torch.tensor(np.array(val[offset:offset+4096])); n=len(actions)
        loss,parts=head.loss({k:v[None].expand(n,-1,-1) for k,v in logits.items()},actions)
        total+=float(loss)*n
        for k,v in parts.items(): component[k]+=float(v.mean())*n
    return dict(loss=total/len(val),parts={k:v/len(val) for k,v in component.items()},train_windows=len(labels),val_windows=len(val))


@torch.no_grad()
def evaluate(model,loader,device,world,text_mode=None):
    model.eval()
    # sample count, total loss, four component losses; then two actions x three thresholds x TP/FP/FN.
    sums=torch.zeros(6,device=device,dtype=torch.float64)
    counts=torch.zeros(2,3,3,device=device,dtype=torch.float64)
    changed=torch.zeros(1,device=device,dtype=torch.float64)
    for batch in loader:
        batch=move(batch,device)
        if text_mode=='none':
            for name in ('shared','current'): batch['inputs'][name+'_text_mask'].zero_()
        elif text_mode=='shuffled':
            text=batch['inputs']['current_text']
            changed+=(text!=text.roll(1,0)).flatten(1).any(-1).sum()
            for key in ('shared_text','current_text','shared_text_mask','current_text_mask'):
                batch['inputs'][key]=batch['inputs'][key].roll(1,0)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(batch['inputs'])
            loss,parts=model.loss(logits,batch['target_actions'],batch['valid_mask'])
        n=len(batch['target_actions']); sums[0]+=n; sums[1]+=loss.double()*n
        for index,key in enumerate(('keys','hotbar','mouse_x','mouse_y')): sums[index+2]+=parts[key].double().mean()*n
        for action,keyslot,slot in ((8,7,0),(9,8,1)):
            true=batch['target_actions'][...,action]>0
            probability=logits['keys'][...,keyslot].sigmoid()
            for j,threshold in enumerate((.1,.3,.5)):
                predicted=probability>=threshold
                counts[slot,j]+=torch.stack(((predicted&true).sum(),(predicted&~true).sum(),(~predicted&true).sum()))
    if world>1:
        dist.all_reduce(sums);dist.all_reduce(counts);dist.all_reduce(changed)
    n=sums[0].clamp_min(1)
    result=dict(windows=int(sums[0]),loss=float(sums[1]/n),
                parts={k:float(sums[j+2]/n) for j,k in enumerate(('keys','hotbar','mouse_x','mouse_y'))})
    for i,name in enumerate(('dig_attack','place')):
        result[name]={}
        for j,threshold in enumerate((.1,.3,.5)):
            tp,fp,fn=counts[i,j]
            result[name][str(threshold)]=dict(tp=int(tp),fp=int(fp),fn=int(fn),
                precision=float(tp/(tp+fp).clamp_min(1)),recall=float(tp/(tp+fn).clamp_min(1)),
                f1=float(2*tp/(2*tp+fp+fn).clamp_min(1)))
    if text_mode=='shuffled':result['changed_current_text_fraction']=float(changed/n)
    model.train();return result


def capture_rng():
    state=np.random.get_state()
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),python=random.getstate(),
        numpy=(state[0],state[1].tolist(),state[2],state[3],state[4]))


def restore_rng(state):
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
    random.setstate(state['python']);value=state['numpy']
    np.random.set_state((value[0],np.array(value[1],dtype=np.uint32),value[2],value[3],value[4]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=100000);parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--lr',type=float,default=2e-4)
    parser.add_argument('--validate-every',type=int,default=5000);parser.add_argument('--checkpoint-every',type=int,default=1000)
    parser.add_argument('--hidden',type=int,default=768);parser.add_argument('--depth',type=int,default=12)
    parser.add_argument('--heads',type=int,default=12);parser.add_argument('--warmup',type=int,default=1000)
    parser.add_argument('--positive-weight',type=float,default=4.);parser.add_argument('--key-weight',type=float,default=2.)
    parser.add_argument('--seed',type=int,default=42);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--allow-pilot',action='store_true',help='smoke tests only; production requires uncapped data')
    args=parser.parse_args();rank,world,device=init_distributed()
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    summary=json.loads((args.cache/'summary.json').read_text())
    if not summary.get('m3_state'):raise ValueError('cache must include M3 camera/resident actions')
    if not args.allow_pilot and not summary['selection']['all_data']:raise ValueError('production requires full eligible data')
    cfg=LargeStatePolicyArgs(summary['num_block_classes'],len(summary['item_vocabulary']),summary['profile'],
        hidden=args.hidden,heads=args.heads,depth=args.depth,text_hidden_size=summary.get('text_hidden_size',768))
    raw=LargeStatePolicy(cfg).to(device)
    optimizer=torch.optim.AdamW(raw.parameters(),lr=args.lr,weight_decay=.01)
    saved=None;start=0;best=float('inf')
    if args.resume:
        saved=torch.load(args.output/'latest.pt',map_location='cpu',weights_only=True)
        if saved['config']!=asdict(cfg) or saved['world_size']!=world or saved['batch_size']!=args.batch_size:
            raise ValueError('resume topology/config mismatch')
        if saved['dataset']!=summary:raise ValueError('resume dataset mismatch')
        for key in ('steps','lr','warmup','positive_weight','key_weight','seed'):
            if saved['training'][key]!=getattr(args,key):raise ValueError('resume optimizer schedule mismatch: '+key)
        raw.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step'];best=saved['best_val_loss']
    model=DDP(raw,device_ids=[device.index],broadcast_buffers=False) if world>1 else raw
    train=CachedStatePolicyDataset(args.cache/'train.jsonl',cfg.profile)
    val=CachedStatePolicyDataset(args.cache/'val_id.jsonl',cfg.profile)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=args.seed)
    loader=DataLoader(train,batch_size=args.batch_size,sampler=sampler,num_workers=args.workers,
        persistent_workers=args.workers>0,drop_last=True,pin_memory=True,collate_fn=collate_state_policy)
    # Every held-out sample exactly once, no distributed-sampler padding.
    val_ids=list(range(rank,len(val),world))
    def evaluation_loader(ids):
        return DataLoader(Subset(val,ids),batch_size=args.batch_size,num_workers=args.workers,
            pin_memory=True,collate_fn=collate_state_policy)
    val_loader=evaluation_loader(val_ids)
    if not len(loader):raise ValueError('too few training samples for batch/world size')
    if rank==0:
        if not args.resume: args.output.mkdir(parents=True,exist_ok=False)
        paths=['plot/models/state_policy_large.py','plot/models/structured_action.py','plot/data/state_policy_dataset.py',
            'plot/models/renderer_backbone/embeddings.py','plot/models/renderer_backbone/voxel_rasterizer.py',
            'plot/models/renderer_backbone/camera_util.py','plot/data/renderer_dataset.py',
            'dataset_toolkits/prepare_state_policy.py','train_scripts/train_state_large.py']
        root=Path(__file__).resolve().parents[1]
        hashes={s:hashlib.sha256((root/s).read_bytes()).hexdigest() for s in paths}
        if not args.resume:
            for path in paths:
                destination=args.output/'source'/path;destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(root/path,destination)
            config=dict(model=asdict(cfg),training={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                dataset=summary,world_size=world,global_batch=args.batch_size*world,
                parameters=sum(p.numel() for p in raw.parameters()),source_sha256=hashes)
            (args.output/'config.json').write_text(json.dumps(config,indent=2))
            (args.output/'baseline.json').write_text(json.dumps(baseline(args.cache),indent=2))
            print(json.dumps({'configured':config['parameters'],'world_size':world,'global_batch':args.batch_size*world}),flush=True)
    if world>1:dist.barrier()

    def save(step,improved=False):
        states=[None]*world
        if world>1:dist.all_gather_object(states,capture_rng())
        else:states=[capture_rng()]
        if rank==0:
            payload=dict(model=raw.state_dict(),optimizer=optimizer.state_dict(),config=asdict(cfg),dataset=summary,
                step=step,best_val_loss=best,rng=states,world_size=world,batch_size=args.batch_size,
                training={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
            temp=args.output/'latest.tmp';torch.save(payload,temp);temp.replace(args.output/'latest.pt')
            if improved:
                shutil.copy2(args.output/'latest.pt',args.output/'best-val-loss.tmp')
                (args.output/'best-val-loss.tmp').replace(args.output/'best-val-loss.pt')
            if step and step%10000==0:
                shutil.copy2(args.output/'latest.pt',args.output/f'step-{step:06d}.tmp')
                (args.output/f'step-{step:06d}.tmp').replace(args.output/f'step-{step:06d}.pt')
        if world>1:dist.barrier()

    def validate(step):
        nonlocal best
        result=evaluate(raw,val_loader,device,world)
        record=dict(step=step,val=result)
        if step==args.steps and cfg.profile=='language_builder':
            ids=np.random.default_rng(args.seed+97).permutation(len(val)).tolist()[rank::world]
            record['val_no_text']=evaluate(raw,evaluation_loader(ids),device,world,text_mode='none')
            record['val_shuffled_text']=evaluate(raw,evaluation_loader(ids),device,world,text_mode='shuffled')
        improved=result['loss']<best;best=min(best,result['loss'])
        if rank==0:
            with (args.output/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        save(step,improved)

    # First full validation after 1k updates; do not spend a full pass evaluating random initialization.
    epoch=start//len(loader);skip=start%len(loader);sampler.set_epoch(epoch);iterator=iter(loader)
    for _ in range(skip):next(iterator)
    if saved is not None:restore_rng(saved['rng'][rank])
    started=time.monotonic()
    for step in range(start+1,args.steps+1):
        try:batch=next(iterator)
        except StopIteration:
            epoch+=1;sampler.set_epoch(epoch);iterator=iter(loader);batch=next(iterator)
        warm=min(1.,step/max(args.warmup,1))
        progress=max(0.,(step-args.warmup)/max(1,args.steps-args.warmup))
        lr=args.lr*warm*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr
        batch=move(batch,device);optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=model(batch['inputs'])
            loss,parts=raw.loss(logits,batch['target_actions'],batch['valid_mask'],
                positive_weight=args.positive_weight,key_weight=args.key_weight,
                horizon_weights=torch.arange(8,0,-1,device=device))
        finite=torch.isfinite(loss).int()
        if world>1:dist.all_reduce(finite,op=dist.ReduceOp.MIN)
        if not finite.item():raise FloatingPointError('nonfinite training loss on a rank')
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(raw.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step==start+1 or step%20==0:
            metrics=torch.stack([loss.detach(),*[parts[k].detach().mean() for k in ('keys','hotbar','mouse_x','mouse_y')]]).float()
            if world>1:dist.all_reduce(metrics);metrics/=world
            if rank==0:
                row=dict(step=step,epoch=epoch,loss=float(metrics[0]),parts={k:float(metrics[j+1]) for j,k in enumerate(('keys','hotbar','mouse_x','mouse_y'))},
                    grad_norm=float(grad),lr=lr,elapsed_seconds=time.monotonic()-started,
                    gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20)
                with (args.output/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
        if step==1000 or step%args.validate_every==0 or step==args.steps:validate(step)
        elif step%args.checkpoint_every==0:save(step)
    if rank==0:
        (args.output/'COMPLETE.json').write_text(json.dumps(dict(steps=args.steps,best_val_loss=best,profile=cfg.profile)))
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()
