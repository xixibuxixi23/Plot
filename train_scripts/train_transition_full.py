"""Full S01 joint M2 training with independent val_id and no HP supervision."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from plot.data.transition_stream import TransitionStream
from plot.data.transition_dataset import collate_transition
from plot.models.transition import TransitionArgs,TransitionNetwork
from plot.training.transition_trainer import transition_loss
from plot.training.transition_runner import evaluate, move, visualize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True)
    p.add_argument('--index',required=True)
    p.add_argument('--vocabulary',required=True)
    p.add_argument('--cache-root',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--width',type=int,default=256)
    p.add_argument('--depth',type=int,default=6);p.add_argument('--heads',type=int,default=8)
    p.add_argument('--device',default='cuda');p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--null-weight',type=float,default=.1);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--save-every',type=int,default=500);p.add_argument('--eval-batches',type=int,default=32)
    p.add_argument('--wandb-mode',choices=['online','offline','disabled'],default='online')
    a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(4)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.jsonl').exists():raise ValueError('output contains an existing run')
    train=TransitionStream(a.index,a.vocabulary,a.cache_root,'train',a.seed,dataset_root=a.dataset_root)
    val=TransitionStream(a.index,a.vocabulary,a.cache_root,'val_id',a.seed,dataset_root=a.dataset_root)
    ti={Path(r['path']).name.split('_seed')[-1] for r in train.records}
    vi={Path(r['path']).name.split('_seed')[-1] for r in val.records}
    if ti & vi:raise ValueError('train/validation scenario seeds overlap')
    loader_args=dict(batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate_transition,
                     pin_memory=True,persistent_workers=a.workers>0)
    if a.workers:loader_args.update(prefetch_factor=1)
    tl=DataLoader(train,**loader_args);vl=DataLoader(val,**loader_args)
    cfg=TransitionArgs(len(train.vocabulary.class_to_raw),len(train.items),a.width,a.depth,a.heads)
    model=TransitionNetwork(cfg).to(a.device)
    # These heads are neither supervised nor updated; HP remains an initial-state condition.
    for head in (model.hp_aux,model.damage_head):
        for param in head.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW((x for x in model.parameters() if x.requires_grad),lr=a.lr)
    config={'training':vars(a),'model':asdict(cfg),'objective':'no_hp',
            'train_episodes':len(train.records),'val_episodes':len(val.records),
            'class_to_raw':train.vocabulary.class_to_raw,'items':train.items,
            'parameters':sum(x.numel() for x in model.parameters()),
            'window_policy':'all eligible stride-8 windows, no episode cap; lazy shuffled episode stream',
            'evaluation':'periodic rotating val_id stream subset; full val_id pass after final update',
            'supervised':['position','angle','camera_position','camera_direction','address','block','held_item']}
    (out/'config.json').write_text(json.dumps(config,indent=2))
    run=None
    if a.wandb_mode!='disabled':
        import wandb
        run=wandb.init(project='plot-m2',name='m2-full-no-hp',mode=a.wandb_mode,dir=str(out),config=config)
        (out/'wandb_run.json').write_text(json.dumps({'url':run.url,'id':run.id},indent=2))
    seen=set();windows=set();iterator=iter(tl)
    def log(step,values):
        row={'step':step,**values}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if run:run.log(values,step=step,commit=False)
    try:
        for step in range(1,a.steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(tl);batch=next(iterator)
            for m in batch['metadata']:
                seen.add(m['episode']);windows.add((m['episode'],m['start']))
            values=move(batch,a.device);model.train();optimizer.zero_grad(set_to_none=True)
            prediction=model(values['inputs'])
            loss,metrics=transition_loss(model,prediction,**values,objective='no_hp',null_weight=a.null_weight)
            if not torch.isfinite(loss):raise FloatingPointError('non-finite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm):raise FloatingPointError('non-finite gradient')
            optimizer.step()
            metrics.update(gradient_norm=float(norm),episodes_seen=len(seen),unique_windows_seen=len(windows))
            log(step,{f'train/{k}':v for k,v in metrics.items()})
            if step%a.save_every==0 or step==a.steps:
                tmp=out/'checkpoint.tmp'
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,'config':config},tmp)
                tmp.replace(out/'checkpoint.pt')
                (out/'coverage.json').write_text(json.dumps({'step':step,'episodes_seen':sorted(seen),'unique_windows_seen':len(windows)}))
                limit=sys.maxsize if step==a.steps else a.eval_batches
                metrics,example=evaluate(model,vl,a.device,max_batches=limit,objective='no_hp',null_weight=a.null_weight)
                prefix='val_full' if step==a.steps else 'val_subset'
                log(step,{f'{prefix}/{k}':v for k,v in metrics.items()})
                path=out/f'step_{step:06d}.png';visualize(example,path,step)
                if run:run.log({'validation/comparison':wandb.Image(str(path))},step=step,commit=False)
            if run:run.log({},step=step,commit=True)
        (out/'COMPLETED.json').write_text(json.dumps({'steps':a.steps,'episodes_seen':len(seen),'unique_windows_seen':len(windows)}))
    finally:
        if run:run.finish()

if __name__=='__main__':main()
