"""Deterministic exhaustive evaluation for M2 ordered voxel edits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from plot.data.transition_dataset import collate_transition
from plot.data.transition_stream import TRANSITION_CACHE_REVISION, TransitionStream
from plot.kinematics import KinematicsConfig
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.training.transition_runner import evaluate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',default='val_id')
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--max-batches',type=int,default=0,
                        help='0 evaluates every window in the fixed split')
    parser.add_argument('--device',default='cuda')
    args=parser.parse_args()

    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=dict(checkpoint['config']['model'])
    cfg['kinematics']=KinematicsConfig(**cfg['kinematics'])
    model=TransitionNetwork(TransitionArgs(**cfg))
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(args.device).eval()

    args.output.parent.mkdir(parents=True,exist_ok=True)
    vocabulary=args.output.with_suffix('.block_vocabulary.json')
    vocabulary.write_text(json.dumps({'class_to_raw':checkpoint['config']['class_to_raw']}))
    manifest=json.loads(args.index.read_text())
    records=manifest['splits'][args.split]
    digest=hashlib.sha256('\n'.join(row['path'] for row in records).encode()).hexdigest()
    stream=TransitionStream(
        index=args.index,vocabulary=vocabulary,cache_root=args.cache_root,
        split=args.split,fixed_order=True,seed=0,items=checkpoint['config']['items'])
    loader=DataLoader(
        stream,batch_size=args.batch_size,num_workers=args.workers,
        collate_fn=collate_transition,pin_memory=True,
        persistent_workers=args.workers>0,
        **({'prefetch_factor':1} if args.workers else {}))
    metrics,_=evaluate(
        model,loader,args.device,max_batches=args.max_batches or None,objective='no_hp')
    report={
        'schema':'m2-edit-evaluation-v1','checkpoint':str(args.checkpoint),
        'checkpoint_step':checkpoint.get('step'),'index':str(args.index),
        'split':args.split,'episode_count':len(records),'episode_list_sha256':digest,
        'fixed_order':True,'exhaustive':args.max_batches==0,
        'max_batches':args.max_batches,'cache_revision':TRANSITION_CACHE_REVISION,
        'model':asdict(model.cfg),'metrics':metrics}
    report['model']['kinematics']=asdict(model.cfg.kinematics)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
