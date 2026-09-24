"""Bounded full-size CUDA/DDP forward-backward and M3 raster equivalence check."""
import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from plot.models.state_policy_large import LargeStatePolicy,LargeStatePolicyArgs
from train_scripts.train_state_large import init_distributed
from plot.data.renderer_dataset import raster_camera
import numpy as np
import json

def main():
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3);p.add_argument('--batch-size',type=int,default=1)
    args=p.parse_args();rank,world,device=init_distributed();torch.manual_seed(42)
    check=torch.tensor(float(rank+1),device=device)
    if world>1:dist.all_reduce(check)
    assert int(check)==world*(world+1)//2
    model=LargeStatePolicy(LargeStatePolicyArgs(402,32,args.profile)).to(device)
    b=args.batch_size
    data=dict(resident_state=torch.randn(b,8,2,18,device=device),resident_actions=torch.zeros(b,8,2,23,device=device),
        resident_valid=torch.ones(b,8,2,dtype=torch.bool,device=device),held_item=torch.zeros(b,8,2,dtype=torch.long,device=device),
        resident_type=torch.zeros(b,8,2,dtype=torch.long,device=device),history_valid=torch.ones(b,8,dtype=torch.bool,device=device),
        target_agent=torch.zeros(b,dtype=torch.long,device=device),voxel_classes=torch.zeros(b,48,48,48,dtype=torch.long,device=device),
        voxel_known=torch.ones(b,48,48,48,dtype=torch.bool,device=device))
    camera=raster_camera(np.array([0.,0.,1.5]),np.array([1.,0.,0.]),1.2,np.zeros(3))
    data['raster_camera']=torch.from_numpy(camera).to(device)[None].repeat(b,1)
    if args.profile=='language_builder':
        for name in ('shared','current'):
            data[name+'_text']=torch.randn(b,64,768,device=device)
            data[name+'_text_mask']=torch.ones(b,64,dtype=torch.bool,device=device)
    # Check depth-stack layout against the exact M3 project_raster implementation.
    from plot.models.renderer_backbone.dit_pixel import FrameDepthStackPixelDiT
    with torch.no_grad():
        features,depth=model.geometry.raster(data)
        reference=FrameDepthStackPixelDiT.project_raster(SimpleNamespace(depth_embedder=model.geometry.depth_embedder),
            dict(raster_features=features[:,None],raster_depth=depth[:,None]),torch.float32)
        own=model.geometry.depth_embedder(features,depth).permute(0,3,4,1,2).reshape(b,752,36,64)
        torch.testing.assert_close(reference,own)
    del features,depth,reference,own
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    wrapped=DDP(model,device_ids=[device.index],broadcast_buffers=False) if world>1 else model
    torch.cuda.synchronize();started=time.monotonic()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=wrapped(data);loss,_=model.loss(logits,torch.zeros(b,8,23,device=device))
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        assert all(p.grad is not None for p in model.parameters())
    torch.cuda.synchronize()
    result=dict(rank=rank,world_size=world,profile=args.profile,parameters=sum(p.numel() for p in model.parameters()),
        steps=args.steps,batch_per_gpu=b,seconds=time.monotonic()-started,peak_mb=torch.cuda.max_memory_allocated()/2**20,
        m3_depth_stack_equivalent=True,finite_gradients=True,all_reduce=True)
    results=[None]*world
    if world>1:dist.all_gather_object(results,result)
    else:results=[result]
    if rank==0:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as stream:json.dump(results,stream,indent=2)
        print(json.dumps({'SMOKE_PASS':results}),flush=True)
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()
