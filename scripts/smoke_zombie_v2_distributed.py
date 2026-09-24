"""Full-size CUDA/DDP smoke for the regularized zombie v2 policy."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from plot.data.renderer_dataset import raster_camera
from plot.models.state_policy_large_v2 import ZombieStatePolicyV2, ZombieStatePolicyV2Args
from train_scripts.train_state_zombie_v2 import init_distributed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    rank, world, device = init_distributed()
    torch.manual_seed(52)
    check = torch.tensor(float(rank + 1), device=device)
    if world > 1:
        dist.all_reduce(check)
    assert int(check) == world * (world + 1) // 2
    model = ZombieStatePolicyV2(ZombieStatePolicyV2Args(402, 32)).to(device)
    batch = args.batch_size
    data = dict(
        resident_state=torch.randn(batch, 8, 2, 18, device=device),
        resident_actions=torch.zeros(batch, 8, 2, 23, device=device),
        resident_valid=torch.ones(batch, 8, 2, dtype=torch.bool, device=device),
        held_item=torch.zeros(batch, 8, 2, dtype=torch.long, device=device),
        resident_type=torch.zeros(batch, 8, 2, dtype=torch.long, device=device),
        history_valid=torch.ones(batch, 8, dtype=torch.bool, device=device),
        target_agent=torch.zeros(batch, dtype=torch.long, device=device),
        voxel_classes=torch.zeros(batch, 48, 48, 48, dtype=torch.long, device=device),
        voxel_known=torch.ones(batch, 48, 48, 48, dtype=torch.bool, device=device))
    camera = raster_camera(np.array([0., 0., 1.5]), np.array([1., 0., 0.]),
                           1.2, np.zeros(3))
    data["raster_camera"] = torch.from_numpy(camera).to(device)[None].repeat(batch, 1)
    from plot.models.renderer_backbone.dit_pixel import FrameDepthStackPixelDiT
    with torch.no_grad():
        features, depth = model.geometry.raster(data)
        reference = FrameDepthStackPixelDiT.project_raster(
            SimpleNamespace(depth_embedder=model.geometry.depth_embedder),
            dict(raster_features=features[:, None], raster_depth=depth[:, None]),
            torch.float32)
        own = model.geometry.depth_embedder(features, depth).permute(0, 3, 4, 1, 2).reshape(
            batch, 752, 36, 64)
        torch.testing.assert_close(reference, own)
    del features, depth, reference, own
    actions = torch.zeros(batch, 8, 23, device=device)
    actions[:, 1:5, 8] = 1
    actions[:, :, 21] = torch.linspace(-0.08, 0.08, 8, device=device)
    actions[:, :, 22] = torch.linspace(0.05, -0.05, 8, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    wrapped = DDP(model, device_ids=[device.index], broadcast_buffers=False) if world > 1 else model
    torch.cuda.synchronize()
    started = time.monotonic()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = wrapped(data)
            loss, parts = model.loss(
                logits, actions, positive_weight=4, attack_weight=2,
                mouse_smoothing=0.2, mouse_move_positive_weight=2,
                horizon_weights=torch.linspace(1, 0.25, 8, device=device))
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value).all() for value in parts.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
        assert all(parameter.grad is not None for parameter in model.parameters())
    torch.cuda.synchronize()
    result = dict(
        rank=rank, world_size=world,
        parameters=sum(parameter.numel() for parameter in model.parameters()),
        steps=args.steps, batch_per_gpu=batch,
        seconds=time.monotonic() - started,
        peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
        m3_depth_stack_equivalent=True, finite_gradients=True, all_reduce=True)
    results = [None] * world
    if world > 1:
        dist.all_gather_object(results, result)
    else:
        results = [result]
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(results, stream, indent=2)
        print(json.dumps({"SMOKE_PASS": results}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
