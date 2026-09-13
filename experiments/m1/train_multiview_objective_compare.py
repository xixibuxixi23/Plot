"""Matched 256-window memorization: flow velocity versus direct clean latent regression."""

# ruff: noqa: E402
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from experiments.m1.train_multiview_persist_full import Cache
from experiments.m1.train_multiview_voxel_dit import codec, evaluate, atomic_save
from scripts.evaluate_multiview_persist_full import sample


def direct_prediction(model, cond, valid):
    # Fixed zero input + existing absolute 3D positions are the queries.
    # Constant t=0 preserves the shared backbone; there is no sampled timestep.
    x = torch.zeros(len(cond), 48, 12, 12, 12, device=cond.device)
    return model(x, torch.zeros(len(cond), device=cond.device), cond, valid)


@torch.no_grad()
def audit(model, decoder, cache, indices, mode, step, out, rank, world):
    model.eval()
    local_indices = indices[rank::world]
    c = cache.evaluation_cache(local_indices)
    data = c["data"]
    preds = []
    for start in range(0, len(local_indices), 8):
        ix = local_indices[start : start + 8]
        cond = data["image_rays"][start : start + len(ix)].cuda()
        valid = (data["agent_mask"] & data["camera_valid"])[start : start + len(ix)].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = (
                direct_prediction(model, cond, valid)
                if mode == "direct"
                else sample(model, cond, valid, ix)
            )
        preds.append(pred.float().cpu())
    data["sampled_latent"] = torch.cat(preds)
    result = evaluate(model, decoder, c, step, out / f"rank_{rank}")
    rows = [dict(index=int(i), **r) for i, r in zip(local_indices, result["samples"])]
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        rows = sorted([r for part in gathered for r in part], key=lambda r: r["index"])
        assert [r["index"] for r in rows] == sorted(indices)
        summary = dict(
            step=step,
            count=len(rows),
            samples=rows,
            mean={k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "index"},
        )
        (out / f"audit_{step:06d}.json").write_text(json.dumps(summary, indent=2))
        print("AUDIT " + json.dumps(dict(step=step, mean=summary["mean"])), flush=True)
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["direct", "flow"], required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--initialize",
        type=Path,
        default=ROOT / "outputs/m1_multiview_persist_gtcam_s01_all_v1/checkpoint_latest.pt",
    )
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/m1_multiview_persist_s01_cache")
    args = p.parse_args()
    torch.set_num_threads(4)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 8, "Protocol requires 8 ranks and global batch 64"
    torch.manual_seed(20260911 + rank)
    args.output.mkdir(exist_ok=True, parents=True)
    (args.output / f"rank_{rank}").mkdir(exist_ok=True)
    cache = Cache(args.cache)
    indices = np.linspace(0, len(cache) - 1, 256).round().astype(int).tolist()
    # Do not include engine-unknown GT in a controlled objective comparison.
    assert all(not (cache.arrays["raw"][i] == 127).any() for i in indices)
    model = MultiViewVoxelDiT().cuda()
    state = torch.load(args.initialize, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    del state
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    ddp = DDP(model, device_ids=[local])
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    # Only the 256 chosen windows' conditioning/targets are held in GPU memory.
    data = {
        k: torch.stack([cache[i][k] for i in indices]).cuda()
        for k in ["latent", "image_rays", "agent_mask", "camera_valid"]
    }
    if rank == 0:
        manifest = dict(
            mode=args.mode,
            steps=args.steps,
            lr=args.lr,
            initialize=str(args.initialize),
            optimizer="reset AdamW; weight_decay=0",
            indices=indices,
            batch=64,
            exposures=args.steps * 64 / 256,
            seed=20260911,
            config=model.config,
            initialization_caveat="Common checkpoint was trained on flow and all scenes; comparison is a warm-start adaptation, not training from scratch.",
            direct="zero latent queries plus 3D positions; constant t=0; one forward; clean latent MSE",
            flow="random noise and sigmoid-normal time; velocity MSE; 20-step Euler inference",
            evaluation="all 256 fitted windows; global noise seed 1234+index for flow; GT latent used only as target",
        )
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    audit(model, decoder, cache, indices, args.mode, 0, args.output, rank, world)
    generator = torch.Generator(device="cuda").manual_seed(77211 + rank)
    permutation = None
    tic = time.time()
    for step in range(1, args.steps + 1):
        # Every four updates cover each selected window exactly once.
        epoch, offset = divmod(step - 1, 4)
        if offset == 0:
            permutation = torch.randperm(256, generator=torch.Generator().manual_seed(4422 + epoch))
        ix = permutation[offset * 64 + rank * 8 : offset * 64 + (rank + 1) * 8].cuda()
        target = data["latent"][ix].float()
        cond = data["image_rays"][ix].float()
        valid = data["agent_mask"][ix] & data["camera_valid"][ix]
        ddp.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.mode == "direct":
                pred = direct_prediction(ddp, cond, valid)
                loss = (pred.float() - target).square().mean()
            else:
                noise = torch.randn(target.shape, device="cuda", generator=generator)
                t = torch.randn(len(target), device="cuda", generator=generator).sigmoid()
                tt = t[:, None, None, None, None]
                x = (1 - tt) * target + (1e-5 + (1 - 1e-5) * tt) * noise
                loss = (
                    (ddp(x, t, cond, valid).float() - ((1 - 1e-5) * noise - target)).square().mean()
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 20 == 0:
            value = loss.detach().clone()
            dist.all_reduce(value)
            if rank == 0:
                row = dict(step=step, loss=float(value / world), seconds=time.time() - tic)
                with (args.output / "train.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        if step % 1000 == 0 or step == args.steps:
            if rank == 0:
                atomic_save(
                    dict(
                        model=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        step=step,
                        config=model.config,
                        mode=args.mode,
                    ),
                    args.output / "checkpoint_latest.pt",
                )
            dist.barrier()
            audit(model, decoder, cache, indices, args.mode, step, args.output, rank, world)
    if rank == 0:
        (args.output / "completion.json").write_text(json.dumps(dict(steps=args.steps, count=256)))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
