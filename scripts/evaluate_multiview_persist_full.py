"""Batched 20-step sampling; unchanged per-scene metrics and exact distributed coverage."""

# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from experiments.m1.train_multiview_persist_full import Cache
from experiments.m1.train_multiview_voxel_dit import codec, evaluate


@torch.no_grad()
def sample(model, cond, valid, indices, steps=20):
    model.eval()
    device = cond.device
    x = torch.cat(
        [
            torch.randn(
                1,
                48,
                12,
                12,
                12,
                device=device,
                generator=torch.Generator(device=device).manual_seed(1234 + i),
            )
            for i in indices
        ]
    )
    schedule = np.linspace(1, 0, steps + 1)
    schedule = 3 * schedule / (1 + 2 * schedule)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for t, tp in zip(schedule[:-1], schedule[1:]):
            x = x - float(t - tp) * model(x, torch.full((len(x),), t, device=device), cond, valid)
    return x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output", type=Path, default=ROOT / "outputs/m1_multiview_persist_gtcam_s01_all_v1"
    )
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/m1_multiview_persist_s01_cache")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--chunk-size", type=int, default=32)
    args = p.parse_args()
    torch.set_num_threads(4)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    model = MultiViewVoxelDiT().cuda().eval()
    state = torch.load(args.output / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    step = state["step"]
    del state
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    cache = Cache(args.cache)
    directory = args.output / f"full_rank_{rank}"
    directory.mkdir(exist_ok=True)
    rows = []
    checks = []
    for start in range(rank * args.chunk_size, len(cache), world * args.chunk_size):
        indices = list(range(start, min(start + args.chunk_size, len(cache))))
        c = cache.evaluation_cache(indices)
        data = c["data"]
        sampled = []
        for offset in range(0, len(indices), args.batch_size):
            ix = indices[offset : offset + args.batch_size]
            cond = data["image_rays"][offset : offset + len(ix)].cuda()
            valid = (data["agent_mask"] & data["camera_valid"])[offset : offset + len(ix)].cuda()
            x = sample(model, cond, valid, ix)
            if not checks:
                serial = sample(model, cond[:1], valid[:1], ix[:1])
                checks.append(
                    dict(
                        index=ix[0],
                        batch_vs_serial_mse=float((x[:1] - serial).square().mean()),
                        max_abs=float((x[:1] - serial).abs().max()),
                    )
                )
            sampled.append(x.cpu())
        data["sampled_latent"] = torch.cat(sampled)
        result = evaluate(model, decoder, c, start, directory, seed_base=1234 + start)
        rows.extend([dict(index=i, **r) for i, r in zip(indices, result["samples"])])
        print(json.dumps(dict(rank=rank, evaluated=len(rows), last_index=indices[-1])), flush=True)
    all_rows = [None] * world
    dist.all_gather_object(all_rows, rows)
    all_checks = [None] * world
    dist.all_gather_object(all_checks, checks)
    if rank == 0:
        rows = sorted([r for part in all_rows for r in part], key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(len(cache))):
            raise RuntimeError("Incomplete or duplicate full evaluation")
        result = dict(
            step=step,
            count=len(rows),
            sampling_batch=args.batch_size,
            seed_protocol="1234 + global sample index",
            batch_checks=all_checks,
            mean={k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "index"},
            samples=rows,
        )
        (args.output / "full_fit_metrics.json").write_text(json.dumps(result, indent=2))
        (args.output / "completion.json").write_text(
            json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2)
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
