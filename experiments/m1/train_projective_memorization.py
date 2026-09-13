"""DDP scene memorization: train on explicitly listed splits, audit fitted data.

This protocol intentionally includes former evaluation scenes. Never report its
audit as held-out generalization. GT geometry supplies labels, not model inputs.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plot.data import TextAgentFillDataset
from plot.models import GeometryConditionedFillArgs, ProjectiveFillArgs
from plot.models.projection import raycast_voxel_targets
from plot.models.visible_supervision import visible_voxel_masks
from scripts.pilot_projective_fill import inputs, metrics


class Indexed(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "sample_index": torch.tensor(index)}


def save_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def region_loss(logits, b, unseen_weight):
    error = F.cross_entropy(logits.float(), b["target"], reduction="none")
    writable = b["target_valid"] & b["fill_mask"] & ~b["known_mask"]
    surface, free = b["surface"] & writable, b["free"] & writable
    unseen = writable & ~surface & ~free
    loss = error.new_zeros(())
    for mask, weight in ((surface, 1.), (free, 1.), (unseen, unseen_weight)):
        loss = loss + weight * ((error * mask).flatten(1).sum(1) /
                               mask.flatten(1).sum(1).clamp_min(1)).mean()
    return loss


def backward_batch(model, b, unseen_weight, microbatch_size):
    """Accumulate an exact scene mean, including an uneven final microbatch."""
    count = b["target"].shape[0]
    size = microbatch_size or count
    total = torch.zeros((), device=b["target"].device)
    for start in range(0, count, size):
        stop = min(start + size, count)
        part = {k: v[start:stop] for k, v in b.items()}
        context = model.no_sync() if isinstance(model, DDP) and stop < count else nullcontext()
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            loss = region_loss(model(**inputs(part)), part, unseen_weight) * ((stop - start) / count)
            loss.backward()
        total += loss.detach()
    return total


@torch.no_grad()
def prepare(b, pose, air, cache):
    ids = b["sample_index"].tolist()
    b["gt_position"] = b["camera_position"].clone()
    b["gt_direction"] = b["camera_direction"].clone()
    missing = [i for i, key in enumerate(ids) if key not in cache]
    if missing:
        m = {k: v[missing] for k, v in b.items()}
        surface, free = visible_voxel_masks(m["target"], m["target_valid"], m["camera_position"],
            m["camera_direction"], m["camera_valid"] & m["agent_mask"], m["fov_x"], m["fov_y"], air)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            geo = pose.geometry(m["images"], m["agent_mask"])
            p, d = pose._anchor_pose(geo)
            p, d = pose._camera_poses(geo, p, d)
        for j, i in enumerate(missing):
            cache[ids[i]] = (surface[j].cpu(), free[j].cpu(), p[j].float().cpu(), d[j].float().cpu())
    for j, key in enumerate(("surface", "free", "camera_position", "camera_direction")):
        b[key] = torch.stack([cache[i][j] for i in ids]).cuda(non_blocking=True)
    return b


@torch.no_grad()
def audit(model, dataset, indices, pose, air, cache):
    model.eval()
    rows = []
    for index in indices:
        b = prepare({k: v[None].cuda() for k, v in dataset[index].items()}, pose, air, cache)
        b["ray_gt"] = []
        for view in range(b["images"].shape[1]):
            h, d, _ = raycast_voxel_targets(((b["target"] != air) & b["target_valid"]).float(),
                b["gt_position"][:, view], b["gt_direction"][:, view],
                b["fov_x"][:, view], b["fov_y"][:, view],
                height=60, width=106, samples=192, max_distance=32.)
            b["ray_gt"].append((h, d))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            row = metrics(model.predict(**inputs(b)), b, air)
        rows.append({"index": index, **row})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="../textagent/data/batches/s01_v1_mask_20260907")
    p.add_argument("--vocabulary", default="datasets/s01_block_vocabulary.json")
    p.add_argument("--episode-index", default="datasets/s01_episode_index.json")
    p.add_argument("--pose-checkpoint", default="outputs/m1_direct_visible_v8/checkpoint_00001500.pt")
    p.add_argument("--splits", nargs="+", default=["train", "val_id", "test_id"])
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--microbatch-size", type=int, default=0,
                   help="Split each per-rank batch for memory; effective batch stays unchanged")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--channels", type=int, default=48)
    p.add_argument("--image-channels", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--constant-lr", action="store_true",
                   help="Continue at a fixed LR instead of restarting cosine scheduling")
    p.add_argument("--unseen-weight", type=float, default=.1)
    p.add_argument("--audit-samples", type=int, default=96)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=0, help="Smoke/throughput check only; 0 completes epochs")
    p.add_argument("--seed", type=int, default=20260910)
    a = p.parse_args()
    world, rank, local = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl")
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    random.seed(a.seed + rank)
    torch.set_float32_matmul_precision("high")
    sets = [TextAgentFillDataset(a.dataset_root, a.vocabulary, split=s, samples_per_agent=1,
        max_agents=2, num_views=2, initial_only=True, episode_index=a.episode_index,
        canonical_yaw=True) for s in a.splits]
    dataset = Indexed(ConcatDataset(sets))
    sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=a.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=a.batch_size, sampler=sampler, num_workers=a.workers,
                        pin_memory=True, persistent_workers=a.workers > 0)
    checkpoint = torch.load(a.pose_checkpoint, map_location="cpu", weights_only=False)
    pose = GeometryConditionedFillArgs(**checkpoint["model_args"]).build().cuda().eval().requires_grad_(False)
    pose.load_state_dict(checkpoint["model"])
    air = checkpoint["model_args"]["air_class"]
    ma = ProjectiveFillArgs(sets[0].vocabulary.size, channels=a.channels, image_channels=a.image_channels)
    model = ma.build().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0.)
    epoch_start = batch_start = step = 0
    if a.resume:
        c = torch.load(a.resume, map_location="cpu", weights_only=False)
        if c["model_args"] != asdict(ma) or c["manifest"]["splits"] != a.splits:
            raise ValueError("Resume model or split mismatch")
        for key in ("dataset_root", "vocabulary", "episode_index", "pose_checkpoint"):
            if c["manifest"][key] != getattr(a, key):
                raise ValueError(f"Resume {key} mismatch")
        if c["manifest"]["seed"] != a.seed:
            raise ValueError("Resume must preserve sampler seed")
        if c["next_batch"]:
            old_world = c["manifest"]["world_size"]
            if c["manifest"]["effective_batch"] != world * a.batch_size:
                raise ValueError("Mid-epoch resume must preserve effective batch")
            if old_world != world and (len(dataset) % world or len(dataset) % old_world):
                raise ValueError("World-size change requires no sampler padding")
        model.load_state_dict(c["model"])
        optimizer.load_state_dict(c["optimizer"])
        epoch_start, batch_start, step = c["epoch"], c["next_batch"], c["step"]
    module = model
    if world > 1:
        model = DDP(model, device_ids=[local])
    a.output_dir.mkdir(parents=True, exist_ok=True)
    if rank == 0 and not a.resume and (a.output_dir / "manifest.json").exists():
        raise FileExistsError("Output directory already used")
    manifest = {**vars(a), "output_dir": str(a.output_dir), "resume": str(a.resume) if a.resume else None,
        "model_args": asdict(ma), "split_counts": dict(zip(a.splits, map(len, sets))),
        "training_population": len(dataset), "world_size": world, "effective_batch": world * a.batch_size,
        "protocol": "memorization, former val/test are TRAINING data; no held-out claims",
        "pose_source": "frozen V8 predictions and dataset FOV", "instance_masks": False,
        "parameter_count": sum(t.numel() for t in module.parameters())}
    audit_ids = torch.linspace(0, len(dataset) - 1, min(a.audit_samples, len(dataset))).round().long().tolist()
    if rank == 0:
        (a.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest), flush=True)
    del checkpoint
    cache = {}
    started = time.monotonic()
    initial_step = step
    def save(name, epoch, next_batch):
        if rank == 0:
            save_atomic(a.output_dir / name, {"model": module.state_dict(), "model_args": asdict(ma),
                "optimizer": optimizer.state_dict(), "epoch": epoch, "next_batch": next_batch,
                "step": step, "manifest": manifest})
    completed_epochs = epoch_start
    for epoch in range(epoch_start, a.epochs):
        sampler.set_epoch(epoch)
        model.train()
        for bi, cpu in enumerate(loader):
            if epoch == epoch_start and bi < batch_start:
                continue
            b = prepare({k: v.cuda(non_blocking=True) for k, v in cpu.items()}, pose, air, cache)
            progress = (epoch + bi / len(loader)) / a.epochs
            lr = a.learning_rate if a.constant_lr else a.learning_rate * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss = backward_batch(model, b, a.unseen_weight, a.microbatch_size)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            step += 1
            if step % 20 == 0:
                reduced = loss.detach().float()
                if world > 1:
                    dist.all_reduce(reduced)
                if rank == 0:
                    print(json.dumps({"step": step, "epoch": epoch + (bi + 1) / len(loader),
                        "loss": float(reduced / world), "lr": lr,
                        "seconds_per_step": (time.monotonic() - started) / (step - initial_step),
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30}), flush=True)
            if step % a.save_every == 0:
                save("checkpoint_latest.pt", epoch, bi + 1)
            if a.max_steps and step - initial_step >= a.max_steps:
                save("checkpoint_smoke.pt", epoch, bi + 1)
                if world > 1:
                    dist.destroy_process_group()
                return
        completed_epochs = epoch + 1
        rows = audit(module, dataset, audit_ids[rank::world], pose, air, cache)
        if world > 1:
            gathered = [None] * world
            dist.all_gather_object(gathered, rows)
            rows = [r for group in gathered for r in group]
        if rank == 0:
            avg = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "index"}
            result = {"epoch": completed_epochs, "step": step, "fitted_scene_audit": avg, "samples": rows}
            (a.output_dir / f"audit_epoch_{completed_epochs:03d}.json").write_text(json.dumps(result, indent=2))
            print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
        save("checkpoint_latest.pt", completed_epochs, 0)
    save("checkpoint_final.pt", completed_epochs, 0)
    if rank == 0:
        (a.output_dir / "completion.json").write_text(json.dumps({"completed_epochs": completed_epochs,
            "steps": step, "population": len(dataset), "status": "configured training completed; assess audit separately"}, indent=2))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
