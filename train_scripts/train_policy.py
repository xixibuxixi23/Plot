#!/usr/bin/env python3
"""Train inserted M4: eight completed M3 frames -> one eight-action chunk."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safetensors.torch import load_file
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler

from plot.data.inserted_policy_dataset import InsertedPolicyDataset, collate_inserted_policy
from plot.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec


class DistributedWeightedSampler(Sampler):
    """Deterministic weighted replacement sampling, evenly sharded across ranks."""
    def __init__(self, weights, replicas, rank, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.replicas, self.rank, self.seed, self.epoch = replicas, rank, seed, 0
        self.samples = (len(self.weights) + replicas - 1) // replicas

    def set_epoch(self, epoch): self.epoch = epoch
    def __len__(self): return self.samples
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        selected = torch.multinomial(self.weights, self.samples * self.replicas,
                                     replacement=True, generator=generator)
        return iter(selected[self.rank::self.replicas].tolist())


def move_conditions(conditions, device):
    return {key: value.to(device, non_blocking=True) for key, value in conditions.items()}


def lookup_text(cache, ids, device, dtype):
    ids = ids.long().cpu()
    return (cache["encoder_hidden"][ids].to(device=device, dtype=dtype, non_blocking=True),
            cache["attention_mask"][ids].to(device=device, dtype=torch.bool, non_blocking=True))


def load_renderer(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint["config"].get("renderer", checkpoint["config"])
    renderer = Renderer(RendererArgs(**config))
    renderer.load_state_dict(checkpoint["model"], strict=True)
    return renderer


@torch.no_grad()
def evaluate(model, codec, text_cache, loader, device, batches, precision):
    model.eval(); values = []
    for number, batch in enumerate(loader):
        if number >= batches: break
        conditions = move_conditions(batch["conditions"], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            latent = codec.encode(batch["rgb"].to(device, non_blocking=True))
            dtype = latent.dtype
            shared, shared_mask = lookup_text(text_cache, batch["shared_text_id"], device, dtype)
            current, current_mask = lookup_text(text_cache, batch["current_text_id"], device, dtype)
            _, logits = model(latent, conditions, batch["family_id"].to(device),
                              batch["profile_id"].to(device), shared_text=shared,
                              shared_text_mask=shared_mask, current_text=current,
                              current_text_mask=current_mask,
                              policy_indices=batch["policy_indices"].to(device))
            loss, _ = model.loss(logits, batch["target_actions"].to(device),
                                 batch["valid_mask"].to(device),
                                 horizon_weights=torch.arange(8, 0, -1, device=device))
            values.append(loss)
    model.train()
    return torch.stack(values).mean() if values else torch.tensor(float("nan"), device=device)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--train-index", required=True); p.add_argument("--val-index", required=True)
    p.add_argument("--vocabulary", required=True); p.add_argument("--text-catalog", required=True)
    p.add_argument("--text-cache", required=True); p.add_argument("--m3-checkpoint", required=True)
    p.add_argument("--pixel-vae", required=True); p.add_argument("--output-dir", required=True)
    p.add_argument("--resume"); p.add_argument("--policy-blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1); p.add_argument("--workers", type=int, default=4)
    p.add_argument("--steps", type=int, default=10000); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--validate-every", type=int, default=500)
    p.add_argument("--val-batches", type=int, default=32); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampling", choices=("balanced", "uniform"), default="balanced")
    p.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1")); rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0")); requested = torch.device(a.device)
    if requested.type != "cuda": p.error("formal M4 training uses M3's CUDA voxel rasterizer")
    if world > 1: dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local}" if world > 1 else a.device)
    torch.cuda.set_device(device); torch.manual_seed(a.seed + rank)

    train = InsertedPolicyDataset(
        a.train_index, a.vocabulary, a.text_catalog, dataset_root=a.dataset_root
    )
    val = InsertedPolicyDataset(
        a.val_index, a.vocabulary, a.text_catalog, dataset_root=a.dataset_root
    )
    weights = [float(row.get("sample_weight", 1.)) for row in train.rows]
    if a.sampling == "balanced":
        sampler = (DistributedWeightedSampler(weights, world, rank, a.seed) if world > 1 else
                   WeightedRandomSampler(weights, len(weights), replacement=True))
    else:
        sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True,
                                     seed=a.seed) if world > 1 else None
    val_sampler = DistributedSampler(val, num_replicas=world, rank=rank,
                                     shuffle=False) if world > 1 else None
    loader = DataLoader(train, batch_size=a.batch_size, sampler=sampler,
                        shuffle=sampler is None, num_workers=a.workers,
                        pin_memory=True, collate_fn=collate_inserted_policy)
    val_loader = DataLoader(val, batch_size=a.batch_size, sampler=val_sampler,
                            shuffle=False, num_workers=a.workers, pin_memory=True,
                            collate_fn=collate_inserted_policy)

    renderer = load_renderer(a.m3_checkpoint)
    text_cache = load_file(str(a.text_cache), device="cpu")
    text_hidden = int(text_cache["encoder_hidden"].shape[-1])
    cfg = InsertedPolicyArgs(num_policy_blocks=a.policy_blocks, text_hidden_size=text_hidden)
    raw = InsertedInhabitantPolicy(renderer, cfg).to(device)
    codec = RendererCodec(load_file(str(a.pixel_vae))).to(device).eval()
    optimizer = torch.optim.AdamW(raw.families.parameters(), lr=a.lr)
    start = 0
    if a.resume:
        checkpoint = torch.load(a.resume, map_location="cpu", weights_only=False)
        raw.families.load_state_dict(checkpoint["families"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"]); start = int(checkpoint["step"])
    model = (DistributedDataParallel(raw, device_ids=[local], broadcast_buffers=False,
                                     find_unused_parameters=True) if world > 1 else raw)
    output = Path(a.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(
            {"policy": asdict(cfg), "training": vars(a),
             "contract": "8 completed frames -> 8-action chunk"}, indent=2))
    if world > 1: dist.barrier()

    iterator = iter(loader); epoch = 0
    horizon_weights = torch.arange(8, 0, -1, device=device)
    for step in range(start + 1, a.steps + 1):
        try: batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None and hasattr(sampler, "set_epoch"): sampler.set_epoch(epoch)
            iterator = iter(loader); batch = next(iterator)
        conditions = move_conditions(batch["conditions"], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
            latent = codec.encode(batch["rgb"].to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.precision == "bf16"):
            shared, shared_mask = lookup_text(text_cache, batch["shared_text_id"], device, latent.dtype)
            current, current_mask = lookup_text(text_cache, batch["current_text_id"], device, latent.dtype)
            _, logits = model(latent, conditions, batch["family_id"].to(device),
                              batch["profile_id"].to(device), shared_text=shared,
                              shared_text_mask=shared_mask, current_text=current,
                              current_text_mask=current_mask,
                              policy_indices=batch["policy_indices"].to(device))
            loss, parts = raw.loss(
                logits, batch["target_actions"].to(device), batch["valid_mask"].to(device),
                horizon_weights=horizon_weights,
                sample_weight=batch["sample_weight"].to(device) if a.sampling == "uniform" else None)
        if not torch.isfinite(loss): raise FloatingPointError(f"non-finite M4 loss at step {step}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(raw.families.parameters(), 1.); optimizer.step()
        reported = loss.detach()
        if world > 1: dist.all_reduce(reported); reported /= world
        if rank == 0: print(f"step={step} loss={reported.item():.6f}", flush=True)
        if step % a.validate_every == 0 or step == a.steps:
            metric = evaluate(raw, codec, text_cache, val_loader, device, a.val_batches, a.precision)
            if world > 1: dist.all_reduce(metric); metric /= world
            if rank == 0:
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, "structured_loss": float(metric)}) + "\n")
        if rank == 0 and (step % a.save_every == 0 or step == a.steps):
            destination = output / f"step_{step:07d}.pt"; temporary = destination.with_suffix(".tmp")
            torch.save({"families": raw.families.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "config": asdict(cfg),
                        "m3_checkpoint": str(Path(a.m3_checkpoint).resolve())}, temporary)
            temporary.replace(destination)
    if world > 1: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__": main()
