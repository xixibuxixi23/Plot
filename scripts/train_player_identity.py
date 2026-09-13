"""Pretrain Minecraft skin/render identity features with contrastive pairs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from plot.data.player_identity_dataset import PlayerIdentityDataset
from plot.models.player_identity import PlayerIdentityEncoder, multi_positive_contrastive_loss
from plot.checkpoint_io import staged_torch_save


def _gather_with_local_gradient(value, world, rank):
    if world == 1:
        return value
    gathered = [torch.empty_like(value) for _ in range(world)]
    dist.all_gather(gathered, value.detach())
    gathered[rank] = value
    return torch.cat(gathered)


def _gather_plain(value, world):
    if world == 1:
        return value
    gathered = [torch.empty_like(value) for _ in range(world)]
    dist.all_gather(gathered, value)
    return torch.cat(gathered)


@torch.no_grad()
def _retrieval_accuracy(crop, reference, identity, valid):
    keep = valid.bool()
    if not keep.any():
        return crop.new_zeros(())
    scores = crop[keep] @ reference.transpose(0, 1)
    predicted = identity[scores.argmax(1)]
    return (predicted == identity[keep]).float().mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--val-window-index")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--min-player-pixels", type=int, default=64)
    parser.add_argument(
        "--canonical-only",
        action="store_true",
        help="Bootstrap from augmented canonical views without expensive video seeks",
    )
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--val-batches", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb-project", default="plot-m3")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.validate_every) < 1:
        parser.error("steps, batch size and intervals must be positive")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed + rank)

    dataset = PlayerIdentityDataset(
        args.dataset_root,
        args.window_index,
        min_pixels=args.min_player_pixels,
        canonical_only=args.canonical_only,
    )
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed
        )
        if world > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = None
    if args.val_window_index:
        val_dataset = PlayerIdentityDataset(
            args.dataset_root,
            args.val_window_index,
            min_pixels=args.min_player_pixels,
            canonical_only=args.canonical_only,
        )
        val_sampler = (
            DistributedSampler(
                val_dataset, num_replicas=world, rank=rank, shuffle=False
            )
            if world > 1
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    raw_model = PlayerIdentityEncoder().to(device)
    model = (
        DistributedDataParallel(raw_model, device_ids=[local_rank])
        if world > 1
        else raw_model
    )
    optimizer = torch.optim.AdamW(raw_model.parameters(), lr=args.lr, weight_decay=0.01)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.json").write_text(json.dumps(vars(args), indent=2))
    if world > 1:
        dist.barrier()
    run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            dir=str(output),
            config={**vars(args), "world_size": world},
        )
        (output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url}, indent=2)
        )

    iterator, epoch = iter(loader), 0
    for step in range(start_step + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        crop = batch["crop"].to(device, non_blocking=True)
        reference = batch["reference"].to(device, non_blocking=True)
        identity = batch["identity"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            crop_embedding, reference_embedding = model(crop, reference)
            all_crop = _gather_with_local_gradient(crop_embedding, world, rank)
            all_reference = _gather_with_local_gradient(reference_embedding, world, rank)
            all_identity = _gather_plain(identity, world)
            all_valid = _gather_plain(valid, world)
            loss = multi_positive_contrastive_loss(
                all_crop,
                all_reference,
                all_identity,
                all_valid,
                temperature=args.temperature,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite identity loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            accuracy = _retrieval_accuracy(
                all_crop.float(), all_reference.float(), all_identity, all_valid
            )
            valid_fraction = all_valid.float().mean()
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            metrics = {
                "train/loss": float(loss.detach()),
                "train/retrieval_at_1": float(accuracy),
                "train/valid_fraction": float(valid_fraction),
            }
            print(
                f"step={step} loss={metrics['train/loss']:.6f} "
                f"retrieval_at_1={metrics['train/retrieval_at_1']:.4f} "
                f"valid_fraction={metrics['train/valid_fraction']:.4f}",
                flush=True,
            )
            if run:
                run.log(metrics, step=step)

        if val_loader is not None and (step % args.validate_every == 0 or step == args.steps):
            raw_model.eval()
            totals = torch.zeros(3, device=device)
            with torch.no_grad():
                for number, val in enumerate(val_loader):
                    if number >= args.val_batches:
                        break
                    val_crop = val["crop"].to(device, non_blocking=True)
                    val_reference = val["reference"].to(device, non_blocking=True)
                    val_identity = val["identity"].to(device, non_blocking=True)
                    val_valid = val["valid"].to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        crop_embedding, reference_embedding = raw_model(
                            val_crop, val_reference
                        )
                    all_crop = _gather_plain(crop_embedding.float(), world)
                    all_reference = _gather_plain(reference_embedding.float(), world)
                    all_identity = _gather_plain(val_identity, world)
                    all_valid = _gather_plain(val_valid, world)
                    val_loss = multi_positive_contrastive_loss(
                        all_crop,
                        all_reference,
                        all_identity,
                        all_valid,
                        temperature=args.temperature,
                    )
                    totals += torch.stack((
                        val_loss,
                        _retrieval_accuracy(
                            all_crop, all_reference, all_identity, all_valid
                        ),
                        val_loss.new_ones(()),
                    ))
            if world > 1:
                dist.all_reduce(totals)
                totals /= world
            if rank == 0:
                values = {
                    "step": step,
                    "loss": float(totals[0] / totals[2].clamp_min(1)),
                    "retrieval_at_1": float(totals[1] / totals[2].clamp_min(1)),
                }
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps(values) + "\n")
                if run:
                    run.log({f"val/{key}": value for key, value in values.items() if key != "step"}, step=step)
            raw_model.train()

        if rank == 0 and (step % args.save_every == 0 or step == args.steps):
            staged_torch_save(
                {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "embedding_dim": raw_model.embedding_dim,
                    "crop_size": (128, 64),
                },
                output / f"step_{step:07d}.pt",
            )
    if run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
