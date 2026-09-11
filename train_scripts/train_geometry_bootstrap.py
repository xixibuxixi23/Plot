#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data import TextAgentFillDataset
from plot.models import GeometryBootstrapArgs
from plot.training import GeometryBootstrapTrainer, GeometryBootstrapTrainerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train first-view-gauge M1 geometry bootstrap")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val_id")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--visualize-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--feature-dim", type=int, default=192)
    parser.add_argument("--attention-heads", type=int, default=6)
    parser.add_argument("--stages", type=int, default=3)
    parser.add_argument("--output-height", type=int, default=24)
    parser.add_argument("--output-width", type=int, default=40)
    parser.add_argument(
        "--multiscale-highres", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--patch-height", type=int, default=12)
    parser.add_argument("--patch-width", type=int, default=20)
    parser.add_argument("--ray-samples", type=int, default=64)
    parser.add_argument("--max-distance", type=float, default=32.0)
    parser.add_argument("--supervised-height-fraction", type=float, default=0.82)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--wandb-project", default="plot")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return world_size, local_rank, int(os.environ.get("RANK", "0"))


def make_loader(dataset, batch_size, workers, distributed, shuffle):
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
        num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
        drop_last=shuffle, prefetch_factor=2 if workers > 0 else None,
    ), sampler


def next_batch(iterator, loader, sampler, epoch):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def save_geometry_visualization(output, target, images, path: Path) -> Path:
    views = images.shape[1]
    fig, axes = plt.subplots(views, 5, figsize=(13, 3.1 * views), dpi=130)
    axes = np.asarray(axes).reshape(views, 5)
    for view in range(views):
        panels = (
            (images[0, view].float().cpu().permute(1, 2, 0).clamp(0, 1), "input", None),
            (target["visibility"][0, view].cpu(), "GT hit", (0, 1)),
            (output["visibility_logits"][0, view].float().sigmoid().cpu(), "pred hit", (0, 1)),
            (target["depth"][0, view].cpu(), "GT depth / max", (0, 1)),
            (output["depth"][0, view].float().cpu(), "pred depth / max", (0, 1)),
        )
        for axis, (image, title, limits) in zip(axes[view], panels):
            if limits is None:
                axis.imshow(image)
            else:
                axis.imshow(image, cmap="viridis", vmin=limits[0], vmax=limits[1])
            axis.set_title(f"view {view}: {title}")
            axis.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return path


def main():
    args = parse_args()
    if args.resume is not None and args.warm_start is not None:
        raise ValueError("--resume and --warm-start are mutually exclusive")
    world_size, local_rank, rank = setup_distributed()
    distributed = world_size > 1
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    train_dataset = TextAgentFillDataset(
        args.dataset_root, args.vocabulary, split=args.train_split, samples_per_agent=1,
        max_agents=args.num_views, num_views=args.num_views, initial_only=True,
        episode_index=args.episode_index,
    )
    val_dataset = TextAgentFillDataset(
        args.dataset_root, args.vocabulary, split=args.val_split, samples_per_agent=1,
        max_agents=args.num_views, num_views=args.num_views, initial_only=True,
        episode_index=args.episode_index,
    )
    try:
        air_class = train_dataset.vocabulary.class_to_raw.index(126)
    except ValueError as error:
        raise ValueError("vocabulary does not contain Minetest CONTENT_AIR=126") from error
    train_loader, train_sampler = make_loader(
        train_dataset, args.batch_size, args.num_workers, distributed, True
    )
    val_loader, val_sampler = make_loader(
        val_dataset, 1, min(1, args.num_workers), distributed, False
    )
    model_args = GeometryBootstrapArgs(
        feature_dim=args.feature_dim, attention_heads=args.attention_heads,
        stages=args.stages, patch_height=args.patch_height, patch_width=args.patch_width,
        output_height=args.output_height, output_width=args.output_width,
        max_views=args.num_views, multiscale_highres=args.multiscale_highres,
    )
    model = model_args.build().to(device)
    if args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        state = dict(checkpoint["model"])
        position = state["patch_position"]
        old_args = checkpoint.get("model_args", {})
        old_height = int(old_args.get("patch_height", 12))
        old_width = int(old_args.get("patch_width", 20))
        if old_height * old_width != position.shape[0]:
            raise RuntimeError("cannot infer warm-start patch grid")
        if (old_height, old_width) != (args.patch_height, args.patch_width):
            position = position.T.reshape(1, position.shape[1], old_height, old_width)
            position = F.interpolate(
                position, size=(args.patch_height, args.patch_width),
                mode="bilinear", align_corners=False,
            )
            state["patch_position"] = position.reshape(position.shape[1], -1).T
        incompatible = model.load_state_dict(state, strict=False)
        allowed_missing_prefixes = (
            "shallow_adapter.", "middle_adapter.", "multiscale_fusion.",
        )
        unexpected_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if incompatible.unexpected_keys or unexpected_missing:
            raise RuntimeError(
                f"incompatible warm start: missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    trainer_config = GeometryBootstrapTrainerConfig(
        learning_rate=args.learning_rate, precision=args.precision, device=device,
        ray_samples=args.ray_samples, max_distance=args.max_distance, air_class=air_class,
        supervised_height_fraction=args.supervised_height_fraction,
    )
    trainer = GeometryBootstrapTrainer(model, trainer_config)
    if args.resume:
        trainer.load(args.resume)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode,
            config={
                **vars(args), "world_size": world_size,
                "effective_batch_size": args.batch_size * world_size * args.gradient_accumulation,
            },
        )
    if rank == 0:
        print(
            f"train_samples={len(train_dataset)} val_samples={len(val_dataset)} "
            f"world_size={world_size} per_gpu_batch={args.batch_size} "
            f"effective_batch={args.batch_size * world_size * args.gradient_accumulation}",
            flush=True,
        )
    train_iterator, val_iterator, epoch = iter(train_loader), iter(val_loader), 0
    while trainer.step < args.steps:
        for micro in range(args.gradient_accumulation):
            batch, train_iterator, epoch = next_batch(
                train_iterator, train_loader, train_sampler, epoch
            )
            update = micro == args.gradient_accumulation - 1
            sync = contextlib.nullcontext()
            if distributed and not update:
                sync = model.no_sync()
            with sync:
                trainer.train_step(batch, update=update, loss_divisor=args.gradient_accumulation)
        step = trainer.step
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            metrics = {f"train/{key}": value for key, value in trainer.last_metrics.items()}
            metrics["train/step"] = step
            print(
                " ".join([f"step={step}"] + [f"{key}={value:.5f}" for key, value in trainer.last_metrics.items()]),
                flush=True,
            )
            if run:
                run.log(metrics, step=step)
        if args.eval_every and step % args.eval_every == 0:
            val_batch, val_iterator, _ = next_batch(val_iterator, val_loader, val_sampler, 0)
            metrics, output, target, device_batch = trainer.evaluate_step(
                val_batch, return_output=True
            )
            if distributed:
                values = torch.tensor(list(metrics.values()), device=device)
                dist.all_reduce(values)
                values /= world_size
                metrics = dict(zip(metrics, values.cpu().tolist()))
            if rank == 0:
                print("val " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()), flush=True)
                log = {f"val/{key}": value for key, value in metrics.items()}
                if args.visualize_every and step % args.visualize_every == 0:
                    path = save_geometry_visualization(
                        output, target, device_batch["images"],
                        args.output_dir / "visualizations" / f"step_{step:08d}.png",
                    )
                    if run:
                        import wandb

                        log["val/geometry"] = wandb.Image(str(path))
                if run:
                    run.log(log, step=step)
        if rank == 0 and args.save_every and step % args.save_every == 0:
            trainer.save(
                args.output_dir / f"checkpoint_{step:08d}.pt",
                {"model_args": vars(model_args), "train_args": vars(args)},
            )
    if rank == 0:
        trainer.save(
            args.output_dir / "checkpoint_final.pt",
            {"model_args": vars(model_args), "train_args": vars(args)},
        )
        if run:
            run.finish()
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
