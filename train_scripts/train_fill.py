from __future__ import annotations

import argparse
import contextlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data import TextAgentFillDataset
from plot.models import FillNetworkArgs
from plot.visualization import render_voxel_cameras
from plot.training import FillTrainer, FillTrainerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train two-view PLOT M1 initialization")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val_id")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100_000, help="Optimizer steps")
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--visualize-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--projection-silhouette-weight", type=float, default=0.0)
    parser.add_argument("--projection-depth-weight", type=float, default=0.0)
    parser.add_argument("--projection-start-step", type=int, default=0)
    parser.add_argument("--projection-warmup-steps", type=int, default=0)
    parser.add_argument("--projection-height", type=int, default=24)
    parser.add_argument("--projection-width", type=int, default=40)
    parser.add_argument("--projection-samples", type=int, default=64)
    parser.add_argument("--projection-max-distance", type=float, default=32.0)
    parser.add_argument(
        "--projection-batch-size", type=int, default=0,
        help="Per-rank examples receiving projection loss; 0 uses the full batch",
    )
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260908)
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
    # Let DistributedSampler pad tiny/non-divisible datasets so every DDP rank
    # executes the same number of collectives. DataLoader owns batch dropping.
    sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=False) if distributed else None
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle and sampler is None, sampler=sampler,
        num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
        drop_last=shuffle, prefetch_factor=2 if workers > 0 else None,
    )
    return loader, sampler


def next_batch(iterator, loader, sampler, epoch):
    try:
        return next(iterator), iterator, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        iterator = iter(loader)
        return next(iterator), iterator, epoch


def main():
    args = parse_args()
    world_size, local_rank, rank = setup_distributed()
    distributed = world_size > 1
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    train_dataset = TextAgentFillDataset(
        args.dataset_root, args.vocabulary, split=args.train_split,
        samples_per_agent=1, max_agents=args.num_views, num_views=args.num_views,
        initial_only=True,
    )
    val_dataset = TextAgentFillDataset(
        args.dataset_root, args.vocabulary, split=args.val_split,
        samples_per_agent=1, max_agents=args.num_views, num_views=args.num_views,
        initial_only=True,
    )
    try:
        air_class = train_dataset.vocabulary.class_to_raw.index(126)
    except ValueError as error:
        raise ValueError("vocabulary does not contain Minetest CONTENT_AIR=126") from error
    train_loader, train_sampler = make_loader(
        train_dataset, args.batch_size, args.num_workers, distributed, True
    )
    val_loader, val_sampler = make_loader(val_dataset, 1, min(1, args.num_workers), distributed, False)

    model_args = FillNetworkArgs(
        num_block_classes=train_dataset.vocabulary.size,
        base_channels=args.base_channels, max_views=args.num_views,
    )
    model = model_args.build().to(device)
    if distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    trainer = FillTrainer(model, FillTrainerConfig(
        learning_rate=args.learning_rate, precision=args.precision, device=device,
        projection_silhouette_weight=args.projection_silhouette_weight,
        projection_depth_weight=args.projection_depth_weight,
        projection_start_step=args.projection_start_step,
        projection_warmup_steps=args.projection_warmup_steps,
        projection_height=args.projection_height, projection_width=args.projection_width,
        projection_samples=args.projection_samples,
        projection_max_distance=args.projection_max_distance, air_class=air_class,
        projection_batch_size=args.projection_batch_size,
    ))
    if args.resume:
        trainer.load(args.resume)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run = None
    if rank == 0 and args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("wandb is required unless --wandb-mode disabled") from error
        run = wandb.init(
            project=args.wandb_project, name=args.wandb_name, mode=args.wandb_mode,
            config={**vars(args), "world_size": world_size,
                    "effective_batch_size": args.batch_size * world_size * args.gradient_accumulation},
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
                trainer.train_step(
                    batch, update=update, loss_divisor=args.gradient_accumulation
                )
        step = trainer.step
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            metrics = {f"train/{k}": v for k, v in trainer.last_metrics.items()}
            metrics["train/step"] = step
            print(" ".join([f"step={step}"] + [f"{k}={v:.5f}" for k, v in trainer.last_metrics.items()]), flush=True)
            if run:
                run.log(metrics, step=step)

        if args.eval_every and step % args.eval_every == 0:
            val_batch, val_iterator, _ = next_batch(val_iterator, val_loader, val_sampler, 0)
            metrics, output, device_batch = trainer.evaluate_step(val_batch, return_output=True)
            if distributed:
                values = torch.tensor(list(metrics.values()), device=device)
                dist.all_reduce(values)
                values /= world_size
                metrics = dict(zip(metrics, values.cpu().tolist()))
            if rank == 0:
                print("val " + " ".join(f"{k}={v:.5f}" for k, v in metrics.items()), flush=True)
                log = {f"val/{k}": v for k, v in metrics.items()}
                if args.visualize_every and step % args.visualize_every == 0:
                    vis_dir = args.output_dir / "visualizations"
                    gt_path = render_voxel_cameras(
                        device_batch["target"][0].cpu().numpy(),
                        device_batch["camera_position"][0].cpu().numpy(),
                        device_batch["camera_direction"][0].cpu().numpy(),
                        vis_dir / f"step_{step:08d}_gt.png", title=f"Ground truth | step {step}",
                        air_class=air_class, render_stride=2,
                    )
                    pred_path = render_voxel_cameras(
                        output["voxel_logits"].argmax(1)[0].cpu().numpy(),
                        output["camera_position"][0].float().cpu().numpy(),
                        output["camera_direction"][0].float().cpu().numpy(),
                        vis_dir / f"step_{step:08d}_pred.png", title=f"M1 prediction | step {step}",
                        air_class=air_class, render_stride=2,
                    )
                    if run:
                        import wandb
                        log.update({"val/ground_truth": wandb.Image(str(gt_path)),
                                    "val/prediction": wandb.Image(str(pred_path))})
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
