#!/usr/bin/env python3
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data import TextAgentFillDataset
from plot.models import GeometryConditionedFillArgs, decode_voxel_prediction
from plot.visualization import render_voxel_cameras
from plot.training import GeometryFillTrainer, GeometryFillTrainerConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train geometry-conditioned 48-cubed M1 fill")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path)
    parser.add_argument("--geometry-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--warm-start-fill", type=Path,
        help="Load fill-model weights while starting a fresh optimizer/run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="val_id")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch")
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40_000)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=1)
    parser.add_argument("--visualize-every", type=int, default=1_000)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--splat-channels", type=int, default=24)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--geometry-patch-height", type=int, default=12)
    parser.add_argument("--geometry-patch-width", type=int, default=20)
    parser.add_argument("--geometry-output-height", type=int, default=24)
    parser.add_argument("--geometry-output-width", type=int, default=40)
    parser.add_argument(
        "--geometry-multiscale-highres", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-distance", type=float, default=32.0)
    parser.add_argument("--air-weight", type=float, default=0.25)
    parser.add_argument("--visible-voxel-weight", type=float, default=2.0)
    parser.add_argument("--occupancy-weight", type=float, default=0.5)
    parser.add_argument("--pixel-semantic-weight", type=float, default=0.0)
    parser.add_argument("--pixel-semantic-class-weight", type=float, default=1.0)
    parser.add_argument("--semantic-head-learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--pixel-semantic-splat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--highres-pixel-semantic-head", action=argparse.BooleanOptionalAction, default=False,
        help="Predict first-hit block classes on the encoder's 90x160 feature grid",
    )
    parser.add_argument(
        "--highres-pixel-semantic-splat", action=argparse.BooleanOptionalAction, default=False,
        help="Lift 90x160 semantic probabilities into the 3D decoder feature volume",
    )
    parser.add_argument(
        "--direct-highres-visible-head", action=argparse.BooleanOptionalAction, default=False,
        help="Place per-pixel semantic candidates directly into the full 48-cubed output",
    )
    parser.add_argument(
        "--highres-point-refinement", action=argparse.BooleanOptionalAction, default=False,
        help="Refine the upsampled point map on the 90x160 semantic feature grid",
    )
    parser.add_argument("--highres-point-weight", type=float, default=0.0)
    parser.add_argument(
        "--pixel-semantic-upweight-raw-ids", default="",
        help="Comma-separated raw content IDs receiving extra pixel CE weight",
    )
    parser.add_argument("--visible-only", action="store_true")
    parser.add_argument("--ray-free-weight", type=float, default=0.0)
    parser.add_argument("--visible-surface-occupancy-weight", type=float, default=1.0)
    parser.add_argument("--visible-material-class-weight", type=float, default=1.0)
    parser.add_argument(
        "--visible-material-upweight-raw-ids", default="",
        help="Comma-separated raw IDs receiving extra visible 3D material CE weight",
    )
    parser.add_argument("--explicit-occupancy", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--visibility-evidence", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--free-space-samples", type=int, default=8)
    parser.add_argument(
        "--adaptive-visibility-fusion", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--evidence-learning-rate-multiplier", type=float, default=1.0)
    parser.add_argument("--projection-weight", type=float, default=0.0)
    parser.add_argument("--projection-surface-weight", type=float, default=1.0)
    parser.add_argument("--projection-free-space-weight", type=float, default=0.5)
    parser.add_argument("--projection-silhouette-weight", type=float, default=0.25)
    parser.add_argument("--projection-depth-weight", type=float, default=0.25)
    parser.add_argument("--projection-edge-weight", type=float, default=3.0)
    parser.add_argument("--projection-height", type=int, default=24)
    parser.add_argument("--projection-width", type=int, default=40)
    parser.add_argument("--projection-samples", type=int, default=64)
    parser.add_argument(
        "--projection-alpha-reference-step", type=float, default=0.0,
        help="Experimental alpha segment length in blocks; 0 preserves per-sample alpha",
    )
    parser.add_argument("--teacher-splat-steps", type=int, default=2_000)
    parser.add_argument("--teacher-splat-decay-steps", type=int, default=3_000)
    parser.add_argument("--freeze-geometry", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-geometry-multiscale-only", action="store_true",
        help="Freeze the geometry backbone except its high-resolution adapters and fusion block",
    )
    parser.add_argument(
        "--train-highres-semantic-only", action="store_true",
        help="Freeze geometry and the 3D decoder; train only the 90x160 semantic branch",
    )
    parser.add_argument(
        "--train-direct-visible-only", action="store_true",
        help="Freeze the old model and train only the full-resolution per-ray residual",
    )
    parser.add_argument(
        "--train-highres-point-only", action="store_true",
        help="Freeze the old model and train only the high-resolution point residual",
    )
    parser.add_argument(
        "--geometry-multiscale-learning-rate-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--full-resolution-surface", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--wandb-project", default="plot")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
    return world_size, local_rank, int(os.environ.get("RANK", "0"))


def make_loader(dataset, batch_size, workers, distributed, shuffle):
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


def save_visualization(output, batch, path: Path, air_class: int, step: int) -> tuple[Path, Path]:
    valid = batch["target_valid"].bool()
    target = torch.where(valid, batch["target"], torch.full_like(batch["target"], air_class))
    predicted = decode_voxel_prediction(output, air_class)
    predicted = torch.where(valid, predicted, torch.full_like(predicted, air_class))
    gt_path = render_voxel_cameras(
        target[0].cpu().numpy(), batch["camera_position"][0].cpu().numpy(),
        batch["camera_direction"][0].cpu().numpy(), path / f"step_{step:08d}_gt.png",
        title=f"Ground truth | step {step}", air_class=air_class, render_stride=2,
    )
    pred_path = render_voxel_cameras(
        predicted[0].cpu().numpy(), output["camera_position"][0].float().cpu().numpy(),
        output["camera_direction"][0].float().cpu().numpy(),
        path / f"step_{step:08d}_pred.png", title=f"Geometry M1 | step {step}",
        air_class=air_class, render_stride=2,
    )
    return gt_path, pred_path


def main() -> None:
    args = parse_args()
    if args.resume is None and args.warm_start_fill is None and args.geometry_checkpoint is None:
        raise ValueError("a geometry checkpoint or fill warm start is required for a new run")
    if args.resume is not None and args.warm_start_fill is not None:
        raise ValueError("--resume and --warm-start-fill are mutually exclusive")
    world_size, local_rank, rank = setup_distributed()
    distributed = world_size > 1
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    dataset_kwargs = dict(
        vocabulary=args.vocabulary, samples_per_agent=1, max_agents=args.num_views,
        num_views=args.num_views, initial_only=True, episode_index=args.episode_index,
        canonical_yaw=True,
    )
    train_dataset = TextAgentFillDataset(args.dataset_root, split=args.train_split, **dataset_kwargs)
    val_dataset = TextAgentFillDataset(args.dataset_root, split=args.val_split, **dataset_kwargs)
    air_class = train_dataset.vocabulary.class_to_raw.index(126)
    train_loader, train_sampler = make_loader(
        train_dataset, args.batch_size, args.num_workers, distributed, True
    )
    val_loader, val_sampler = make_loader(
        val_dataset, 1, min(1, args.num_workers), distributed, False
    )
    model_args = GeometryConditionedFillArgs(
        num_block_classes=train_dataset.vocabulary.size, air_class=air_class,
        max_distance=args.max_distance,
        base_channels=args.base_channels, splat_channels=args.splat_channels,
        geometry_patch_height=args.geometry_patch_height,
        geometry_patch_width=args.geometry_patch_width,
        geometry_output_height=args.geometry_output_height,
        geometry_output_width=args.geometry_output_width,
        max_views=args.num_views, full_resolution_surface=args.full_resolution_surface,
        explicit_occupancy=args.explicit_occupancy,
        visibility_evidence=args.visibility_evidence,
        free_space_samples=args.free_space_samples,
        adaptive_visibility_fusion=args.adaptive_visibility_fusion,
        pixel_semantic_head=args.pixel_semantic_weight > 0,
        pixel_semantic_splat=args.pixel_semantic_splat,
        geometry_multiscale_highres=args.geometry_multiscale_highres,
        highres_pixel_semantic_head=args.highres_pixel_semantic_head,
        highres_pixel_semantic_splat=args.highres_pixel_semantic_splat,
        direct_highres_visible_head=args.direct_highres_visible_head,
        highres_point_refinement=args.highres_point_refinement,
    )
    model = model_args.build().to(device)
    if args.warm_start_fill is not None:
        checkpoint = torch.load(args.warm_start_fill, map_location="cpu", weights_only=False)
        warm_state = dict(checkpoint["model"])
        position_key = "geometry.patch_position"
        if position_key in warm_state and warm_state[position_key].shape != model.state_dict()[position_key].shape:
            old_args = checkpoint.get("model_args", {})
            old_height = int(old_args.get("geometry_patch_height", 12))
            old_width = int(old_args.get("geometry_patch_width", 20))
            position = warm_state[position_key]
            if old_height * old_width != position.shape[0]:
                raise RuntimeError("cannot infer the warm-start geometry patch grid")
            position = position.T.reshape(1, position.shape[1], old_height, old_width)
            position = F.interpolate(
                position, size=(args.geometry_patch_height, args.geometry_patch_width),
                mode="bilinear", align_corners=False,
            )
            warm_state[position_key] = position.reshape(position.shape[1], -1).T
        incompatible = model.load_state_dict(warm_state, strict=False)
        allowed_missing = {
            "occupancy_residual.weight", "occupancy_residual.bias",
            "surface_evidence_gain_raw", "free_space_evidence_gain_raw",
        }
        unexpected_missing = {
            key for key in incompatible.missing_keys
            if key not in allowed_missing and not key.startswith("visibility_refiner.")
            and not key.startswith("geometry.semantic_head.")
            and key != "pixel_semantic_embedding"
            and not key.startswith("geometry.shallow_adapter.")
            and not key.startswith("geometry.middle_adapter.")
            and not key.startswith("geometry.multiscale_fusion.")
            and not key.startswith("geometry.highres_semantic_")
            and key != "highres_pixel_semantic_embedding"
            and key != "direct_visible_embedding"
            and not key.startswith("direct_visible_classifier.")
            and not key.startswith("geometry.highres_point_refiner.")
        }
        if incompatible.unexpected_keys or unexpected_missing:
            raise RuntimeError(
                f"incompatible warm start: missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    if args.geometry_checkpoint is not None and args.resume is None:
        model.load_geometry_checkpoint(args.geometry_checkpoint)
    if args.train_highres_point_only:
        model.train_highres_point_only()
    elif args.train_direct_visible_only:
        model.train_direct_visible_only()
    elif args.train_highres_semantic_only:
        model.train_highres_semantic_only()
    elif args.train_geometry_multiscale_only:
        model.train_geometry_multiscale_only()
    else:
        model.freeze_geometry(args.freeze_geometry)
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], broadcast_buffers=False,
            find_unused_parameters=False,
        )
    semantic_raw_ids = tuple(
        int(value) for value in args.pixel_semantic_upweight_raw_ids.split(",") if value.strip()
    )
    raw_to_class = {
        raw: class_id for class_id, raw in enumerate(train_dataset.vocabulary.class_to_raw)
    }
    missing_semantic_ids = [raw for raw in semantic_raw_ids if raw not in raw_to_class]
    if missing_semantic_ids:
        raise ValueError(f"semantic raw IDs are absent from vocabulary: {missing_semantic_ids}")
    semantic_class_ids = tuple(raw_to_class[raw] for raw in semantic_raw_ids)
    visible_material_raw_ids = tuple(
        int(value) for value in args.visible_material_upweight_raw_ids.split(",") if value.strip()
    )
    missing_visible_ids = [raw for raw in visible_material_raw_ids if raw not in raw_to_class]
    if missing_visible_ids:
        raise ValueError(f"visible-material raw IDs are absent from vocabulary: {missing_visible_ids}")
    visible_material_class_ids = tuple(raw_to_class[raw] for raw in visible_material_raw_ids)
    config = GeometryFillTrainerConfig(
        learning_rate=args.learning_rate, air_class=air_class, air_weight=args.air_weight,
        visible_voxel_weight=args.visible_voxel_weight, occupancy_weight=args.occupancy_weight,
        pixel_semantic_weight=args.pixel_semantic_weight,
        pixel_semantic_class_ids=semantic_class_ids,
        pixel_semantic_class_weight=args.pixel_semantic_class_weight,
        highres_point_weight=args.highres_point_weight,
        semantic_head_learning_rate_multiplier=args.semantic_head_learning_rate_multiplier,
        geometry_multiscale_learning_rate_multiplier=(
            args.geometry_multiscale_learning_rate_multiplier
        ),
        visible_only=args.visible_only,
        ray_free_weight=args.ray_free_weight,
        visible_surface_occupancy_weight=args.visible_surface_occupancy_weight,
        visible_material_class_ids=visible_material_class_ids,
        visible_material_class_weight=args.visible_material_class_weight,
        projection_weight=args.projection_weight,
        projection_surface_weight=args.projection_surface_weight,
        projection_free_space_weight=args.projection_free_space_weight,
        projection_silhouette_weight=args.projection_silhouette_weight,
        projection_depth_weight=args.projection_depth_weight,
        projection_edge_weight=args.projection_edge_weight,
        projection_height=args.projection_height, projection_width=args.projection_width,
        projection_samples=args.projection_samples,
        projection_alpha_reference_step=args.projection_alpha_reference_step,
        projection_max_distance=args.max_distance,
        evidence_learning_rate_multiplier=args.evidence_learning_rate_multiplier,
        teacher_splat_steps=args.teacher_splat_steps,
        teacher_splat_decay_steps=args.teacher_splat_decay_steps,
        precision=args.precision, device=device,
    )
    trainer = GeometryFillTrainer(model, config)
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
            runtime_metrics = dict(trainer.last_metrics)
            if torch.cuda.is_available():
                gib = float(1024 ** 3)
                runtime_metrics["cuda_memory_allocated_gib"] = (
                    torch.cuda.memory_allocated(device) / gib
                )
                runtime_metrics["cuda_max_memory_reserved_gib"] = (
                    torch.cuda.max_memory_reserved(device) / gib
                )
            print(
                " ".join(
                    [f"step={step}"]
                    + [f"{name}={value:.5f}" for name, value in runtime_metrics.items()]
                ), flush=True,
            )
            if run:
                run.log({f"train/{name}": value for name, value in runtime_metrics.items()}, step=step)
        if args.eval_every and step % args.eval_every == 0:
            # Fixed validation examples across checkpoints for comparable curves.
            val_iterator = iter(val_loader)
            collected = []
            for _ in range(args.eval_batches):
                val_batch, val_iterator, _ = next_batch(val_iterator, val_loader, val_sampler, 0)
                metrics, output, device_batch = trainer.evaluate_step(val_batch, return_output=True)
                collected.append(metrics)
            metrics = {key: sum(m[key] for m in collected) / len(collected) for key in metrics}
            if distributed:
                values = torch.tensor(list(metrics.values()), device=device)
                dist.all_reduce(values)
                values /= world_size
                metrics = dict(zip(metrics, values.cpu().tolist()))
            if rank == 0:
                print("val " + " ".join(f"{key}={value:.5f}" for key, value in metrics.items()), flush=True)
                log = {f"val/{key}": value for key, value in metrics.items()}
                if args.visualize_every and step % args.visualize_every == 0:
                    gt_path, pred_path = save_visualization(
                        output, device_batch, args.output_dir / "visualizations", air_class, step
                    )
                    if run:
                        import wandb

                        log["val/ground_truth"] = wandb.Image(str(gt_path))
                        log["val/prediction"] = wandb.Image(str(pred_path))
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
