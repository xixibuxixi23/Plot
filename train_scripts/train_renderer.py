"""Train M3 from accepted continuous TextAgent episodes and a frozen Pixel VAE."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from plot.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec
from plot.checkpoint_io import record_checkpoint_failure, staged_torch_save
from plot.training.renderer_monitoring import render_probe, save_probe_manifest, select_renderer_probes
from plot.training.renderer_trainer import renderer_training_losses


LOSS_NAMES = (
    "total_loss",
    "flow_loss",
    "auxiliary_loss",
    "entity_pixel_l1",
    "entity_pixel_edge",
    "health_pixel_l1",
)


def load_weights(path):
    if Path(path).suffix == ".safetensors":
        return load_file(str(path))
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value.get("model", value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--pixel-vae", required=True)
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument(
        "--warm-start",
        help="Load model weights without optimizer state; intended for compatible architecture changes",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-index")
    parser.add_argument("--val-window-index")
    parser.add_argument("--health-focus-index")
    parser.add_argument(
        "--health-focus-oversample",
        type=int,
        default=1,
        help="Total sampling multiplicity for windows containing a non-full-health target",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--deep-condition-reinjection",
        action="store_true",
        help="Reinject aligned raster, actor, action, resident state, and HUD conditions at every DiT block",
    )
    parser.add_argument(
        "--view-aware-appearance",
        action="store_true",
        help="Warp dense four-view resident references into masked per-block appearance adapters",
    )
    parser.add_argument(
        "--freeze-base-for-appearance",
        action="store_true",
        help="Stage-one training: update only the new dense appearance modules",
    )
    parser.add_argument("--context-frames", type=int, default=65)
    parser.add_argument("--cache-frames", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--voxel-channels", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--actor-channels", type=int, default=16)
    parser.add_argument("--target-views-per-window", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--visualize-every", type=int, default=1000)
    parser.add_argument("--visualization-denoising-steps", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--latent-entity-region-upweight",
        type=float,
        default=0.0,
        help="Extra latent flow weight inside entity masks; 0 keeps flow loss uniform",
    )
    parser.add_argument(
        "--pixel-loss-frames",
        type=int,
        default=2,
        help="Entity-rich and HP-informative future frames decoded per target view",
    )
    parser.add_argument("--entity-pixel-l1-weight", type=float, default=0.5)
    parser.add_argument("--entity-pixel-edge-weight", type=float, default=0.2)
    parser.add_argument("--health-pixel-l1-weight", type=float, default=1.0)
    parser.add_argument("--damaged-health-upweight", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-project", default="plot-m3")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--checkpoint-staging-dir",
        help="Local/PFS directory used to serialize before copying to output (also PLOT_CHECKPOINT_STAGING_DIR)",
    )
    parser.add_argument(
        "--checkpoint-errors",
        choices=("warn", "raise"),
        default="warn",
        help="Keep training and record failures, or stop if a checkpoint cannot be copied",
    )
    args = parser.parse_args()
    if sum(bool(path) for path in (args.resume, args.warm_start, args.backbone_checkpoint)) > 1:
        parser.error("--resume, --warm-start, and --backbone-checkpoint are mutually exclusive")
    if args.freeze_base_for_appearance and not args.view_aware_appearance:
        parser.error("--freeze-base-for-appearance requires --view-aware-appearance")
    if args.freeze_base_for_appearance and args.resume:
        parser.error("use --warm-start for staged appearance training")
    if (
        min(
            args.steps,
            args.save_every,
            args.validate_every,
            args.target_views_per_window,
            args.batch_size,
            args.gradient_accumulation,
        )
        < 1
    ):
        parser.error("steps, save interval, anchors and batch size must be positive")
    if not 1 <= args.pixel_loss_frames < args.context_frames:
        parser.error("pixel-loss-frames must be within the future-frame count")
    if args.health_focus_oversample < 1:
        parser.error("health-focus-oversample must be positive")
    if args.health_focus_oversample > 1 and not args.health_focus_index:
        parser.error("health-focus-oversample above 1 requires --health-focus-index")
    if min(
        args.latent_entity_region_upweight,
        args.entity_pixel_l1_weight,
        args.entity_pixel_edge_weight,
        args.health_pixel_l1_weight,
        args.damaged_health_upweight,
    ) < 0:
        parser.error("region and pixel loss weights must be nonnegative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if world > 1 else args.device)
    if device.type != "cuda":
        parser.error(
            "raw voxel rasterization needs CUDA; use CPU contract tests for smoke validation"
        )
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    compiled_arch = f"sm_{capability[0]}{capability[1]}"
    if capability >= (10, 0) and compiled_arch not in torch.cuda.get_arch_list():
        parser.error(
            f"PyTorch {torch.__version__} was not compiled for {compiled_arch}; "
            "run scripts/check_m3_environment.py on the B200 node"
        )
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed + rank)
    dataset = TextAgentRendererDataset(
        args.dataset_root,
        args.vocabulary,
        context_frames=args.context_frames,
        window_index=args.window_index,
        targets_per_window=args.target_views_per_window,
        entity_region_upweight=args.latent_entity_region_upweight,
        health_focus_index=args.health_focus_index,
        health_focus_oversample=args.health_focus_oversample,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
        if world > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=collate_renderer,
    )
    cfg = RendererArgs(
        dataset.vocabulary.size,
        max(dataset.item_vocabulary.values()) + 1,
        context_frames=args.context_frames,
        cache_frames=args.cache_frames,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        voxel_channels=args.voxel_channels,
        condition_dim=args.condition_dim,
        actor_channels=args.actor_channels,
        deep_condition_reinjection=args.deep_condition_reinjection,
        view_aware_appearance=args.view_aware_appearance,
    )
    with torch.cuda.device(device):
        raw_model = Renderer(cfg).to(device).train()
    if args.backbone_checkpoint:
        report = raw_model.load_2daction_backbone(load_weights(args.backbone_checkpoint))
        if rank == 0:
            print(json.dumps(report))
    codec = RendererCodec(load_weights(args.pixel_vae)).to(device).eval()
    if args.freeze_base_for_appearance:
        for name, parameter in raw_model.named_parameters():
            parameter.requires_grad_(
                name.startswith((
                    "core.appearance_condition_embedder.",
                    "core.appearance_reinjectors.",
                ))
            )
    trainable_parameters = [p for p in raw_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    if rank == 0:
        print(json.dumps({
            "trainable_parameters": sum(p.numel() for p in trainable_parameters),
            "total_parameters": sum(p.numel() for p in raw_model.parameters()),
        }))
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
    elif args.warm_start:
        checkpoint = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        incompatible = raw_model.load_state_dict(checkpoint["model"], strict=False)
        allowed_missing = (
            "core.hud_condition_embedder.",
            "core.condition_reinjectors.",
            "core.appearance_condition_embedder.",
            "core.appearance_reinjectors.",
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible warm start: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        start_step = int(checkpoint.get("step", 0))
        if rank == 0:
            print(json.dumps({
                "warm_start": str(args.warm_start),
                "step": start_step,
                "initialized_parameters": incompatible.missing_keys,
            }))
    model = (
        DistributedDataParallel(raw_model, device_ids=[local_rank], broadcast_buffers=False)
        if world > 1
        else raw_model
    )
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    run_config = {
        "renderer": asdict(cfg),
        "training": vars(args),
        "item_vocabulary": dataset.item_vocabulary,
        "class_to_raw": dataset.vocabulary.class_to_raw,
    }
    if rank == 0:
        (output / "config.json").write_text(json.dumps(run_config, indent=2))
    run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            dir=str(output),
            config={
                **run_config,
                "world_size": world,
                "effective_source_window_batch_size": (
                    args.batch_size * world * args.gradient_accumulation
                ),
                "effective_view_batch_size": (
                    args.batch_size
                    * args.target_views_per_window
                    * world
                    * args.gradient_accumulation
                ),
            },
        )
        (output / "wandb_run.json").write_text(
            json.dumps({"id": run.id, "url": run.url, "project": args.wandb_project}, indent=2)
        )
    val_loader = None
    probes = []
    if args.val_window_index:
        val_dataset = TextAgentRendererDataset(
            args.dataset_root,
            args.vocabulary,
            split="val_id",
            context_frames=args.context_frames,
            window_index=args.val_window_index,
            entity_region_upweight=args.latent_entity_region_upweight,
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world, rank=rank, shuffle=False)
            if world > 1
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_renderer,
        )
        if rank == 0 and args.visualize_every:
            probes = select_renderer_probes(val_dataset)
            save_probe_manifest(probes, output / "visualizations" / "probes.json")
    elif rank == 0 and args.visualize_every:
        print(
            "warning: --visualize-every requires --val-window-index; visualization disabled",
            flush=True,
        )
    iterator = iter(loader)
    epoch = 0
    loss_kwargs = {
        "frames_per_sample": args.pixel_loss_frames,
        "entity_pixel_l1_weight": args.entity_pixel_l1_weight,
        "entity_pixel_edge_weight": args.entity_pixel_edge_weight,
        "health_pixel_l1_weight": args.health_pixel_l1_weight,
        "damaged_health_upweight": args.damaged_health_upweight,
    }
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated = {name: 0.0 for name in LOSS_NAMES}
        for micro in range(args.gradient_accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            rgb = batch["rgb"].to(device)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"),
            ):
                latent = codec.encode(rgb)
            conditions = {k: v.to(device) for k, v in batch["conditions"].items()}
            sync = (
                model.no_sync()
                if world > 1 and micro + 1 < args.gradient_accumulation
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    losses = renderer_training_losses(
                        model,
                        codec,
                        latent,
                        conditions,
                        rgb,
                        batch["pixel_region_mask"].to(device),
                        region_weight=batch["region_weight"].to(device),
                        **loss_kwargs,
                    )
                    loss = losses["total_loss"] / args.gradient_accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite M3 loss at step {step}")
                loss.backward()
            for name in LOSS_NAMES:
                accumulated[name] += float(losses[name].detach()) / args.gradient_accumulation
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        optimizer.step()
        reduced = torch.tensor([accumulated[name] for name in LOSS_NAMES], device=device)
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
        log_now = step == 1 or step % args.log_every == 0
        if log_now:
            local_memory = torch.tensor(
                [
                    torch.cuda.memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30,
                ],
                device=device,
            )
            if world > 1:
                memory_by_rank = [torch.zeros_like(local_memory) for _ in range(world)]
                dist.all_gather(memory_by_rank, local_memory)
                memory_by_rank = torch.stack(memory_by_rank)
            else:
                memory_by_rank = local_memory[None]
        if rank == 0 and log_now:
            train_log = {
                **{f"train/{name}": float(reduced[index])
                   for index, name in enumerate(LOSS_NAMES)},
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/cuda_memory_allocated_max_gib": float(memory_by_rank[:, 0].max()),
                "train/cuda_peak_allocated_max_gib": float(memory_by_rank[:, 1].max()),
                "train/cuda_peak_reserved_max_gib": float(memory_by_rank[:, 2].max()),
            }
            memory_text = " ".join(
                f"rank{index}_peak_reserved_gib={float(values[2]):.2f}"
                for index, values in enumerate(memory_by_rank)
            )
            print(
                f"step={step} loss={train_log['train/total_loss']:.6f} "
                f"flow={train_log['train/flow_loss']:.6f} "
                f"entity_l1={train_log['train/entity_pixel_l1']:.6f} "
                f"health_l1={train_log['train/health_pixel_l1']:.6f} "
                f"peak_reserved_max_gib={train_log['train/cuda_peak_reserved_max_gib']:.2f} "
                f"{memory_text}",
                flush=True,
            )
            if run:
                run.log(train_log, step=step)
        if val_loader is not None and (step % args.validate_every == 0 or step == args.steps):
            raw_model.eval()
            values = []
            with torch.no_grad():
                for number, val in enumerate(val_loader):
                    if number >= args.val_batches:
                        break
                    rgb = val["rgb"].to(device)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"
                    ):
                        latent = codec.encode(rgb)
                        condition = {k: v.to(device) for k, v in val["conditions"].items()}
                        val_losses = renderer_training_losses(
                            raw_model,
                            codec,
                            latent,
                            condition,
                            rgb,
                            val["pixel_region_mask"].to(device),
                            region_weight=val["region_weight"].to(device),
                            generator=torch.Generator(device=device).manual_seed(
                                args.seed + number
                            ),
                            **loss_kwargs,
                        )
                        values.append(torch.stack([val_losses[name] for name in LOSS_NAMES]))
            metric = (
                torch.stack(values).mean(0)
                if values
                else torch.full((len(LOSS_NAMES),), float("nan"), device=device)
            )
            if world > 1:
                dist.all_reduce(metric)
                metric /= world
            if rank == 0:
                val_metrics = {
                    name: float(metric[index]) for index, name in enumerate(LOSS_NAMES)
                }
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, **val_metrics}) + "\n")
                if run:
                    run.log({f"val/{name}": value for name, value in val_metrics.items()}, step=step)
            raw_model.train()
        visualize = bool(
            args.val_window_index
            and args.visualize_every
            and (step % args.visualize_every == 0 or step == args.steps)
        )
        if visualize:
            # Only rank zero renders; the barrier prevents other ranks from
            # entering the next DDP backward while it owns the model caches.
            if world > 1:
                dist.barrier()
            if rank == 0:
                raw_model.eval()
                visual_log = {}
                visual_dir = output / "visualizations" / f"step_{step:07d}"
                visual_dir.mkdir(parents=True, exist_ok=True)
                for probe_number, probe in enumerate(probes):
                    try:
                        probe_sample = TextAgentRendererDataset.read_window(
                            probe.episode,
                            val_dataset.vocabulary,
                            start=probe.start,
                            target=probe.target,
                            context_frames=args.context_frames,
                        )
                        sample = collate_renderer([probe_sample])
                        video_path = visual_dir / f"{probe.name}.mp4"
                        metrics = render_probe(
                            raw_model,
                            codec,
                            sample,
                            video_path,
                            seed=args.seed + probe_number,
                            denoising_steps=args.visualization_denoising_steps,
                            precision=args.precision,
                        )
                        visual_log.update(
                            {f"visual/{probe.name}_{key}": value for key, value in metrics.items()}
                        )
                        if run:
                            import wandb

                            visual_log[f"visual/{probe.name}"] = wandb.Video(
                                str(video_path),
                                fps=8,
                                format="mp4",
                                caption=(
                                    f"{probe.scenario_id}, target agent{probe.target}, "
                                    f"event={probe.event or 'movement'}"
                                ),
                            )
                    except Exception as error:
                        raw_model.clear_cache()
                        visual_log[f"visual/{probe.name}_failure"] = 1
                        print(
                            f"warning: visualization probe {probe.name} failed: {error}", flush=True
                        )
                try:
                    (visual_dir / "metrics.json").write_text(
                        json.dumps(visual_log, indent=2, default=lambda value: "wandb.Video")
                    )
                    if run:
                        run.log(visual_log, step=step)
                except Exception as error:
                    print(f"warning: visualization logging failed: {error}", flush=True)
                raw_model.train()
            if world > 1:
                dist.barrier()
        save_now = step % args.save_every == 0 or step == args.steps
        if save_now and world > 1:
            dist.barrier()
        checkpoint_error = None
        if rank == 0 and save_now:
            destination = output / f"step_{step:07d}.pt"
            try:
                staged_torch_save(
                    {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": run_config,
                    },
                    destination,
                    staging_dir=args.checkpoint_staging_dir,
                )
                if run:
                    run.log({"checkpoint/saved_step": step}, step=step)
            except Exception as error:
                try:
                    record_checkpoint_failure(output, destination, error)
                except Exception as record_error:
                    print(
                        f"warning: could not write checkpoint failure record: {record_error}",
                        flush=True,
                    )
                print(f"warning: checkpoint save failed at step {step}: {error}", flush=True)
                if run:
                    run.log(
                        {
                            "checkpoint/save_failure": 1,
                            "checkpoint/save_failure_message": str(error),
                        },
                        step=step,
                    )
                if args.checkpoint_errors == "raise":
                    checkpoint_error = str(error)
        if save_now and world > 1:
            status = [checkpoint_error]
            dist.broadcast_object_list(status, src=0)
            checkpoint_error = status[0]
        if checkpoint_error is not None:
            raise RuntimeError(f"checkpoint save failed: {checkpoint_error}")
    if world > 1:
        dist.barrier()
    if rank == 0 and run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
