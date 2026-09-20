"""Bounded single-clip full-network fitting with independent player flow loss.

The only cached neural output is the frozen VAE target. Trainable voxel,
reference, ROI and DiT computations are repeated with gradients every step.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, fields
import json
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec
from plot.training.renderer_monitoring import render_probe
from plot.training.renderer_optimizer import build_full_player_optimizer
from plot.training.renderer_trainer import renderer_training_losses
from train_scripts.train_renderer import load_weights


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def append_json(path, value):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def scalar_diagnostics(diagnostics):
    """Keep aggregate statistics; framewise tensors are for internal binning."""
    return {key: float(value.detach()) for key, value in diagnostics.items()
            if value.numel() == 1}


def rgb_image(frame):
    return cv2.cvtColor((frame.detach().float().clamp(0, 1).permute(1, 2, 0)
                        .cpu().numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def label(frame, text):
    frame = cv2.copyMakeBorder(frame, 26, 0, 0, 0, cv2.BORDER_CONSTANT)
    cv2.putText(frame, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    return frame


def player_bounds(mask):
    ys, xs = np.where(mask.detach().cpu().numpy() > 0)
    if not len(xs):
        raise ValueError("selected clip has no visible player")
    return (max(0, int(xs.min()) - 12), max(0, int(ys.min()) - 12),
            min(mask.shape[-1], int(xs.max()) + 13), min(mask.shape[-2], int(ys.max()) + 13))


def audit_vae(rgb, reconstruction, mask, output):
    err = (rgb[:, 1:].float() - reconstruction[:, 1:].float()).abs()
    roi = mask[:, 1:].float()
    metrics = {"global_l1": float(err.mean()),
               "player_l1": float((err * roi).sum() / (3 * roi.sum().clamp_min(1))),
               "player_pixels_per_frame": mask[0].flatten(1).sum(1).cpu().tolist(),
               "note": "GT encode/decode reference, not a strict lower bound on model error"}
    tiles = []
    for frame in range(rgb.shape[1]):
        raw, rec = rgb_image(rgb[0, frame]), rgb_image(reconstruction[0, frame])
        overlay = raw.copy()
        active = mask[0, frame, 0].cpu().numpy().astype(bool)
        overlay[active] = (.6 * overlay[active] + .4 * np.array([0, 255, 255])).astype(np.uint8)
        tiles.append(np.concatenate([label(raw, f"GT frame {frame}"),
                                     label(rec, "Frozen VAE reconstruction"),
                                     label(overlay, "GT player loss mask")], axis=1))
    cv2.imwrite(str(output / "vae_all_frames.jpg"), np.concatenate(tiles, axis=0))
    x0, y0, x1, y1 = player_bounds(mask[0, -1, 0])
    crop = np.concatenate([label(rgb_image(rgb[0, -1])[y0:y1, x0:x1], "GT player"),
                           label(rgb_image(reconstruction[0, -1])[y0:y1, x0:x1], "VAE player")], 1)
    cv2.imwrite(str(output / "vae_player_crop.png"), crop)
    write_json(output / "vae_metrics.json", metrics)
    print(json.dumps({"vae_audit": metrics}), flush=True)
    return metrics


def save_rollout_still(video_path, player_mask):
    capture = cv2.VideoCapture(str(video_path))
    frame = None
    while True:
        ok, candidate = capture.read()
        if not ok:
            break
        frame = candidate
    capture.release()
    if frame is None:
        raise RuntimeError(f"cannot read evaluation video {video_path}")
    cv2.imwrite(str(video_path.with_suffix(".png")), frame)
    x0, y0, x1, y1 = player_bounds(player_mask[0, -1, 0])
    width = frame.shape[1] // 2
    crop = np.concatenate([label(frame[y0:y1, x0:x1], "GT player"),
                           label(frame[y0:y1, width+x0:width+x1], "Generated player")], 1)
    cv2.imwrite(str(video_path.with_name(video_path.stem + "_player.png")), crop)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=Path("derived/m3_player_short_20260919/selection.json"))
    parser.add_argument("--selection-index", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True, help="Shared visualizations and metrics")
    parser.add_argument("--run-dir", type=Path, help="Fresh machine-local checkpoint/W&B directory")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1000, help="Additional optimizer steps")
    parser.add_argument("--evaluate-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--base-lr", type=float, default=1e-5)
    parser.add_argument("--appearance-lr", type=float, default=5e-5)
    parser.add_argument("--player-flow-weight", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if min(args.steps, args.evaluate_every, args.save_every) < 1:
        parser.error("step counts must be positive")
    if not args.audit_only and (args.run_dir is None or args.run_dir.exists()):
        parser.error("training requires a fresh --run-dir; existing runs are never overwritten")
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    config = json.loads((args.warm_start.parent / "config.json").read_text())
    parent_step = int(args.warm_start.stem.removeprefix("step_"))
    row = json.loads(args.selection.read_text())["train"][args.selection_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = collate_renderer([TextAgentRendererDataset.read_window(
        row["path"], config["training"]["vocabulary"], start=row["start"],
        target=row["target"], context_frames=9)])
    rgb = sample["rgb"].cuda()
    mask = sample["player_region_mask"].cuda()
    if (mask[:, 1:].flatten(2).sum(-1) <= 0).any():
        raise ValueError("single-clip diagnostic requires a visible player in every future frame")
    codec = RendererCodec.from_run_config(load_weights(config["training"]["pixel_vae"]), config).cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean = codec.encode(rgb).detach()
        reconstruction = codec.decode(clean)
    audit = audit_vae(rgb, reconstruction, mask, args.output_dir)
    del reconstruction
    write_json(args.output_dir / "selection.json", row)
    if args.audit_only:
        return
    args.run_dir.mkdir(parents=True)
    names = {field.name for field in fields(RendererArgs)}
    cfg = {key: value for key, value in config["renderer"].items() if key in names}
    cfg["context_frames"] = 9
    model = Renderer(RendererArgs(**cfg)).cuda()
    model.load_state_dict(load_weights(args.warm_start), strict=True)
    model.requires_grad_(True).train()
    optimizer = build_full_player_optimizer(model, base_lr=args.base_lr, appearance_lr=args.appearance_lr)
    conditions = {key: value.cuda() for key, value in sample["conditions"].items()}
    pixel_mask = sample["pixel_region_mask"].cuda()
    experiment = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    experiment.update(parent_step=parent_step, final_step=parent_step + args.steps,
                      trainable_parameters=sum(p.numel() for p in model.parameters()),
                      protocol="fixed 9-frame train clip; clean prefix; eight pure-noise future frames; 20 Euler steps; no future GT at inference",
                      selection=row, vae_audit=audit)
    config["renderer"] = asdict(model.cfg)
    config["experiment"] = experiment
    config["training"].update(warm_start=str(args.warm_start), output_dir=str(args.run_dir),
        steps=parent_step + args.steps, batch_size=1, gradient_accumulation=1,
        context_frames=9, lr=args.base_lr, appearance_lr=args.appearance_lr,
        flow_loss_weight=1., player_flow_loss_weight=args.player_flow_weight,
        latent_player_region_upweight=0., latent_entity_region_upweight=0.,
        player_pixel_l1_weight=.1, player_pixel_edge_weight=.025,
        entity_pixel_l1_weight=0., entity_pixel_edge_weight=0., health_pixel_l1_weight=0.,
        pixel_loss_frames=2, pixel_frame_selection="player",
        window_index=None, val_window_index=None, optimizer_reset=True)
    write_json(args.run_dir / "config.json", config)
    write_json(args.output_dir / "config.json", config)
    import wandb
    run = wandb.init(project="plot-m3", entity="ckx23-tsinghua-university",
                     name="m3-single-clip-independent-player-flow-from5500",
                     config=experiment, mode=args.wandb_mode, dir=str(args.run_dir))
    write_json(args.run_dir / "wandb_run.json", {"id": run.id, "url": run.url})
    write_json(args.output_dir / "wandb_run.json", {"id": run.id, "url": run.url})
    print(json.dumps({"experiment": experiment, "wandb_url": run.url}), flush=True)

    def evaluate(local_step, tag=None, evaluation_sample=None, seed=None):
        name = tag or f"step_{local_step:04d}"
        video = args.output_dir / f"{name}.mp4"
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            metrics = render_probe(model, codec, evaluation_sample or sample, video,
                                   seed=args.seed if seed is None else seed,
                                   denoising_steps=20, precision="bf16")
        save_rollout_still(video, mask)
        append_json(args.output_dir / "evaluations.jsonl", {"local_step": local_step,
            "global_step": parent_step + local_step, "tag": name, **metrics})
        print(json.dumps({"evaluation": name, **metrics}), flush=True)
        # Swapped-reference outputs have no matching GT: log as sensitivity,
        # never label their GT distance a quality measurement.
        if tag is None:
            run.log({"local_step": local_step, **{f"pure_noise/{k}": v for k, v in metrics.items()},
                     "pure_noise/video": wandb.Video(str(video), format="mp4"),
                     "pure_noise/player_crop": wandb.Image(str(video.with_name(video.stem + "_player.png")))},
                    step=parent_step + local_step)
        model.train()

    evaluate(0)
    totals = defaultdict(float)
    block_started = time.perf_counter()
    for local_step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        diagnostics = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses = renderer_training_losses(model, codec, clean, conditions, rgb, pixel_mask,
                player_region_mask=mask, region_weight=None, flow_loss_weight=1.,
                player_flow_loss_weight=args.player_flow_weight, flow_diagnostics=diagnostics,
                frames_per_sample=2, pixel_frame_selection="player",
                player_pixel_l1_weight=.1, player_pixel_edge_weight=.025,
                entity_pixel_l1_weight=0., entity_pixel_edge_weight=0., health_pixel_l1_weight=0.)
        if not torch.isfinite(losses["total_loss"]):
            raise FloatingPointError(f"non-finite loss at local step {local_step}")
        losses["total_loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        # Monitor actual parameter-group gradients, not just requires_grad flags.
        norms = {f"grad/{g['name']}": float(torch.stack([p.grad.float().square().sum()
                 for p in g["params"] if p.grad is not None]).sum().sqrt()) for g in optimizer.param_groups}
        optimizer.step()
        log = {"local_step": local_step, "global_step": parent_step + local_step,
               **{key: float(value.detach()) for key, value in losses.items()},
               **scalar_diagnostics(diagnostics),
               "grad_norm_before_clip": float(grad_norm), **norms}
        append_json(args.output_dir / "training.jsonl", log)
        for key, value in log.items():
            if key not in ("local_step", "global_step"):
                totals[key] += value
        if local_step % 10 == 0:
            averaged = {key: value / 10 for key, value in totals.items()
                        if not key.endswith(("_sum", "_count"))}
            for level in ("low", "mid", "high"):
                count = totals[f"player_flow_{level}_count"]
                if count:
                    averaged[f"player_flow_{level}"] = totals[f"player_flow_{level}_sum"] / count
                averaged[f"player_flow_{level}_frames"] = count
            averaged["seconds_per_step"] = (time.perf_counter() - block_started) / 10
            averaged["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            print(json.dumps({"local_step": local_step, **averaged}), flush=True)
            run.log({"local_step": local_step, **{f"train/{k}": v for k, v in averaged.items()}},
                    step=parent_step + local_step)
            totals.clear()
            block_started = time.perf_counter()
        if local_step % args.save_every == 0 or local_step == args.steps:
            target = args.run_dir / f"step_{parent_step + local_step:07d}.pt"
            temporary = target.with_suffix(".tmp")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": parent_step + local_step, "config": config,
                        "cpu_rng_state": torch.get_rng_state(),
                        "cuda_rng_state": torch.cuda.get_rng_state()}, temporary)
            temporary.replace(target)
        if local_step % args.evaluate_every == 0 or local_step == args.steps:
            evaluate(local_step)
            block_started = time.perf_counter()
    evaluate(args.steps, tag="final_new_noise", seed=args.seed + 1)
    swapped = dict(sample)
    swapped["conditions"] = dict(sample["conditions"])
    swapped["conditions"]["player_reference"] = sample["conditions"]["player_reference"].roll(1, dims=1)
    evaluate(args.steps, tag="final_swapped_references_sensitivity_only", evaluation_sample=swapped)
    write_json(args.output_dir / "completed.json", {"local_steps": args.steps, "global_step": parent_step + args.steps})
    run.finish()


if __name__ == "__main__":
    main()
