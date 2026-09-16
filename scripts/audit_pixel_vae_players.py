#!/usr/bin/env python3
"""Measure the frozen Pixel VAE's reconstruction ceiling on visible players.

This audit deliberately bypasses M3.  It encodes and decodes ground-truth RGB
frames, then reports full-frame and player-mask errors together with the frozen
domain identity encoder's retrieval accuracy.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file

from plot.data.renderer_dataset import TextAgentRendererDataset
from plot.models.player_identity import PlayerIdentityEncoder, crop_masked_players
from plot.models.renderer_codec import RendererCodec


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--pixel-vae", required=True)
    parser.add_argument("--identity-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser.parse_args()


def masked_mean(value, mask):
    expanded = mask.expand_as(value)
    return value[expanded].float().mean().item() if expanded.any() else float("nan")


def summarize(rows, key):
    values = np.asarray([row[key] for row in rows if math.isfinite(row[key])], np.float64)
    if not len(values):
        return None
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def area_bin(area):
    # At 640x360 the largest visible-player frame in a 65-frame window is
    # commonly above 1k pixels, so use bins that retain useful separation.
    if area < 2_000:
        return "small_lt2000px"
    if area < 10_000:
        return "medium_2000_9999px"
    return "large_ge10000px"


def distance_bin(distance):
    if distance < 2:
        return "near_lt2m"
    if distance < 5:
        return "mid_2_5m"
    return "far_ge5m"


def save_panel(path, raw, recon, mask):
    raw_np = raw.permute(1, 2, 0).float().cpu().numpy()
    recon_np = recon.permute(1, 2, 0).float().cpu().numpy()
    mask_np = mask[0].cpu().numpy().astype(bool)
    overlay = raw_np.copy()
    overlay[mask_np] = 0.55 * overlay[mask_np] + 0.45 * np.array([1.0, 0.0, 1.0])
    error = np.abs(raw_np - recon_np)
    error = np.clip(error * 4.0, 0, 1)
    panel = np.concatenate((raw_np, recon_np, overlay, error), axis=1)
    cv2.imwrite(str(path), cv2.cvtColor((panel * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_bf16 = args.precision == "bf16"

    dataset = TextAgentRendererDataset(
        args.dataset_root,
        args.vocabulary,
        split="val_id",
        context_frames=65,
        window_index=args.window_index,
        targets_per_window=1,
    )
    codec = RendererCodec(load_file(args.pixel_vae)).to(device).eval()
    identity_payload = torch.load(args.identity_checkpoint, map_location="cpu", weights_only=False)
    identity = PlayerIdentityEncoder(identity_payload["embedding_dim"])
    identity.load_state_dict(identity_payload["model"], strict=True)
    identity = identity.to(device).eval().requires_grad_(False)

    generator = torch.Generator().manual_seed(args.seed)
    candidates = torch.randperm(len(dataset), generator=generator).tolist()
    rows = []
    for dataset_index in candidates:
        sample = dataset[dataset_index]
        if not bool(sample["player_identity_valid"]):
            continue
        frame = int(sample["player_identity_frame"])
        slot = int(sample["player_identity_slot"])
        raw = sample["rgb"][frame : frame + 1].to(device)
        mask = sample["player_identity_mask"].to(device)
        area = int(mask.sum().item())
        if area < 96:
            continue
        conditions = sample["conditions"]
        target = int(conditions["target_agent"])
        camera_world = (
            conditions["player_position"][frame, target]
            + conditions["camera_relative"][frame, target]
        )
        distance = float(torch.linalg.vector_norm(
            conditions["player_position"][frame, slot] - camera_world
        ))

        rgb = raw[None]
        autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16)
        with torch.inference_mode(), autocast:
            latent = codec.encode(rgb, chunk_size=1)
            recon = codec.decode(latent, chunk_size=1)[0, 0]
            raw_frame = raw[0]
            raw_crop, _ = crop_masked_players(raw_frame[None], mask[None], output_size=identity.crop_size)
            recon_crop, _ = crop_masked_players(recon[None], mask[None], output_size=identity.crop_size)
            references = conditions["player_reference"].to(device)
            ref_embeddings = identity.encode_reference(references)
            raw_embedding = identity.encode_crop(raw_crop)
            recon_embedding = identity.encode_crop(recon_crop)
            raw_scores = (raw_embedding @ ref_embeddings.T)[0]
            recon_scores = (recon_embedding @ ref_embeddings.T)[0]

        player_mse = masked_mean((recon - raw_frame).square(), mask)
        row = {
            "dataset_index": int(dataset_index),
            "frame": frame,
            "target_slot": target,
            "visible_slot": slot,
            "player_pixels": area,
            "distance_m": distance,
            "area_bin": area_bin(area),
            "distance_bin": distance_bin(distance),
            "global_l1": float((recon - raw_frame).abs().float().mean().item()),
            "player_l1": masked_mean((recon - raw_frame).abs(), mask),
            "background_l1": masked_mean((recon - raw_frame).abs(), ~mask),
            "player_psnr": float(-10 * math.log10(max(player_mse, 1e-12))),
            "raw_identity_similarity": float(raw_scores[slot].float().item()),
            "recon_identity_similarity": float(recon_scores[slot].float().item()),
            "identity_similarity_drop": float(
                (raw_scores[slot] - recon_scores[slot]).float().item()
            ),
            "raw_identity_top1": int(raw_scores.argmax().item() == slot),
            "recon_identity_top1": int(recon_scores.argmax().item() == slot),
            "raw_recon_identity_cosine": float((raw_embedding * recon_embedding).sum().float().item()),
        }
        rows.append(row)
        if len(rows) <= 12:
            save_panel(output / f"sample_{len(rows):02d}.jpg", raw_frame, recon, mask)
        print(json.dumps(row), flush=True)
        if len(rows) >= args.samples:
            break

    if not rows:
        raise RuntimeError("no valid visible-player samples were found")
    metrics = (
        "global_l1", "player_l1", "background_l1", "player_psnr",
        "raw_identity_similarity", "recon_identity_similarity",
        "identity_similarity_drop",
        "raw_identity_top1", "recon_identity_top1", "raw_recon_identity_cosine",
    )
    grouped = {"all": rows}
    for kind in ("area_bin", "distance_bin"):
        buckets = defaultdict(list)
        for row in rows:
            buckets[row[kind]].append(row)
        grouped.update(buckets)
    report = {
        "protocol": {
            "description": "Ground-truth RGB -> frozen Pixel VAE posterior mean -> RGB; M3 bypassed",
            "dataset_root": args.dataset_root,
            "window_index": args.window_index,
            "pixel_vae": args.pixel_vae,
            "identity_checkpoint": args.identity_checkpoint,
            "seed": args.seed,
            "requested_samples": args.samples,
            "evaluated_samples": len(rows),
            "panel_columns": ["ground_truth", "vae_reconstruction", "ground_truth_mask_overlay", "4x_absolute_error"],
        },
        "groups": {
            name: {metric: summarize(values, metric) for metric in metrics}
            for name, values in grouped.items()
        },
        "samples": rows,
    }
    (output / "pixel_vae_player_audit.json").write_text(json.dumps(report, indent=2) + "\n")

    all_group = report["groups"]["all"]
    lines = [
        "# Frozen Pixel VAE player reconstruction audit",
        "",
        "M3 is bypassed: each ground-truth validation frame is encoded with the frozen",
        "Pixel VAE posterior mean and immediately decoded.",
        "",
        "| Metric | Mean | Median |",
        "|---|---:|---:|",
    ]
    for metric in metrics:
        value = all_group[metric]
        lines.append(f"| {metric} | {value['mean']:.6f} | {value['median']:.6f} |")
    lines.extend((
        "",
        "Montages are ordered as ground truth / VAE reconstruction / mask overlay / 4x absolute error.",
        "Area- and distance-stratified values are in `pixel_vae_player_audit.json`.",
    ))
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {output / 'pixel_vae_player_audit.json'}", flush=True)


if __name__ == "__main__":
    main()
