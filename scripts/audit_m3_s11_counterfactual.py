#!/usr/bin/env python3
"""Causally audit whether S11 player references control rendered appearance."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from plot.data.fill_dataset import TextAgentFillDataset
from plot.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec
from plot.training.renderer_monitoring import write_comparison_video
from scripts.audit_m3_reference_condition import _masked_mean, _predict, _weights


def _episodes(root: Path, group_id: str | None = None) -> list[Path]:
    rows = []
    for manifest_path in root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("scenario_id") == "S11"
                and (group_id is None or manifest.get("appearance_group_id") == group_id)):
            rows.append((
                str(manifest["appearance_group_id"]),
                int(manifest["appearance_variant_index"]),
                manifest_path.parent,
            ))
    return [row[2] for row in sorted(rows)]


def _save_contact(path: Path, truth: torch.Tensor, correct: torch.Tensor,
                  shuffled: torch.Tensor, mask: torch.Tensor) -> int:
    coverage = mask.flatten(1).sum(1)
    frame = int(coverage.argmax())
    panels = []
    for label, value in (("truth", truth), ("correct ref", correct),
                         ("shuffled ref", shuffled)):
        image = (value[frame].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.putText(image, label, (18, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(image)
    cv2.imwrite(str(path), cv2.hconcat(panels))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=83)
    parser.add_argument(
        "--start-offset",
        type=int,
        help="window start relative to each episode's model_start_observation",
    )
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--group-id", help="appearance_group_id in a multi-trajectory root")
    parser.add_argument("--denoising-steps", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-player-delta", type=float, default=0.005)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    config = json.loads((checkpoint_path.parent / "config.json").read_text())
    renderer_keys = {field.name for field in fields(RendererArgs)}
    renderer_config = {
        key: value for key, value in config["renderer"].items() if key in renderer_keys
    }
    device = torch.device(args.device)
    model = Renderer(RendererArgs(**renderer_config)).to(device).eval()
    model.load_state_dict(_weights(checkpoint_path), strict=True)
    codec = RendererCodec(_weights(config["training"]["pixel_vae"])).to(device).eval()
    episodes = _episodes(Path(args.dataset_root), args.group_id)
    if len(episodes) < 2:
        raise ValueError("S11 causal audit needs at least two appearance variants")
    found_groups = {
        json.loads((episode / "manifest.json").read_text())["appearance_group_id"]
        for episode in episodes
    }
    if len(found_groups) != 1:
        raise ValueError("multi-trajectory roots require --group-id")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    starts = []
    for episode in episodes:
        if args.start_offset is None:
            starts.append(args.start)
            continue
        manifest = json.loads((episode / "manifest.json").read_text())
        with np.load(
            episode / manifest.get("training_data_file", "data.npz"),
            allow_pickle=False,
        ) as data:
            starts.append(
                TextAgentFillDataset._model_start(data, manifest) + args.start_offset
            )
    raws = [
        TextAgentRendererDataset.read_window(
            episode,
            config["training"]["vocabulary"],
            start=start,
            target=args.target,
            context_frames=65,
        )
        for episode, start in zip(episodes, starts)
    ]
    # Match the training adapter: fresh Minetest processes may assign shifted
    # numeric IDs to the same named nodes, which is not a causal appearance cue.
    for raw in raws[1:]:
        raw["conditions"]["voxel_classes"] = raws[0]["conditions"]["voxel_classes"]
        raw["conditions"]["voxel_known"] = raws[0]["conditions"]["voxel_known"]

    rows = []
    for variant, raw in enumerate(raws):
        predictions = {}
        truth = None
        for name in ("correct", "shuffled"):
            sample = collate_renderer([raw])
            if name == "shuffled":
                # Same four skins, wrong actor assignment: this isolates binding
                # rather than merely changing the global style distribution.
                sample["conditions"]["player_reference"] = sample["conditions"][
                    "player_reference"
                ].roll(1, dims=1)
            truth, prediction = _predict(
                model,
                codec,
                sample,
                seed=1000 + variant,
                precision=config["training"].get("precision", "bf16"),
                mask_prefix_players=True,
                denoising_steps=args.denoising_steps,
                horizon=args.horizon,
            )
            predictions[name] = prediction.float()
            write_comparison_video(
                output / f"variant{variant}_{name}.mp4",
                truth,
                prediction,
                sample["region_weight"][0, 1:1 + args.horizon],
            )
        assert truth is not None
        player_mask = raw["player_region_mask"][1:1 + args.horizon].to(device).float()
        outside = 1 - player_mask
        correct_error = (predictions["correct"] - truth.float()).abs()
        shuffled_error = (predictions["shuffled"] - truth.float()).abs()
        delta = (predictions["shuffled"] - predictions["correct"]).abs()
        row = {
            "variant": variant,
            "episode": str(episodes[variant]),
            "player_pixels": float(player_mask.sum()),
            "correct_player_l1": _masked_mean(correct_error, player_mask),
            "shuffled_player_l1": _masked_mean(shuffled_error, player_mask),
            "shuffled_minus_correct_player_l1": _masked_mean(
                shuffled_error - correct_error, player_mask
            ),
            "conditioning_delta_player": _masked_mean(delta, player_mask),
            "conditioning_delta_outside": _masked_mean(delta, outside),
        }
        row["contact_frame"] = _save_contact(
            output / f"variant{variant}_contact.jpg",
            truth, predictions["correct"], predictions["shuffled"], player_mask[:, 0],
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "schema_version": "plot-m3-s11-reference-causal-audit-v1",
        "checkpoint": str(checkpoint_path),
        "start": args.start if args.start_offset is None else None,
        "start_offset": args.start_offset,
        "episode_starts": starts,
        "target": args.target,
        "appearance_group_id": args.group_id,
        "denoising_steps": args.denoising_steps,
        "horizon": args.horizon,
        "rows": rows,
        "all_variants_reference_sensitive": all(
            row["conditioning_delta_player"] >= args.minimum_player_delta for row in rows
        ),
        "all_variants_correct_reference_better": all(
            row["shuffled_minus_correct_player_l1"] > 0 for row in rows
        ),
    }
    report["passed"] = bool(
        report["all_variants_reference_sensitive"]
        and report["all_variants_correct_reference_better"]
    )
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "passed", "all_variants_reference_sensitive",
        "all_variants_correct_reference_better",
    )}, indent=2))


if __name__ == "__main__":
    main()
