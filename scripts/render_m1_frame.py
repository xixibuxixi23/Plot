#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data import TextAgentFillDataset
from plot.models import FillNetworkArgs
from plot.visualization import render_voxel_cameras


def main():
    parser = argparse.ArgumentParser(description="Render one M1 voxel tile with camera frustums")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--target-agent", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dataset = TextAgentFillDataset(
        args.episode, args.vocabulary, split=None, samples_per_agent=1,
        num_views=2, max_agents=2, initial_only=True,
    )
    sample = dataset[args.target_agent]
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config = checkpoint.get("model_args", {})
        model = FillNetworkArgs(
            num_block_classes=dataset.vocabulary.size,
            base_channels=int(config.get("base_channels", 32)), max_views=2,
        ).build().to(args.device)
        model.load_state_dict(checkpoint["model"])
        batch = {k: v.unsqueeze(0).to(args.device) for k, v in sample.items()}
        model.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(
                batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
                batch["images"], batch["agent_mask"], return_aux=True,
            )
        voxels = output["voxel_logits"].argmax(1)[0].cpu().numpy()
        positions = output["camera_position"][0].float().cpu().numpy()
        directions = output["camera_direction"][0].float().cpu().numpy()
        title = "M1 prediction"
    else:
        voxels = sample["target"].numpy()
        positions = sample["camera_position"].numpy()
        directions = sample["camera_direction"].numpy()
        title = "Ground truth initialization"
    try:
        air_class = dataset.vocabulary.class_to_raw.index(126)
    except ValueError as error:
        raise ValueError("vocabulary does not contain Minetest CONTENT_AIR=126") from error
    render_voxel_cameras(
        voxels, positions, directions, args.output, title=title, air_class=air_class
    )
    print(args.output)


if __name__ == "__main__":
    main()
