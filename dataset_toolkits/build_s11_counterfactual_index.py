#!/usr/bin/env python3
"""Build a small, explicit M3 window index for accepted S11 variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--context-frames", type=int, default=65)
    parser.add_argument("--start", type=int, action="append", required=True)
    parser.add_argument("--target", type=int, action="append", required=True)
    args = parser.parse_args()

    episodes = []
    windows = []
    item_vocabulary = None
    for manifest_path in sorted(args.root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("scenario_id") != "S11" or manifest.get("split") != args.split:
            continue
        validation_path = manifest_path.with_name("validation.json")
        if not validation_path.exists() or not json.loads(validation_path.read_text()).get("usable"):
            continue
        episode_path = manifest_path.parent
        metadata = json.loads((episode_path / "training_metadata.json").read_text())
        current_items = metadata["item_vocabulary"]
        if item_vocabulary is not None and current_items != item_vocabulary:
            raise ValueError("item vocabularies differ across S11 episodes")
        item_vocabulary = current_items
        with np.load(episode_path / manifest.get("training_data_file", "data.npz"),
                     allow_pickle=False) as data:
            frames = len(data["cam_pos"])
        episode_id = len(episodes)
        episodes.append({
            "path": str(episode_path.resolve().relative_to(args.root.resolve())),
            "manifest": manifest,
        })
        for start in args.start:
            if start < 0 or start + args.context_frames > frames:
                raise ValueError(f"window start {start} is invalid for {episode_path}")
            for target in args.target:
                if not 0 <= target < int(manifest["num_agents"]):
                    raise ValueError(f"target {target} is invalid for {episode_path}")
                windows.append((episode_id, int(start), int(target)))
    if not episodes:
        raise ValueError("no accepted S11 episodes found")
    payload = {
        "schema_version": "plot-m3-window-index-v2",
        "split": args.split,
        "context_frames": args.context_frames,
        "stride": None,
        "path_mode": "relative_to_dataset_root",
        "item_vocabulary": item_vocabulary,
        "episodes": episodes,
        "windows": windows,
        "selection": {"starts": args.start, "targets": args.target},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output),
        "episodes": len(episodes),
        "windows": len(windows),
    }, indent=2))


if __name__ == "__main__":
    main()
