#!/usr/bin/env python3
"""Build a grouped M3 window index from accepted S11 episode directories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from plot.data.fill_dataset import TextAgentFillDataset


def _integers(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or min(result) < 0:
        raise argparse.ArgumentTypeError("expected comma-separated nonnegative integers")
    return result


def _strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--context-frames", type=int, default=65)
    parser.add_argument(
        "--offsets", type=_integers, default=_integers("0,24,48,72"),
        help="window starts relative to each episode's model_start_observation",
    )
    parser.add_argument("--targets", type=_integers, default=_integers("0,1,2,3"))
    parser.add_argument("--variants-per-group", type=int, default=4)
    parser.add_argument(
        "--exclude-groups", type=_strings, default=[],
        help="comma-separated appearance_group_id values rejected by a paired-data audit",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    grouped: dict[str, list[tuple[int, Path, dict]]] = defaultdict(list)
    for manifest_path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("split") != args.split or manifest.get("scenario_id") != "S11":
            continue
        validation = manifest_path.with_name("validation.json")
        if not validation.exists() or not json.loads(validation.read_text()).get("usable"):
            continue
        group = manifest.get("appearance_group_id")
        variant = manifest.get("appearance_variant_index")
        if group is not None and variant is not None:
            grouped[str(group)].append((int(variant), manifest_path.parent, manifest))

    complete = []
    skipped = {}
    excluded = {}
    excluded_groups = set(args.exclude_groups)
    expected = list(range(args.variants_per_group))
    for group, rows in sorted(grouped.items()):
        rows.sort()
        variants = [row[0] for row in rows]
        if group in excluded_groups:
            excluded[group] = variants
            continue
        if variants == expected:
            complete.extend(rows)
        else:
            skipped[group] = variants
    if not complete:
        raise ValueError("no complete accepted S11 appearance groups found")

    episodes, windows, item_vocabulary = [], [], None
    for _, episode, manifest in complete:
        metadata = json.loads((episode / "training_metadata.json").read_text())
        items = metadata["item_vocabulary"]
        if item_vocabulary is not None and items != item_vocabulary:
            raise ValueError("S11 item vocabularies differ")
        item_vocabulary = items
        with np.load(episode / manifest.get("training_data_file", "data.npz")) as data:
            frames = len(data["cam_pos"])
            model_start = TextAgentFillDataset._model_start(data, manifest)
        episode_id = len(episodes)
        indexed_manifest = dict(manifest)
        indexed_manifest["model_start_observation"] = model_start
        episodes.append({
            "path": str(episode.resolve().relative_to(root)),
            "manifest": indexed_manifest,
        })
        for offset in args.offsets:
            start = model_start + offset
            if start + args.context_frames > frames:
                raise ValueError(
                    f"window offset {offset} at {start} exceeds {frames} frames in {episode}"
                )
            for target in args.targets:
                if target >= int(manifest["num_agents"]):
                    raise ValueError(f"target {target} is absent in {episode}")
                windows.append((episode_id, start, target))

    payload = {
        "schema_version": "plot-m3-window-index-v2",
        "split": args.split,
        "context_frames": args.context_frames,
        "stride": None,
        "path_mode": "relative_to_dataset_root",
        "selection": {
            "model_start_offsets": args.offsets,
            "targets": args.targets,
            "excluded_groups": sorted(excluded_groups),
        },
        "variants_per_group": args.variants_per_group,
        "item_vocabulary": item_vocabulary,
        "episodes": episodes,
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(args.output)
    print(json.dumps({
        "output": str(args.output),
        "complete_groups": len(complete) // args.variants_per_group,
        "episodes": len(episodes),
        "windows": len(windows),
        "skipped_incomplete_groups": skipped,
        "excluded_groups": excluded,
    }, indent=2))


if __name__ == "__main__":
    main()
