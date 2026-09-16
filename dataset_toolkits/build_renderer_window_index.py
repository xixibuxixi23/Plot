#!/usr/bin/env python3
"""Precompute valid M3 windows without caching prohibitively large RGB latents."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from plot.data.fill_dataset import TextAgentFillDataset


def scan(job):
    episode_id, row, context, stride, max_agents, release = job
    # Release ledgers retain provenance paths from the collection machine.
    # Resolve the canonical extracted layout first so regeneration is portable.
    path = Path(release) / row["split"] / row["episode_id"]
    if not path.is_dir():
        source = Path(row["source_path"])
        path = source if source.is_absolute() else Path(release) / source
    manifest = json.loads((path / "manifest.json").read_text())
    agents = int(manifest["num_agents"])
    if not 1 <= agents <= max_agents:
        raise ValueError(f"unsupported resident count in {path}")
    metadata = json.loads((path / "training_metadata.json").read_text())
    with np.load(path / manifest.get("training_data_file", "data.npz"), allow_pickle=False) as data:
        frames = len(data["cam_pos"])
        if len(data["action_continuous"]) != frames - 1:
            raise ValueError(f"bad action alignment in {path}")
        start = TextAgentFillDataset._model_start(data, manifest)
        health = data["player_health"]
        valid = data["player_health_valid"].astype(bool) & np.isfinite(health)
        terminations = np.asarray(data["termination_flag"], bool)
        if terminations.ndim > 1:
            terminations = terminations.any(axis=tuple(range(1, terminations.ndim)))
    windows = []
    for begin in range(start, frames - context + 1, stride):
        end = begin + context
        if terminations[begin:end - 1].any() or not valid[begin:end].all():
            continue
        for target in range(agents):
            if not (health[begin:end, target] <= 0).any():
                windows.append((episode_id, begin, target))
    resolved_path = path.resolve()
    try:
        stored_path = resolved_path.relative_to(Path(release).resolve())
    except ValueError:
        # Snapshot ledgers may intentionally point at completed episodes in a
        # still-growing collection. Keep those paths absolute so the cached
        # index remains a stable, read-only view without copying video data.
        stored_path = resolved_path
    return {"path": str(stored_path), "manifest": manifest,
            "items": metadata["item_vocabulary"]}, windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--context-frames", type=int, default=65)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-agents", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.release / f"{args.split}.jsonl").open()]
    episodes, windows, items = [], [], None
    jobs = ((i, row, args.context_frames, args.stride, args.max_agents, args.release)
            for i, row in enumerate(rows))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        for number, (episode, found) in enumerate(executor.map(scan, jobs, chunksize=8), 1):
            current = episode.pop("items")
            if items is not None and current != items:
                raise ValueError("item vocabularies differ")
            items = current
            episodes.append(episode)
            windows.extend(found)
            if number % 1000 == 0:
                print(json.dumps({"episodes": number, "windows": len(windows)}), flush=True)
    payload = {"schema_version": "plot-m3-window-index-v2", "split": args.split,
               "context_frames": args.context_frames, "stride": args.stride,
               "path_mode": "relative_to_dataset_root",
               "item_vocabulary": items, "episodes": episodes, "windows": windows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(payload, temporary); temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "episodes": len(episodes),
                      "windows": len(windows)}, indent=2))


if __name__ == "__main__":
    main()
