"""Derive all shorter block-aligned windows contained in a validated M3 index.

No raw episode is rewritten and no unvalidated temporal span is admitted.
Overlapping parent windows are deduplicated per episode and camera.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.checkpoint_io import staged_torch_save


def shorter_windows(windows, source_frames, target_frames=9, block_frames=8):
    if (target_frames < block_frames + 1 or source_frames < target_frames
            or (source_frames - 1) % block_frames or (target_frames - 1) % block_frames):
        raise ValueError("contexts must be 1+8*k, and the target cannot be longer")
    previous_episode = -1
    for episode, rows in groupby(windows, key=lambda row: int(row[0])):
        if episode <= previous_episode:
            raise ValueError("source windows must be ordered by episode")
        previous_episode = episode
        unique = set()
        for _, start, target in rows:
            for offset in range(0, source_frames - target_frames + 1, block_frames):
                unique.add((int(start) + offset, int(target)))
        for start, target in sorted(unique):
            yield episode, start, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=9)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a fresh derived index")
    digest = hashlib.sha256()
    with args.source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    windows = list(shorter_windows(source["windows"], source["context_frames"], args.context_frames))
    used = {episode for episode, _, _ in windows}
    counts = Counter(source["episodes"][episode]["manifest"].get("scenario_id", "unknown")
                     for episode in used)
    provenance = {"path": str(args.source.resolve()), "sha256": digest.hexdigest(),
                  "context_frames": source["context_frames"], "windows": len(source["windows"])}
    summary = {"split": source["split"], "episodes_with_windows": len(used),
               "indexed_episodes": len(source["episodes"]), "windows": len(windows),
               "context_frames": args.context_frames, "episodes_by_scenario": dict(sorted(counts.items())),
               "source": provenance,
               "coverage": "All block-aligned subwindows of every validated parent; not a new raw-data validation"}
    payload = dict(source, context_frames=args.context_frames, windows=windows,
                   created_utc=datetime.now(timezone.utc).isoformat(), summary=summary,
                   derived_from=provenance)
    staged_torch_save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **summary}), flush=True)


if __name__ == "__main__":
    main()
