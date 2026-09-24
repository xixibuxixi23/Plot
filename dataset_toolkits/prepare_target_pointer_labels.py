"""Build lightweight next-placement labels for an existing M4 state cache.

The expensive 48^3 cache is reused unchanged.  Each sidecar row is either an
exact 13^3 voxel address, 2197 for no future placement, -1 for a future target
outside the local cube, or -2 while a distributed shard has not touched it.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


VOXEL_COUNT = 13 ** 3


def event_position(event):
    position = event.get("position")
    if not isinstance(position, dict) or not all(key in position for key in ("x", "y", "z")):
        return None
    return np.asarray([position["x"], position["z"], position["y"]], np.int64)


def assigned_shard(episode_id, shards):
    digest = hashlib.sha1(episode_id.encode()).digest()
    return int.from_bytes(digest[:8], "little") % shards


def build(index, output, shard, shards):
    index = Path(index)
    summary = json.loads((index.parent / "summary.json").read_text())
    dataset_root = Path(summary["dataset_root"])
    split = index.stem
    rows = sum(1 for _ in index.open())
    labels = np.lib.format.open_memmap(output, mode="w+", dtype=np.int32, shape=(rows,))
    labels[:] = -2

    @lru_cache(maxsize=8)
    def load_episode(episode_id):
        episode = dataset_root / split / episode_id
        manifest = json.loads((episode / "manifest.json").read_text())
        data_path = episode / manifest.get("training_data_file", "data.npz")
        with np.load(data_path, allow_pickle=False) as data:
            positions = data["player_pos"].astype(np.float32)
        events = {}
        event_path = episode / manifest.get("event_file", "events.jsonl")
        for line in event_path.open():
            event = json.loads(line)
            if event.get("initialization_event") or event.get("event") not in (
                    "block_placed", "scaffold_placed"):
                continue
            actor = event.get("actor") or event.get("source")
            coordinate = event_position(event)
            transition = event.get("transition_index")
            if actor is None or coordinate is None or transition is None:
                continue
            events.setdefault(actor, []).append((int(transition), coordinate))
        for actor in events:
            events[actor].sort(key=lambda row: row[0])
        return positions, events

    touched = voxel = null = outside = 0
    for row_index, line in enumerate(index.open()):
        row = json.loads(line)
        episode_id = row["episode_id"]
        if assigned_shard(episode_id, shards) != shard:
            continue
        positions, events = load_episode(episode_id)
        anchor_frame = int(row["anchor"])
        agent = int(row["agent_slot"])
        future = events.get(f"agent{agent}", [])
        offset = bisect_left([item[0] for item in future], anchor_frame)
        if offset == len(future):
            label = VOXEL_COUNT
            null += 1
        else:
            coordinate = future[offset][1]
            anchor = np.rint(positions[anchor_frame, agent]).astype(np.int64)
            local = coordinate - (anchor - 6)
            if (local < 0).any() or (local >= 13).any():
                label = -1
                outside += 1
            else:
                label = int(np.ravel_multi_index(tuple(local), (13, 13, 13)))
                voxel += 1
        labels[row_index] = label
        touched += 1
        if touched % 100000 == 0:
            print(json.dumps(dict(touched=touched, voxel=voxel, null=null, outside=outside)), flush=True)
    labels.flush()
    print(json.dumps(dict(output=str(output), rows=rows, touched=touched, voxel=voxel,
                          null=null, outside=outside, shard=shard, shards=shards)), flush=True)


def merge(parts, output):
    arrays = [np.load(path, mmap_mode="r", allow_pickle=False) for path in parts]
    if not arrays or any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("target shard shapes differ")
    merged = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.int32, shape=arrays[0].shape)
    touched = np.zeros(arrays[0].shape, dtype=np.uint8)
    merged[:] = -2
    for array in arrays:
        selected = array != -2
        if (touched[selected] != 0).any():
            raise ValueError("target shards overlap")
        merged[selected] = array[selected]
        touched[selected] += 1
    if (touched != 1).any():
        raise ValueError(f"target shards incomplete: {int((touched != 1).sum())} rows")
    merged.flush()
    unique, counts = np.unique(merged, return_counts=True)
    print(json.dumps(dict(output=str(output), rows=len(merged), invalid=int((merged == -1).sum()),
                          null=int((merged == VOXEL_COUNT).sum()),
                          voxel=int(((merged >= 0) & (merged < VOXEL_COUNT)).sum()),
                          values=len(unique))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--index", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--shard", type=int, required=True)
    build_parser.add_argument("--shards", type=int, required=True)
    merge_parser = sub.add_parser("merge")
    merge_parser.add_argument("--parts", type=Path, nargs="+", required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if not 0 <= args.shard < args.shards:
            raise ValueError("invalid shard")
        build(args.index, args.output, args.shard, args.shards)
    else:
        merge(args.parts, args.output)


if __name__ == "__main__":
    main()
