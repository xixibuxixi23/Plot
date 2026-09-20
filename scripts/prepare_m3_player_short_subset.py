"""Select visible resident clips from validated indexes for a small c9 experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.data.chunked_npz import open_npz


def select(root, index_path, count_per_kind, seed):
    source = torch.load(index_path, map_location="cpu", weights_only=False)
    by_episode = defaultdict(list)
    for episode, start, target in source["windows"]:
        by_episode[int(episode)].append((int(start), int(target)))
    rng = random.Random(seed)
    candidates = list(by_episode)
    rng.shuffle(candidates)
    episodes, windows, audit = [], [], []
    used = set()
    for kind in ("human_like", "npc_villager", "npc_zombie", "npc_skeleton"):
        found = 0
        for episode_id in candidates:
            row = source["episodes"][episode_id]
            manifest = row["manifest"]
            kinds = manifest["agent_kinds"]
            if episode_id in used or kind not in kinds.values():
                continue
            valid = [(start, target) for start, target in by_episode[episode_id]
                     if kinds[f"agent{target}"] == "human_like"
                     and any(value == kind and slot != f"agent{target}" for slot, value in kinds.items())]
            if not valid:
                continue
            path = root / row["path"]
            # Each short clip is contained in a previously validated 65-frame
            # window (health, termination and action alignment already checked).
            rng.shuffle(valid)
            best = None
            with open_npz(path / manifest.get("training_data_file", "data.npz")) as data:
                masks = data["instance_mask"]
                render_ids = data["entity_render_object_id"]
                entity_ids = list(data["entity_id"].astype(str))
                for start, target in valid[:4]:
                    source_masks = masks[start:start + source["context_frames"], target]
                    per_kind = np.zeros_like(source_masks, dtype=bool)
                    for slot, slot_kind in kinds.items():
                        if slot_kind != kind or slot == f"agent{target}":
                            continue
                        ids = render_ids[start:start + len(source_masks), entity_ids.index(slot)]
                        per_kind |= (source_masks == ids[:, None, None]) & ((ids > 0) & (ids != 65535))[:, None, None]
                    coverage = per_kind.reshape(len(per_kind), -1).sum(-1)
                    for offset in range(0, len(coverage) - 8, 8):
                        future = coverage[offset + 1:offset + 9]
                        mean = float(future.mean())
                        if int(future.min()) < 600 or not 2500 <= mean <= 60000:
                            continue
                        if best is None or mean > best[0]:
                            best = (mean, start + offset, target, coverage[offset:offset + 9].tolist())
            if best is None:
                continue
            mean, start, target, coverage = best
            new_id = len(episodes)
            episodes.append(row)
            windows.append((new_id, start, target))
            hashes = {}
            for reference in sorted((path / "players").glob("agent*/*.png")):
                hashes[str(reference.relative_to(path))] = hashlib.sha256(reference.read_bytes()).hexdigest()
            audit.append({"path": str(path), "kind": kind, "start": start, "target": target,
                          "mean_future_kind_pixels": mean, "kind_pixels_per_frame": coverage,
                          "reference_sha256": hashes})
            used.add(episode_id)
            found += 1
            print(json.dumps({"split": source["split"], "kind": kind, "selected": found,
                              "start": start, "mean_pixels": mean, "episode": path.name}), flush=True)
            if found == count_per_kind:
                break
        if found != count_per_kind:
            raise RuntimeError(f"not enough visible {kind} clips: {found}/{count_per_kind}")
    payload = {key: source[key] for key in ("schema_version", "split", "item_vocabulary")}
    payload.update(context_frames=9, stride=8, episodes=episodes, windows=windows,
                   path_mode=source.get("path_mode", "relative_to_dataset_root"))
    return payload, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-kind", type=int, default=4)
    parser.add_argument("--val-per-kind", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260919)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports, probes = {}, []
    for split, count in (("train", args.train_per_kind), ("val_id", args.val_per_kind)):
        index = args.dataset_root / "derived/m3/validated" / f"{split}_c65.pt"
        payload, audit = select(args.dataset_root, index, count, args.seed)
        torch.save(payload, args.output_dir / f"{split}_c9.pt")
        reports[split] = audit
        seen = set()
        for i, row in enumerate(audit):
            if row["kind"] in seen:
                continue
            seen.add(row["kind"])
            probes.append({"name": f"{split}_{row['kind']}", "dataset_index": i,
                           "episode": row["path"], "start": row["start"], "target": row["target"],
                           "scenario_id": payload["episodes"][i]["manifest"]["scenario_id"], "event": None})
    if {row["path"] for row in reports["train"]} & {row["path"] for row in reports["val_id"]}:
        raise RuntimeError("train/held-out episodes overlap")
    (args.output_dir / "selection.json").write_text(json.dumps(reports, indent=2) + "\n")
    (args.output_dir / "probes.json").write_text(json.dumps(probes, indent=2) + "\n")


if __name__ == "__main__":
    main()
