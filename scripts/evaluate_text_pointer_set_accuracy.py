"""Evaluate exact-next and any-remaining-goal target-pointer accuracy."""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from plot.data.state_policy_dataset import CachedStatePolicyDataset, collate_state_policy
from plot.models.state_policy_text_v2 import IndependentStatePolicyV4, TextStatePolicyV2Args


VOXELS = 13 ** 3


def event_position(event):
    value = event.get("position")
    if not isinstance(value, dict) or not all(key in value for key in ("x", "y", "z")):
        return None
    return np.asarray([value["x"], value["z"], value["y"]], np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-scenario", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = IndependentStatePolicyV4(TextStatePolicyV2Args(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    dataset = CachedStatePolicyDataset(args.cache / "val_id.jsonl", "language_builder")
    summary = json.loads((args.cache / "summary.json").read_text())
    root = Path(summary["dataset_root"]) / "val_id"
    indices = list(range(len(dataset)))
    if args.per_scenario:
        grouped = defaultdict(list)
        for index, row in enumerate(dataset.rows):
            grouped[row.get("scenario_id", "unknown")].append(index)
        rng = random.Random(20260923)
        indices = []
        for scenario in sorted(grouped):
            rng.shuffle(grouped[scenario])
            indices.extend(grouped[scenario][:args.per_scenario])
        rng.shuffle(indices)
    if args.limit:
        indices = indices[:args.limit]
    count = len(indices)

    @lru_cache(maxsize=16)
    def episode_info(episode_id):
        episode = root / episode_id
        manifest = json.loads((episode / "manifest.json").read_text())
        data_path = episode / manifest.get("training_data_file", "data.npz")
        with np.load(data_path, allow_pickle=False) as data:
            positions = data["player_pos"].astype(np.float32)
        placements = []
        event_path = episode / manifest.get("event_file", "events.jsonl")
        for line in event_path.open():
            event = json.loads(line)
            if event.get("initialization_event") or event.get("event") not in (
                    "block_placed", "scaffold_placed"):
                continue
            coordinate = event_position(event)
            transition = event.get("transition_index")
            if coordinate is not None and transition is not None:
                placements.append((int(transition), tuple(map(int, coordinate))))
        placements.sort()
        return positions, placements, manifest.get("scenario_id", "unknown")

    totals = defaultdict(lambda: defaultdict(int))
    examples = []
    with torch.no_grad():
        for begin in range(0, count, args.batch_size):
            batch_indices = indices[begin:min(begin + args.batch_size, count)]
            batch = collate_state_policy([dataset[index] for index in batch_indices])
            inputs = {key: value.to(args.device) for key, value in batch["inputs"].items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                target_logits = model(inputs)["target"]
                prediction = target_logits.argmax(-1).cpu().numpy()
                null_index = target_logits.shape[-1] - 1
            exact = batch["target_address"].numpy()
            for offset, index in enumerate(batch_indices):
                row = dataset.rows[index]
                positions, placements, scenario = episode_info(row["episode_id"])
                frame = int(row["anchor"])
                target = int(row["agent_slot"])
                center = np.rint(positions[frame, target]).astype(np.int64)
                future = {coordinate for transition, coordinate in placements
                          if transition >= frame}
                valid = set()
                for coordinate in future:
                    local = np.asarray(coordinate) - (center - 6)
                    if (local >= 0).all() and (local < 13).all():
                        valid.add(int(np.ravel_multi_index(tuple(local), (13, 13, 13))))
                predicted = int(prediction[offset])
                groups = ("all", scenario)
                for group in groups:
                    stats = totals[group]
                    if exact[offset] >= 0:
                        stats["exact_rows"] += 1
                        stats["exact_correct"] += int(predicted == int(exact[offset]))
                    if valid:
                        stats["set_rows"] += 1
                        stats["set_correct"] += int(predicted in valid)
                        stats["set_size_sum"] += len(valid)
                        if 0 <= exact[offset] < VOXELS:
                            stats["voxel_rows"] += 1
                            stats["voxel_exact_correct"] += int(predicted == int(exact[offset]))
                            stats["voxel_set_correct"] += int(predicted in valid)
                    elif not future:
                        stats["finished_rows"] += 1
                        stats["finished_null_correct"] += int(predicted == null_index)
                    else:
                        stats["future_outside_rows"] += 1
                if len(examples) < 20 and valid and predicted in valid and predicted != exact[offset]:
                    examples.append(dict(episode=row["episode_id"], frame=frame,
                                         scenario=scenario, exact=int(exact[offset]),
                                         predicted=predicted, valid_targets=len(valid)))

    result = {"checkpoint": str(args.checkpoint), "rows": count, "groups": {},
              "definition": "prediction is correct when it selects any future placed coordinate still local to the agent",
              "alternative_correct_examples": examples}
    for group, stats in totals.items():
        values = dict(stats)
        values["exact_accuracy"] = values.get("exact_correct", 0) / max(1, values.get("exact_rows", 0))
        values["set_accuracy"] = values.get("set_correct", 0) / max(1, values.get("set_rows", 0))
        values["voxel_exact_accuracy"] = values.get("voxel_exact_correct", 0) / max(1, values.get("voxel_rows", 0))
        values["voxel_set_accuracy"] = values.get("voxel_set_correct", 0) / max(1, values.get("voxel_rows", 0))
        values["mean_valid_targets"] = values.get("set_size_sum", 0) / max(1, values.get("set_rows", 0))
        values["finished_null_accuracy"] = values.get("finished_null_correct", 0) / max(1, values.get("finished_rows", 0))
        result["groups"][group] = values
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
