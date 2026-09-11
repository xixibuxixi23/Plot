#!/usr/bin/env python3
"""Select causal M4 windows from a canonical continuous PLOT release."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from plot.data.fill_dataset import TextAgentFillDataset


SUPPORTED = {"language_builder", "villager", "combat"}


def spread(values, limit):
    if len(values) <= limit:
        return values
    indices = np.linspace(0, len(values) - 1, limit).round().astype(int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def episode_windows(job):
    row, max_windows, max_wait_windows, release = job
    # Prefer the extracted release layout; source_path is provenance and may
    # refer to the machine that originally assembled the release.
    episode = Path(release) / row["split"] / row["episode_id"]
    if not episode.is_dir():
        source = Path(row["source_path"])
        episode = source if source.is_absolute() else Path(release) / source
    manifest = json.loads((episode / "manifest.json").read_text())
    routes = manifest.get("behavior_routes") or {}
    if row["scenario_id"] == "S01" and not routes:
        routes = {
            f"agent{i}": {"family": "language_builder", "profile": "language_builder"}
            for i in range(int(manifest["num_agents"]))
        }
    results = []
    with np.load(episode / "data.npz", allow_pickle=False) as data:
        actions = data["action_continuous"]
        model_start = TextAgentFillDataset._model_start(data, manifest)
        frames = len(actions) + 1
        if frames - model_start < 65:
            return results
        intentional = data.get("intentional_wait_mask", np.zeros(actions.shape[:2], bool))
        health = data["player_health"]
        health_valid = data["player_health_valid"].astype(bool)
        termination = np.asarray(data["termination_flag"], bool)
        if termination.ndim > 1:
            termination = termination.any(axis=tuple(range(1, termination.ndim)))
        for agent in range(actions.shape[1]):
            route = routes.get(f"agent{agent}", {})
            family, profile = route.get("family"), route.get("profile")
            if family not in SUPPORTED:
                continue
            mask_key = "language_policy_train_mask" if family == "language_builder" else "npc_policy_train_mask"
            if mask_key not in data:
                mask_key = "native_npc_policy_train_mask" if family != "language_builder" else "policy_train_mask"
            valid = np.asarray(data[mask_key][:, agent], bool)
            candidates = []
            for anchor in range(model_start + 8, len(actions) - 7, 8):
                if not valid[anchor:anchor + 8].all():
                    continue
                if termination[anchor:anchor + 8].any():
                    continue
                if not health_valid[anchor-8:anchor + 9, agent].all():
                    continue
                if (health[anchor-8:anchor + 9, agent] <= 0).any():
                    continue
                target = actions[anchor:anchor + 8, agent]
                if family == "combat" and np.any(target[:, 8] > 0):
                    priority = "attack"
                elif family == "language_builder" and np.any(target[:, [8, 9]] > 0):
                    priority = "edit"
                elif np.any(intentional[anchor:anchor + 8, agent]):
                    priority = "intentional_wait"
                else:
                    priority = "ordinary"
                candidates.append((anchor, priority))
            # Preserve every rare causal edit/attack. Waiting is common and is
            # sampled sparsely so that all-zero targets do not dominate M4.
            important = [value for value in candidates if value[1] in {"attack", "edit"}]
            waits = spread([value for value in candidates if value[1] == "intentional_wait"],
                           max_wait_windows)
            ordinary = [value for value in candidates if value[1] == "ordinary"]
            selected = sorted(important + waits + spread(
                ordinary, max(0, max_windows - len(waits))))
            task_text = manifest.get("task_text", "")
            shared_text = task_text if family == "language_builder" else ""
            per_agent_text = (manifest.get("agent_task_texts") or
                              manifest.get("agent_subtasks") or {})
            current_text = per_agent_text.get(f"agent{agent}", shared_text)
            for anchor, priority in selected:
                results.append({
                    "episode_id": row["episode_id"],
                    "episode_path": str(episode.resolve().relative_to(Path(release).resolve())),
                    "split": row["split"], "scenario_id": row["scenario_id"],
                    "task_id": row["task_id"], "task_text": task_text,
                    **({"shared_task_text": shared_text} if shared_text != task_text else {}),
                    **({"current_task_text": current_text} if current_text != shared_text else {}),
                    "agent_slot": agent, "anchor": anchor, "family": family,
                    "context_start": min(max(model_start, anchor - 64), frames - 65),
                    "profile": profile, "priority": priority,
                })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-windows-per-agent", type=int, default=12)
    parser.add_argument("--max-wait-windows-per-agent", type=int, default=2)
    args = parser.parse_args()
    splits = args.split or ["train", "val_id"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": "plot-m4-index-v2",
               "path_mode": "relative_to_dataset_root",
               "max_windows_per_agent": args.max_windows_per_agent,
               "max_wait_windows_per_agent": args.max_wait_windows_per_agent, "splits": {}}
    for split in splits:
        source = args.release / f"{split}.jsonl"
        rows = [json.loads(line) for line in source.open()]
        records = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            for number, result in enumerate(
                executor.map(episode_windows, ((row, args.max_windows_per_agent,
                                                args.max_wait_windows_per_agent, args.release)
                                               for row in rows), chunksize=8), 1
            ):
                records.extend(result)
                if number % 1000 == 0:
                    print(json.dumps({"split": split, "episodes": number, "windows": len(records)}), flush=True)
        profiles = Counter(row["profile"] for row in records)
        largest = max(profiles.values(), default=1)
        for row in records:
            # Weighted replacement sampling should expose every profile equally;
            # clipping only protects against a genuinely tiny/corrupt stratum.
            balance = largest / profiles[row["profile"]]
            priority = {"attack": 2.0, "edit": 1.5, "intentional_wait": 1.25}.get(row["priority"], 1.0)
            row["sample_weight"] = min(64.0, balance) * priority
        output = args.output_dir / f"{split}.jsonl"
        with output.open("w") as handle:
            for row in records:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["splits"][split] = {
            "episodes_scanned": len(rows), "windows": len(records),
            "families": dict(Counter(row["family"] for row in records)),
            "profiles": dict(profiles),
            "priorities": dict(Counter(row["priority"] for row in records)),
            "scenarios": dict(Counter(row["scenario_id"] for row in records)),
            "weighted_profiles": {
                profile: sum(row["sample_weight"] for row in records if row["profile"] == profile)
                for profile in profiles
            },
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
