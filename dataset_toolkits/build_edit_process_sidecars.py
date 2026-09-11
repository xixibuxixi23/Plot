"""Post-process recorded episodes into compact edit-process supervision."""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from plot.data.edit_process import derive_edit_process
from plot.data.transition_dataset import aligned_events


def _one(job):
    episode, destination = map(Path, job)
    manifest = json.loads((episode / "manifest.json").read_text())
    with np.load(episode / manifest.get("training_data_file", "data.npz")) as data:
        actions = data["action_continuous"]
        fields = [data[key] for key in (
            "pointed_type", "pointed_node_under", "pointed_node_above", "dt_minetest")]
    events = [json.loads(line) for line in
              (episode / manifest.get("event_file", "events.jsonl")).read_text().splitlines()
              if line.strip()]
    by_step = aligned_events(events, len(actions))
    agents = [f"agent{i}" for i in range(actions.shape[1])]
    trace = derive_edit_process(actions, *fields, by_step, agents)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **trace)
    temporary.replace(destination)
    return {
        "events": sum(event.get("event") in {
            "block_dug", "scaffold_dug", "block_placed", "scaffold_placed"}
            for records in by_step.values() for event in records),
        "explained": int(trace["completed"].sum()),
        "dig_completed": int((trace["completed"] & (trace["kind"] == 1)).sum()),
        "place_completed": int((trace["completed"] & (trace["kind"] == 2)).sum()),
        "delayed": int((trace["completed"] & (actions[..., 8] <= .5)
                        & (actions[..., 9] <= .5)).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val_id"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    index = json.loads(Path(args.index).read_text())
    root = Path(args.dataset_root)
    output = Path(args.output_root)
    jobs = []
    for split in args.splits:
        for record in index["splits"][split]:
            episode = root / record["path"]
            jobs.append((episode, output / split / f"{episode.name}.npz"))
    totals = {key: 0 for key in
              ("episodes", "events", "explained", "dig_completed", "place_completed", "delayed")}
    with Pool(args.workers) as pool:
        for count, result in enumerate(pool.imap_unordered(_one, jobs), 1):
            totals["episodes"] += 1
            for key, value in result.items():
                totals[key] += value
            if count % 500 == 0:
                print(json.dumps(totals), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    totals["coverage"] = totals["explained"] / max(1, totals["events"])
    (output / "summary.json").write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
