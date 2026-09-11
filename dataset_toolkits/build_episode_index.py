#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index usable TextAgent episodes once for DDP")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    splits: dict[str, dict[str, dict]] = {}

    def add_episode(episode: Path, split: str) -> None:
        manifest = json.loads((episode / "manifest.json").read_text())
        relative = str(episode.relative_to(root))
        kept_keys = (
            "action_steps", "agent_video_files", "episode_id", "height",
            "model_start_observation", "num_agents", "scenario_id", "split",
            "training_data_file", "width",
        )
        slim_manifest = {key: manifest[key] for key in kept_keys if key in manifest}
        splits.setdefault(split, {})[relative] = {
            "path": relative,
            "manifest": slim_manifest,
        }

    # Scan the extracted release itself. Collection ledgers can contain absolute
    # source-machine output paths and therefore are not a portable authority.
    for manifest_path in root.rglob("manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        validation_path = manifest_path.parent / "validation.json"
        if validation_path.exists() and not json.loads(validation_path.read_text()).get("usable"):
            continue
        split = str(manifest.get("split", "unspecified"))
        add_episode(manifest_path.parent, split)
    payload = {
        "schema_version": "plot-episode-index-v2",
        "dataset_root": ".",
        "path_mode": "relative_to_dataset_root",
        "splits": {
            split: [records[path] for path in sorted(records)]
            for split, records in sorted(splits.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(
        " ".join([f"{split}={len(records)}" for split, records in payload["splits"].items()])
        + f" output={args.output}"
    )


if __name__ == "__main__":
    main()
