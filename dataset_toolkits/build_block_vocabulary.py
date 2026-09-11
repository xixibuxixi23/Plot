from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def accepted_manifests(dataset_root: Path, split: str) -> list[Path]:
    """Prefer collector ledgers, which avoid traversing rejected episode trees."""
    ledgers = sorted(dataset_root.glob("plan_results_queue_*.jsonl"))
    if not ledgers:
        return sorted(dataset_root.rglob("manifest.json"))

    root = dataset_root.resolve()
    manifests: set[Path] = set()
    for ledger in ledgers:
        with ledger.open() as handle:
            for line in handle:
                row = json.loads(line)
                if not row.get("success") or not row.get("validation", {}).get("usable"):
                    continue
                if split and row.get("split") != split:
                    continue
                episode = Path(row["output_dir"]).resolve()
                if root not in episode.parents:
                    raise ValueError(f"ledger output is outside dataset root: {episode}")
                manifest = episode / "manifest.json"
                if manifest.exists():
                    manifests.add(manifest)
    return sorted(manifests)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a raw-content-ID vocabulary")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: set[int] = set()
    episodes = 0
    for manifest_path in accepted_manifests(args.dataset_root, args.split):
        manifest = json.loads(manifest_path.read_text())
        if args.split and manifest.get("split") != args.split:
            continue
        validation = manifest_path.parent / "validation.json"
        if validation.exists() and not json.loads(validation.read_text()).get("usable"):
            continue
        data_path = manifest_path.parent / manifest.get("training_data_file", "data.npz")
        if not data_path.exists():
            continue
        cache_path = manifest_path.parent / "m1_initial.npz"
        with np.load(cache_path if cache_path.exists() else data_path, allow_pickle=False) as data:
            if cache_path.exists() and "raw_classes" in data:
                classes = data["raw_classes"]
            else:
                raw = np.asarray(data["obs_voxel_mt"])[..., 0]
                classes = np.unique(raw)
            values.update(int(v) for v in classes if int(v) != 127)
        episodes += 1
        if episodes % 100 == 0:
            print(f"scanned={episodes} classes={len(values)}", flush=True)
    if not values:
        raise RuntimeError("no usable voxel content IDs found")
    result = {
        "schema_version": "plot-block-vocabulary-v1",
        "class_to_raw": sorted(values),
        "content_ignore_raw_id": 127,
        "episodes_scanned": episodes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {len(values)} classes from {episodes} episodes to {args.output}")


if __name__ == "__main__":
    main()
