#!/usr/bin/env python3
"""Download and extract PLOT tar shards one at a time to bound disk overhead."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tarfile

from huggingface_hub import hf_hub_download


REPO = "xixibuxixi/polis-v1"
METADATA = (
    "README.md", "COMPLETE.json", "DERIVATION.json", "release.json",
    "shards.jsonl", "train.jsonl", "val_id.jsonl", "test_id.jsonl",
)


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r:") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"unsafe tar member: {member.name}")
        handle.extractall(destination, filter="data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--splits", nargs="+", choices=("train", "val_id", "test_id"),
        default=("train", "val_id"),
    )
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_cache = output / ".download_metadata"
    for name in METADATA:
        source = Path(hf_hub_download(
            REPO, name, repo_type="dataset", revision=args.revision,
            cache_dir=metadata_cache,
        ))
        shutil.copy2(source, output / name)

    rows = [json.loads(line) for line in (output / "shards.jsonl").read_text().splitlines()]
    selected = [row for row in rows if Path(row["path"]).parts[1] in args.splits]
    total = sum(int(row["bytes"]) for row in selected)
    largest = max(int(row["bytes"]) for row in selected)
    required = total + largest
    free = shutil.disk_usage(output).free
    if free < required:
        raise RuntimeError(
            f"insufficient free space: need at least {required / 2**30:.1f} GiB "
            f"for extracted data plus one shard, found {free / 2**30:.1f} GiB"
        )

    completed = 0
    for number, row in enumerate(selected, 1):
        relative = Path(row["path"])
        marker = output / ".extracted" / (relative.name + ".json")
        if marker.exists():
            completed += int(row["bytes"])
            print(f"[{number}/{len(selected)}] already extracted {relative}", flush=True)
            continue
        shard_cache = output / ".shard_cache"
        archive = Path(hf_hub_download(
            REPO, str(relative), repo_type="dataset", revision=args.revision,
            cache_dir=shard_cache,
        ))
        print(f"[{number}/{len(selected)}] extracting {relative}", flush=True)
        safe_extract(archive, output)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(row, sort_keys=True) + "\n")
        shutil.rmtree(shard_cache, ignore_errors=True)
        completed += int(row["bytes"])
        print(f"completed {completed / total:.1%}", flush=True)
    shutil.rmtree(metadata_cache, ignore_errors=True)
    print(json.dumps({
        "status": "complete", "output": str(output), "splits": args.splits,
        "shards": len(selected), "bytes": total,
    }, indent=2))


if __name__ == "__main__":
    main()
