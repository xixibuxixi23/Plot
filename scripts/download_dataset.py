#!/usr/bin/env python3
"""Download and extract PLOT tar shards one at a time to bound disk overhead."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from huggingface_hub import hf_hub_download


REPO = "xixibuxixi/polis-v1"
DEFAULT_VERSION = "fixed_skins_20260917"
METADATA = (
    "README.md", "COMPLETE.json", "DERIVATION.json", "release.json",
    "shards.jsonl", "train.jsonl", "val_id.jsonl", "test_id.jsonl",
    "episodes.jsonl", "balanced_v1_train.jsonl", "M3_SPLIT_PLAN.json",
    "M3_SPLIT_COMPLETE.json", "SHARDING_COMPLETE.json",
)
ARTIFACTS = (
    "derived/m3/validated/train_c65.pt",
    "derived/m3/validated/val_id_c65.pt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument(
        "--version", default=DEFAULT_VERSION,
        help="directory prefix in the Hugging Face dataset repo",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_cache = output / ".download_metadata"
    for name in METADATA:
        source = Path(hf_hub_download(
            REPO, f"{args.version}/{name}", repo_type="dataset", revision=args.revision,
            cache_dir=metadata_cache,
        ))
        shutil.copy2(source, output / name)
    for name in ARTIFACTS:
        source = Path(hf_hub_download(
            REPO, f"{args.version}/{name}", repo_type="dataset", revision=args.revision,
            cache_dir=metadata_cache,
        ))
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    rows = [json.loads(line) for line in (output / "shards.jsonl").read_text().splitlines()]
    selected = [row for row in rows if Path(row["path"]).parts[1] in args.splits]
    def marker_for(row: dict) -> Path:
        relative = Path(row["path"])
        return output / ".extracted" / relative.parent.name / (relative.name + ".json")

    total = sum(int(row["bytes"]) for row in selected)
    remaining = [row for row in selected if not marker_for(row).exists()]
    remaining_bytes = sum(int(row["bytes"]) for row in remaining)
    largest = max((int(row["bytes"]) for row in remaining), default=0)
    required = remaining_bytes + largest
    free = shutil.disk_usage(output).free
    if free < required:
        raise RuntimeError(
            f"insufficient free space: need at least {required / 2**30:.1f} GiB "
            f"for extracted data plus one shard, found {free / 2**30:.1f} GiB"
        )

    completed = 0
    for number, row in enumerate(selected, 1):
        relative = Path(row["path"])
        marker = marker_for(row)
        if marker.exists():
            completed += int(row["bytes"])
            print(f"[{number}/{len(selected)}] already extracted {relative}", flush=True)
            continue
        shard_cache = output / ".shard_cache"
        archive = Path(hf_hub_download(
            REPO, f"{args.version}/{relative}", repo_type="dataset", revision=args.revision,
            cache_dir=shard_cache,
        ))
        actual_bytes = archive.stat().st_size
        if actual_bytes != int(row["bytes"]):
            raise RuntimeError(
                f"shard size mismatch for {relative}: expected {row['bytes']}, "
                f"got {actual_bytes}"
            )
        actual_sha256 = sha256(archive)
        if actual_sha256 != row["sha256"]:
            raise RuntimeError(
                f"shard SHA-256 mismatch for {relative}: expected {row['sha256']}, "
                f"got {actual_sha256}"
            )
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
        "version": args.version, "shards": len(selected), "bytes": total,
    }, indent=2))


if __name__ == "__main__":
    main()
