#!/usr/bin/env python3
"""Capture immutable source, data, command, and host provenance for a run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
from datetime import datetime, timezone


SCHEMA_VERSION = "plot-run-manifest-v1"


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_inventory() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    result = []
    for line in output.splitlines():
        index, name, memory = (part.strip() for part in line.split(",", 2))
        result.append({"index": index, "name": name, "memory_mib": memory})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("m1", "m2", "m3", "m4"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--command-file", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(git(Path.cwd(), "rev-parse", "--show-toplevel"))
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status and not args.allow_dirty:
        raise SystemExit(
            "Refusing to register a dirty source tree. Commit the run source first, "
            "or use --allow-dirty only for a documented diagnostic run."
        )
    command = args.command_file.read_text().strip()
    forbidden = ("TOKEN=", "PASSWORD=", "SECRET=", "API_KEY=")
    if any(marker in command.upper() for marker in forbidden):
        raise SystemExit("Command file appears to contain a credential; use environment injection.")

    parent = None
    if args.parent_checkpoint:
        checkpoint = args.parent_checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"Parent checkpoint does not exist: {checkpoint}")
        parent = {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": sha256(checkpoint),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": args.run_id,
        "model": args.model,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": git(root, "remote", "get-url", "origin"),
            "commit": git(root, "rev-parse", "HEAD"),
            "branch": git(root, "branch", "--show-current") or None,
            "dirty": bool(status),
            "dirty_status": status.splitlines() if status else [],
        },
        "dataset": {
            "repository": args.dataset_repo,
            "revision": args.dataset_revision,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "user": os.environ.get("USER"),
            "gpus": gpu_inventory(),
        },
        "parent_checkpoint": parent,
        "command": command,
        "output_dir": str(args.output_dir.expanduser().resolve()),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "run_manifest.json"
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite existing manifest: {destination}")
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(destination)
    print(destination)


if __name__ == "__main__":
    main()
