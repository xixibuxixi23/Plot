#!/usr/bin/env python3
"""Build a portable, disposable chunk cache for M3 training.

The output index stores source episode paths relative to --source-root and
cache paths relative to --output-root.  It can therefore be copied between
machines without retaining paths from the machine that created it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data.chunked_npz import (
    DEFAULT_CHUNKED_KEYS,
    SCHEMA_VERSION,
    validate_chunked_npz,
    write_chunked_npz,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--output-index",
        help="Default: <output-root>/<input-stem>_chunk<N>.pt",
    )
    parser.add_argument("--chunk-frames", type=int, choices=(8, 16), default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--compression-level", type=int, choices=range(10), default=6)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _episode_relative_path(raw_path, source_root):
    path = Path(raw_path)
    if path.is_absolute():
        try:
            return path.resolve().relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"indexed episode {path} is not below --source-root {source_root}"
            ) from error
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe relative episode path: {path}")
    return path


def _convert_one(job):
    source, destination, chunk_frames, compression_level, overwrite = job
    source, destination = Path(source), Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    if not overwrite and destination.is_file() and validate_chunked_npz(
        destination,
        expected_chunk_frames=chunk_frames,
        expected_keys=DEFAULT_CHUNKED_KEYS,
    ):
        status = "reused"
    else:
        write_chunked_npz(
            source,
            destination,
            chunk_frames=chunk_frames,
            chunked_keys=DEFAULT_CHUNKED_KEYS,
            compression_level=compression_level,
        )
        if not validate_chunked_npz(
            destination,
            expected_chunk_frames=chunk_frames,
            expected_keys=DEFAULT_CHUNKED_KEYS,
        ):
            raise RuntimeError(f"cache validation failed: {destination}")
        status = "written"
    return status, source.stat().st_size, destination.stat().st_size


def _atomic_torch_save(payload, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json_save(payload, destination):
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main():
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve()
    input_index = Path(args.window_index).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_index = (
        Path(args.output_index).resolve()
        if args.output_index
        else output_root / f"{input_index.stem}_chunk{args.chunk_frames}.pt"
    )
    payload = torch.load(input_index, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("window index has no episodes")

    rewritten = []
    jobs = []
    for row in episodes:
        relative_episode = _episode_relative_path(row["path"], source_root)
        manifest = dict(row["manifest"])
        source_name = manifest.get("training_data_file", "data.npz")
        source_file = source_root / relative_episode / source_name
        cache_name = f"{Path(source_name).stem}.m3c{args.chunk_frames}.npz"
        relative_cache = relative_episode / cache_name
        cache_file = output_root / relative_cache
        manifest["m3_chunk_cache_file"] = relative_cache.as_posix()
        manifest["m3_chunk_frames"] = args.chunk_frames
        manifest["m3_chunk_schema_version"] = SCHEMA_VERSION
        rewritten.append({"path": relative_episode.as_posix(), "manifest": manifest})
        jobs.append((
            str(source_file), str(cache_file), args.chunk_frames,
            args.compression_level, args.overwrite,
        ))

    counts = {"written": 0, "reused": 0}
    source_bytes = cache_bytes = 0
    if args.workers == 1:
        iterator = map(_convert_one, jobs)
        for completed, result in enumerate(iterator, 1):
            status, source_size, cache_size = result
            counts[status] += 1
            source_bytes += source_size
            cache_bytes += cache_size
            print(f"[{completed}/{len(jobs)}] {status}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_convert_one, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), 1):
                status, source_size, cache_size = future.result()
                counts[status] += 1
                source_bytes += source_size
                cache_bytes += cache_size
                print(f"[{completed}/{len(jobs)}] {status}", flush=True)

    portable = dict(payload)
    portable["episodes"] = rewritten
    portable["m3_chunk_cache"] = {
        "schema_version": SCHEMA_VERSION,
        "chunk_frames": args.chunk_frames,
        "chunked_keys": list(DEFAULT_CHUNKED_KEYS),
    }
    _atomic_torch_save(portable, output_index)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "chunk_frames": args.chunk_frames,
        "chunked_keys": list(DEFAULT_CHUNKED_KEYS),
        "episodes": len(episodes),
        **counts,
        "source_bytes": source_bytes,
        "cache_bytes": cache_bytes,
        "output_index": os.path.relpath(output_index, output_root),
    }
    _atomic_json_save(summary, output_root / "cache_manifest.json")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
