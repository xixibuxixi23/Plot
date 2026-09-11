#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.data.fill_dataset import TextAgentFillDataset
from plot.geometry import crop_raw_49_to_tile_48


def build_one(manifest_path_string: str, overwrite: bool = False):
    manifest_path = Path(manifest_path_string)
    episode = manifest_path.parent
    cache = episode / "m1_initial.npz"
    if cache.exists() and not overwrite:
        return "skipped", str(episode)
    manifest = json.loads(manifest_path.read_text())
    validation = episode / "validation.json"
    if validation.exists() and not json.loads(validation.read_text()).get("usable", False):
        return "rejected", str(episode)
    source = episode / manifest.get("training_data_file", "data.npz")
    if not source.exists():
        return "missing", str(episode)
    with np.load(source, allow_pickle=False) as data:
        start = min(TextAgentFillDataset._model_start(data, manifest), len(data["obs_voxel_center"]) - 1)
        tiles = crop_raw_49_to_tile_48(np.asarray(data["obs_voxel_mt"][start]))
        centers = np.asarray(data["obs_voxel_center"][start], dtype=np.int16)
        cam_pos = np.asarray(data["cam_pos"][start], dtype=np.float32)
        cam_dir = np.asarray(data["cam_dir"][start], dtype=np.float32)
        fov_x = np.asarray(data["fov_x"][start], dtype=np.float32)
        fov_y = np.asarray(data["fov_y"][start], dtype=np.float32)
        intrinsics = np.asarray(data["intrinsics"][start], dtype=np.float32)
        raw_classes = np.unique(tiles[..., 0]).astype(np.int16)
    temp_cache = episode / f".m1_initial.{os.getpid()}.npz"
    with temp_cache.open("wb") as handle:
        np.savez_compressed(
            handle, voxel_tiles=tiles, voxel_center=centers, cam_pos=cam_pos,
            cam_dir=cam_dir, fov_x=fov_x, fov_y=fov_y, intrinsics=intrinsics,
            observation=np.asarray(start), raw_classes=raw_classes,
        )
    videos = manifest.get("agent_video_files") or [
        f"rgb_agent{i}.mp4" for i in range(int(manifest["num_agents"]))
    ]
    for index, video in enumerate(videos):
        capture = cv2.VideoCapture(str(episode / video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            temp_cache.unlink(missing_ok=True)
            return "video_error", str(episode)
        frame = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
        temp_image = episode / f".m1_rgb_agent{index}.{os.getpid()}.jpg"
        if not cv2.imwrite(str(temp_image), frame, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            temp_cache.unlink(missing_ok=True)
            return "video_error", str(episode)
        temp_image.replace(episode / f"m1_rgb_agent{index}.jpg")
    temp_cache.replace(cache)
    return "built", str(episode)


def main():
    parser = argparse.ArgumentParser(description="Cache M1 initial tiles, cameras and RGB views")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    wanted = set(args.split)
    manifests = set()
    ledgers = sorted(args.dataset_root.glob("plan_results_queue_*.jsonl"))
    if ledgers:
        root = args.dataset_root.resolve()
        for ledger in ledgers:
            with ledger.open() as handle:
                for line in handle:
                    row = json.loads(line)
                    if not row.get("success") or not row.get("validation", {}).get("usable"):
                        continue
                    if wanted and row.get("split") not in wanted:
                        continue
                    episode = Path(row["output_dir"]).resolve()
                    if root not in episode.parents:
                        raise ValueError(f"ledger output is outside dataset root: {episode}")
                    manifest = episode / "manifest.json"
                    if manifest.exists():
                        manifests.add(str(manifest))
    else:
        for path in args.dataset_root.rglob("manifest.json"):
            manifest = json.loads(path.read_text())
            if not wanted or manifest.get("split") in wanted:
                manifests.add(str(path))
    manifests = sorted(manifests)
    counts = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for number, (status, path) in enumerate(
            pool.map(build_one, manifests, [args.overwrite] * len(manifests)), 1
        ):
            counts[status] = counts.get(status, 0) + 1
            if status not in {"built", "skipped", "rejected"}:
                print(f"{status}: {path}", flush=True)
            if number % 100 == 0:
                print(f"processed={number}/{len(manifests)} counts={counts}", flush=True)
    print(f"complete total={len(manifests)} counts={counts}")


if __name__ == "__main__":
    main()
