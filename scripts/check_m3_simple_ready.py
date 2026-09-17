#!/usr/bin/env python3
"""Fail fast on the portable inputs required by the M3-Simple recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED_PIXEL_VAE_BYTES = 909_781_080
EXPECTED_PIXEL_VAE_SHA256 = (
    "eb634803c94aeea980046961382f2ab67157aa71e92c5a183a35e4f61f8cbc36"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_index(dataset_root: Path, path: Path, split: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    index = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "schema_version": "plot-m3-window-index-v2",
        "split": split,
        "context_frames": 65,
        "path_mode": "relative_to_dataset_root",
    }
    for key, value in expected.items():
        if index.get(key) != value:
            raise RuntimeError(f"{path}: expected {key}={value!r}, got {index.get(key)!r}")
    episodes = index.get("episodes", [])
    windows = index.get("windows", [])
    if not episodes or not windows:
        raise RuntimeError(f"{path}: empty episodes or windows")
    for episode_index in sorted({0, len(episodes) // 2, len(episodes) - 1}):
        episode = dataset_root / episodes[episode_index]["path"]
        for relative in ("manifest.json", "data.npz", "rgb_agent0.mp4"):
            if not (episode / relative).is_file():
                raise FileNotFoundError(episode / relative)
    return {
        "path": str(path),
        "episodes": len(episodes),
        "windows": len(windows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--pixel-vae",
        type=Path,
        default=Path("checkpoints/pixel_vae/model.safetensors"),
    )
    parser.add_argument("--check-pixel-vae-hash", action="store_true")
    parser.add_argument("--check-wandb", action="store_true")
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    completion = load_json(root / "COMPLETE.json")
    split_completion = load_json(root / "M3_SPLIT_COMPLETE.json")
    if not completion.get("complete"):
        raise RuntimeError("dataset COMPLETE.json does not mark the release complete")
    if completion.get("missing_count") or completion.get("error_count"):
        raise RuntimeError("dataset completion reports missing or failed episodes")
    if split_completion.get("total_episodes") != completion.get("expected"):
        raise RuntimeError("M3 split count does not match dataset completion count")

    index_root = root / "derived/m3/validated"
    train = inspect_index(root, index_root / "train_c65.pt", "train")
    val_id = inspect_index(root, index_root / "val_id_c65.pt", "val_id")
    if train["episodes"] != split_completion.get("train_episodes"):
        raise RuntimeError("train index episode count does not match split manifest")
    if val_id["episodes"] != split_completion.get("val_id_episodes"):
        raise RuntimeError("val-ID index episode count does not match split manifest")

    pixel_vae = args.pixel_vae.resolve()
    if not pixel_vae.is_file() or pixel_vae.stat().st_size != EXPECTED_PIXEL_VAE_BYTES:
        raise RuntimeError(f"missing or truncated Pixel-VAE: {pixel_vae}")
    pixel_vae_hash = "not_checked"
    if args.check_pixel_vae_hash:
        pixel_vae_hash = sha256(pixel_vae)
        if pixel_vae_hash != EXPECTED_PIXEL_VAE_SHA256:
            raise RuntimeError(f"Pixel-VAE SHA-256 mismatch: {pixel_vae}")

    report = {
        "status": "ok",
        "dataset_root": str(root),
        "dataset_episodes": completion["expected"],
        "train": train,
        "val_id": val_id,
        "pixel_vae": str(pixel_vae),
        "pixel_vae_sha256": pixel_vae_hash,
    }
    if args.check_wandb:
        import wandb

        api = wandb.Api(timeout=20)
        report["wandb"] = {
            "username": api.viewer.username,
            "entity": api.default_entity,
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
