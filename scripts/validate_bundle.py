#!/usr/bin/env python3
"""CPU-only validation for a moved PLOT code/data bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--check-checkpoint-hashes", action="store_true")
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    if not root.is_dir():
        parser.error(f"dataset root does not exist: {root}")

    from plot.data.fill_dataset import BlockVocabulary
    from plot.data.renderer_dataset import TextAgentRendererDataset

    vocabulary = BlockVocabulary.load(ROOT / "derived/common/block_vocabulary.json")
    index_path = ROOT / "derived/m3/train_c65.pt"
    index = torch.load(index_path, map_location="cpu", weights_only=False)
    if index.get("path_mode") != "relative_to_dataset_root":
        raise RuntimeError("M3 index is not portable")
    first_m3 = root / index["episodes"][0]["path"]
    if not (first_m3 / "manifest.json").is_file():
        raise FileNotFoundError(f"M3 index cannot resolve under dataset root: {first_m3}")

    first_m4 = json.loads((ROOT / "derived/m4/train.jsonl").open().readline())
    if Path(first_m4["episode_path"]).is_absolute():
        raise RuntimeError("M4 index contains an absolute path")
    if not (root / first_m4["episode_path"] / "manifest.json").is_file():
        raise FileNotFoundError("M4 index cannot resolve under dataset root")

    manifest = json.loads((ROOT / "checkpoints/MANIFEST.json").read_text())
    for asset in manifest["assets"]:
        path = ROOT / "checkpoints" / asset["path"]
        if not path.is_file() or path.stat().st_size != asset["bytes"]:
            raise RuntimeError(f"missing or truncated checkpoint asset: {path}")
        if args.check_checkpoint_hashes and sha256(path) != asset["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")

    dataset = TextAgentRendererDataset(
        root, ROOT / "derived/common/block_vocabulary.json", split="train",
        context_frames=65, window_index=index_path, targets_per_window=2,
    )
    print(json.dumps({
        "status": "ok", "dataset_root": str(root),
        "block_classes": vocabulary.size, "m3_windows_two_view": len(dataset),
        "m4_first_episode": first_m4["episode_path"],
        "checkpoint_hashes_checked": args.check_checkpoint_hashes,
    }, indent=2))


if __name__ == "__main__":
    main()
