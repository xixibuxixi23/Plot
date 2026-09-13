"""Collect a distributed fixed-audit prediction into a small visual gallery."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.train_multiview_persist_full import Cache, Source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--audit-samples", type=int, default=80)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--gallery-samples", type=int, default=8)
    args = parser.parse_args()

    root = ROOT
    cache = Cache(root / "outputs/m1_multiview_persist_s01_cache")
    audit = np.linspace(0, len(cache) - 1, args.audit_samples).round().astype(int)
    predictions = {}
    metrics = {}
    for rank in range(args.world_size):
        indices = audit[rank:: args.world_size].tolist()
        result = torch.load(
            args.output / f"rank_{rank}/prediction_{args.step:06d}.pt", weights_only=False
        )
        for index, pred, row in zip(indices, result["pred"], result["metrics"]["samples"]):
            predictions[index] = pred
            metrics[index] = row
    if sorted(predictions) != audit.tolist():
        raise RuntimeError("Distributed audit predictions do not cover the expected samples")

    # Include difficult, median and strong cases while spanning the dataset.
    ranked = sorted(audit.tolist(), key=lambda i: metrics[i]["surface_exact_recall"])
    positions = np.linspace(0, len(ranked) - 1, args.gallery_samples).round().astype(int)
    chosen = [ranked[i] for i in positions]
    source = Source(
        SimpleNamespace(dataset_root=Path(cache.meta["dataset_root"]), persist=root.parent / "PERSIST")
    )
    gallery = args.output / f"gallery_{args.step:06d}"
    gallery.mkdir(exist_ok=True)
    selected = cache.evaluation_cache(chosen)
    selected["data"]["images"] = torch.stack([source[i]["images"] for i in chosen])
    torch.save(selected, gallery / "cache.pt")
    torch.save(
        dict(pred=torch.stack([predictions[i] for i in chosen])),
        gallery / f"prediction_{args.step:06d}.pt",
    )
    rows = [dict(index=i, **metrics[i]) for i in chosen]
    (gallery / "selection.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
