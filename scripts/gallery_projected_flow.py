"""Identical evenly spaced scene gallery for projection ablation."""

# ruff: noqa: E402
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.train_multiview_persist_full import Cache, Source


def main():
    cache = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    source = Source(
        SimpleNamespace(
            dataset_root=Path(cache.meta["dataset_root"]), persist=ROOT.parent / "PERSIST"
        )
    )
    c = None
    for name in ["m1_projected_flow1024_v1", "m1_baseline_flow1024_v1"]:
        p = ROOT / "outputs" / name
        step = json.loads((p / "completion.json").read_text())["steps"]
        indices = json.loads((p / "manifest.json").read_text())["indices"]
        positions = np.linspace(0, len(indices) - 1, 8).round().astype(int).tolist()
        chosen = [indices[i] for i in positions]
        if c is None:
            c = cache.evaluation_cache(chosen)
            c["data"]["images"] = torch.stack([source[i]["images"] for i in chosen])
        pred = {}
        for rank in set(i % 8 for i in positions):
            pred[rank] = torch.load(
                p / f"rank_{rank}/prediction_{step:06d}.pt", weights_only=False
            )["pred"]
        gallery = p / "gallery"
        gallery.mkdir(exist_ok=True)
        torch.save(c, gallery / "cache.pt")
        torch.save(
            dict(pred=torch.stack([pred[i % 8][i // 8] for i in positions])),
            gallery / f"prediction_{step:06d}.pt",
        )
        (gallery / "indices.json").write_text(json.dumps(chosen))


if __name__ == "__main__":
    main()
