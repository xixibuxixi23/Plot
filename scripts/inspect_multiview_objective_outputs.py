"""Tree counts and evenly spaced input/depth galleries for the objective experiment."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.train_multiview_persist_full import Cache, Source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["direct", "flow"], required=True)
    args = parser.parse_args()
    out = ROOT / f"outputs/m1_objective_{args.mode}256_v1"
    manifest = json.loads((out / "manifest.json").read_text())
    step = json.loads((out / "completion.json").read_text())["steps"]
    indices = manifest["indices"]
    cache = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    predictions = {}
    for rank in range(8):
        pred = torch.load(out / f"rank_{rank}/prediction_{step:06d}.pt", weights_only=False)["pred"]
        predictions.update(zip(indices[rank::8], pred))
    assert sorted(predictions) == sorted(indices)
    mapping = json.loads(
        (ROOT / "outputs/diagnostics/node_geometry/node_geometry_mapping.json").read_text()
    )
    tree_ids = torch.tensor(
        [
            int(n["id"])
            for n in mapping["nodes"]
            if n.get("groups", {}).get("tree") or n.get("groups", {}).get("leaves")
        ]
    )
    rows = []
    for i in indices:
        gt = torch.from_numpy(np.array(cache.arrays["raw"][i], copy=True))
        mask = torch.isin(gt, tree_ids)
        rows.append(
            dict(index=i, gt=int(mask.sum()), correct=int(((predictions[i] == gt) & mask).sum()))
        )
    included = [r for r in rows if r["gt"]]
    result = dict(
        scope="all tree/log/leaves voxels in target volume, including hidden; not visibility-filtered",
        count=len(included),
        macro_recall=sum(r["correct"] / r["gt"] for r in included) / len(included),
        samples=rows,
    )
    (out / "tree_metrics.json").write_text(json.dumps(result, indent=2))
    selected = [indices[i] for i in np.linspace(0, 255, 8).round().astype(int)]
    src = Source(
        SimpleNamespace(
            dataset_root=Path(cache.meta["dataset_root"]), persist=ROOT.parent / "PERSIST"
        )
    )
    gallery = out / "gallery"
    gallery.mkdir(exist_ok=True)
    c = cache.evaluation_cache(selected)
    c["data"]["images"] = torch.stack([src[i]["images"] for i in selected])
    torch.save(c, gallery / "cache.pt")
    torch.save(
        dict(pred=torch.stack([predictions[i] for i in selected])),
        gallery / f"prediction_{step:06d}.pt",
    )
    (gallery / "indices.json").write_text(json.dumps(selected))
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}))


if __name__ == "__main__":
    main()
