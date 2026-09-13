"""Visible trunk/leaves precision and recall for saved 1024-window predictions."""

# ruff: noqa: E402
import argparse
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.train_multiview_persist_full import Cache
from plot.models.visible_supervision import visible_voxel_masks


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(4)
    indices = json.loads((a.output / "manifest.json").read_text())["indices"]
    step = json.loads((a.output / "completion.json").read_text())["steps"]
    cache = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    mapping = json.loads(
        (ROOT / "outputs/diagnostics/node_geometry/node_geometry_mapping.json").read_text()
    )
    ids = {
        k: torch.tensor(
            [int(n["id"]) for n in mapping["nodes"] if n.get("groups", {}).get(g)], device="cuda"
        )
        for k, g in [("trunk", "tree"), ("leaves", "leaves")]
    }
    rows = []
    for rank in range(8):
        preds = torch.load(a.output / f"rank_{rank}/prediction_{step:06d}.pt", weights_only=False)[
            "pred"
        ]
        for index, pred in zip(indices[rank::8], preds):
            b = {
                k: v.cuda()
                for k, v in cache.evaluation_cache([index])["data"].items()
                if k not in ["latent", "image_rays"]
            }
            gt = b["raw"]
            pred = pred[None].cuda()
            surface, free = visible_voxel_masks(
                gt,
                gt != 127,
                b["camera_position"],
                b["camera_direction"],
                b["agent_mask"] & b["camera_valid"],
                b["fov_x"],
                b["fov_y"],
                126,
            )
            row = dict(index=index)
            for k, v in ids.items():
                target = torch.isin(gt, v) & surface
                predicted = torch.isin(pred, v) & (surface | free)
                row[k] = dict(
                    gt=int(target.sum()),
                    pred=int(predicted.sum()),
                    correct=int(((pred == gt) & target).sum()),
                )
            rows.append(row)
    assert sorted(r["index"] for r in rows) == sorted(indices)
    mean = {}
    for k in ids:
        has = [r[k] for r in rows if r[k]["gt"]]
        mean[k] = dict(
            windows=len(has),
            macro_recall=sum(r["correct"] / r["gt"] for r in has) / len(has),
            micro_precision=sum(r[k]["correct"] for r in rows)
            / max(1, sum(r[k]["pred"] for r in rows)),
        )
    (a.output / "visible_tree_metrics.json").write_text(
        json.dumps(dict(mean=mean, samples=rows), indent=2)
    )
    print(mean)


if __name__ == "__main__":
    main()
