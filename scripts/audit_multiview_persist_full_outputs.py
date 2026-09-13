"""Audit saved full predictions, tree recall, and unknown engine cells."""

# ruff: noqa: E402
import json
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.train_multiview_persist_full import Cache
from plot.models.visible_supervision import visible_voxel_masks
from plot.models.projection import raycast_voxel_targets


@torch.no_grad()
def rescore(pred, b, old):
    pred = pred[None].cuda()
    gt = b["raw"].cuda()
    valid = gt != 127
    views = (b["agent_mask"] & b["camera_valid"]).cuda()
    poses = [b[k].cuda() for k in ["camera_position", "camera_direction", "fov_x", "fov_y"]]
    surface, free = visible_voxel_masks(
        gt, valid, poses[0], poses[1], views, poses[2], poses[3], 126
    )
    correct = (pred == gt) & surface
    visible = (pred != 126) & (surface | free)
    r = {
        **old,
        "surface_exact_recall": float(correct.sum() / surface.sum().clamp_min(1)),
        "visible_exact_precision": float(correct.sum() / visible.sum().clamp_min(1)),
        "free_accuracy": float(((pred == 126) & free).sum() / free.sum().clamp_min(1)),
    }
    hits = count = 0
    for v in range(2):
        if not views[0, v]:
            continue
        cam = [p[:, v] for p in poses]
        kw = dict(height=60, width=106, samples=192, max_distance=32.0)
        ph, pd, _ = raycast_voxel_targets((pred != 126).float(), *cam, **kw)
        gh, gd, _ = raycast_voxel_targets(((gt != 126) & valid).float(), *cam, **kw)
        uh, ud, _ = raycast_voxel_targets((~valid).float(), *cam, **kw)
        active = gh.bool() & ~(uh.bool() & (ud <= gd))
        active[:, round(60 * 0.82) :] = False
        hits += int((active & ph.bool() & ((pd - gd).abs() <= 0.5)).sum())
        count += int(active.sum())
    r["gt_camera_half_block_hit"] = hits / max(count, 1)
    return r


def main():
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/m1_multiview_persist_gtcam_s01_all_v1"
    )
    out = parser.parse_args().output
    c = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    result = json.loads((out / "full_fit_metrics.json").read_text())
    original_mean = result["mean"].copy()
    mapping = json.loads(
        (ROOT / "outputs/diagnostics/node_geometry/node_geometry_mapping.json").read_text()
    )
    lookup = np.zeros(max(c.meta["raw_lut"]) + 1, dtype=bool)
    for node in mapping["nodes"]:
        i = int(node["id"])
        groups = node.get("groups", {})
        if i < len(lookup) and (groups.get("tree") or groups.get("leaves")):
            lookup[i] = True
    tree_rows = []
    unknown = []
    seen = []
    for path in sorted(out.glob("full_rank_*/prediction_*.pt")):
        start = int(path.stem.split("_")[-1])
        pred = torch.load(path, weights_only=False)["pred"]
        gt = np.array(c.arrays["raw"][start : start + len(pred)], copy=True)
        for local, pp in enumerate(pred):
            idx = start + local
            seen.append(idx)
            if (gt[local] == 127).any():
                before = result["samples"][idx].copy()
                after = rescore(pp, c.evaluation_cache([idx])["data"], before)
                result["samples"][idx] = after
                unknown.append(
                    dict(
                        index=idx,
                        unknown_voxels=int((gt[local] == 127).sum()),
                        before=before,
                        after=after,
                    )
                )
            gtree = lookup[gt[local]]
            ptree = lookup[pp.numpy()]
            correct = int(((pp.numpy() == gt[local]) & gtree).sum())
            tree_rows.append(
                dict(index=idx, gt=int(gtree.sum()), pred=int(ptree.sum()), correct=correct)
            )
    assert sorted(seen) == list(range(len(c)))
    result["mean"] = {k: sum(r[k] for r in result["samples"]) / len(c) for k in original_mean}
    result["unknown_cell_audit"] = dict(
        windows=len(unknown),
        voxels=sum(r["unknown_voxels"] for r in unknown),
        original_mean=original_mean,
        policy="raw ID 127 denotes unknown engine cells: exclude them and GT rays occluded by unknown cells from scoring; training cache retained these labels in this run.",
    )
    known_tree = [r for r in tree_rows if r["gt"]]
    tree = dict(
        windows_with_tree=len(known_tree),
        macro_recall=sum(r["correct"] / r["gt"] for r in known_tree) / len(known_tree),
        micro_recall=sum(r["correct"] for r in tree_rows) / sum(r["gt"] for r in tree_rows),
        micro_precision=sum(r["correct"] for r in tree_rows)
        / max(sum(r["pred"] for r in tree_rows), 1),
        samples=tree_rows,
    )
    (out / "tree_full_metrics.json").write_text(json.dumps(tree, indent=2))
    (out / "unknown_cell_audit.json").write_text(json.dumps(unknown, indent=2))
    (out / "full_fit_metrics.json").write_text(json.dumps(result, indent=2))
    (out / "completion.json").write_text(
        json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2)
    )
    print(
        json.dumps(
            dict(
                mean=result["mean"],
                tree_macro_recall=tree["macro_recall"],
                unknown_windows=len(unknown),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
