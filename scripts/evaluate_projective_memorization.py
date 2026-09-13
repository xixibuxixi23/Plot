"""Exhaustive fitted-scene evaluation; splits are taken from training manifest."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.data import TextAgentFillDataset
from plot.models import GeometryConditionedFillArgs, ProjectiveFillArgs
from plot.models.projection import raycast_voxel_targets
from scripts.pilot_projective_fill import inputs
from experiments.m1.train_projective_memorization import Indexed, prepare


@torch.no_grad()
def batch_metrics(pred, b, air):
    surface, free = b["surface"], b["free"]
    occupied = pred != air
    correct = (pred == b["target"]) & surface
    def ratio(numerator, denominator):
        return numerator.flatten(1).sum(1).float() / denominator.flatten(1).sum(1).clamp_min(1)
    values = {
        "surface_exact_recall": ratio(correct, surface),
        "visible_exact_precision": ratio(correct, occupied & (surface | free)),
        "surface_occupancy_recall": ratio(occupied & surface, surface),
        "free_accuracy": ratio(~occupied & free, free),
    }
    target_volume = ((b["target"] != air) & b["target_valid"]).float()
    pred_volume = (occupied & b["target_valid"]).float()
    gt = []
    kw = dict(height=60, width=106, samples=192, max_distance=32.)
    for v in range(b["images"].shape[1]):
        gh, gd, _ = raycast_voxel_targets(target_volume, b["gt_position"][:, v], b["gt_direction"][:, v],
            b["fov_x"][:, v], b["fov_y"][:, v], **kw)
        gt.append((gh, gd))
    for prefix, pos, direction in (("gt_pose", b["gt_position"], b["gt_direction"]),
                                    ("pred_pose", b["camera_position"], b["camera_direction"])):
        counts = torch.zeros(pred.shape[0], device=pred.device)
        totals = {k: counts.clone() for k in ("half_block_hit", "hit_or_missing_mae", "early_hit", "late_hit", "missing_hit")}
        for v, (gh, gd) in enumerate(gt):
            ph, pd, _ = raycast_voxel_targets(pred_volume, pos[:, v], direction[:, v],
                b["fov_x"][:, v], b["fov_y"][:, v], **kw)
            active = gh.bool().clone()
            active[:, round(60 * .82):] = False
            active &= (b["camera_valid"] & b["agent_mask"])[:, v, None, None]
            delta = torch.where(ph.bool(), pd, 32.) - gd
            observations = {
                "half_block_hit": (delta.abs() <= .5) & ph.bool(),
                "hit_or_missing_mae": delta.abs(),
                "early_hit": (delta < -.5) & ph.bool(),
                "late_hit": (delta > .5) & ph.bool(),
                "missing_hit": ~ph.bool(),
            }
            counts += active.flatten(1).sum(1)
            for key, value in observations.items():
                totals[key] += (value * active).flatten(1).sum(1)
        values.update({prefix + "/" + k: total / counts.clamp_min(1) for k, total in totals.items()})
    cpu_values = {k: v.cpu().tolist() for k, v in values.items()}
    return [{"index": index, **{k: v[i] for k, v in cpu_values.items()}}
            for i, index in enumerate(b["sample_index"].tolist())]


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="0 evaluates every fitted sample")
    p.add_argument("--oracle-volume", action="store_true",
                   help="Diagnostic only: substitute GT blocks to isolate predicted camera error")
    a = p.parse_args()
    world, rank, local = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0")), int(os.getenv("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group("nccl")
    c = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    m = c["manifest"]
    sets = [TextAgentFillDataset(m["dataset_root"], m["vocabulary"], split=s,
        samples_per_agent=1, max_agents=2, num_views=2, initial_only=True,
        episode_index=m["episode_index"], canonical_yaw=True) for s in m["splits"]]
    dataset = Indexed(ConcatDataset(sets))
    n = min(a.limit, len(dataset)) if a.limit else len(dataset)
    loader = DataLoader(Subset(dataset, range(rank, n, world)), batch_size=a.batch_size,
                        num_workers=2, pin_memory=True)
    pc = torch.load(m["pose_checkpoint"], map_location="cpu", weights_only=False)
    pose = GeometryConditionedFillArgs(**pc["model_args"]).build().cuda().eval()
    pose.load_state_dict(pc["model"])
    model = ProjectiveFillArgs(**c["model_args"]).build().cuda().eval()
    model.load_state_dict(c["model"])
    air = pc["model_args"]["air_class"]
    rows = []
    for step, cpu in enumerate(loader):
        b = prepare({k: v.cuda(non_blocking=True) for k, v in cpu.items()}, pose, air, {})
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = b["target"] if a.oracle_volume else model.predict(**inputs(b))
        rows.extend(batch_metrics(pred, b, air))
        if rank == 0 and step % 50 == 0:
            print(json.dumps({"rank0_samples_evaluated": len(rows), "total_population": n}), flush=True)
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, rows)
        rows = [r for part in gathered for r in part]
    if rank == 0:
        assert len(rows) == n and len({r["index"] for r in rows}) == n
        avg = {k: sum(r[k] for r in rows) / n for k in rows[0] if k != "index"}
        by_split = {}
        offset = 0
        for split, dataset_part in zip(m["splits"], sets):
            subset_rows = [r for r in rows if offset <= r["index"] < offset + len(dataset_part)]
            if subset_rows:
                by_split[split] = {"count": len(subset_rows), "scene_macro_metrics": {
                    k: sum(r[k] for r in subset_rows) / len(subset_rows) for k in avg}}
            offset += len(dataset_part)
        result = {"checkpoint": str(a.checkpoint), "step": c["step"], "count": n,
            "oracle_volume_diagnostic": a.oracle_volume,
            "entire_population": n == len(dataset), "protocol": "fitted scenes, NOT held-out",
            "scene_macro_metrics": avg, "original_split_fitting_metrics": by_split,
            "fraction_scenes_passing_90_recall_precision_and_projection": sum(
                r["surface_exact_recall"] >= .9 and r["visible_exact_precision"] >= .9 and
                r["pred_pose/half_block_hit"] >= .9 for r in rows) / n,
            "samples": sorted(rows, key=lambda r: r["index"])}
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
