"""Train fixed-query M1; frozen V8 supplies predicted cameras only.

No target pose or depth is fed to the new model. Small train subsets measure
memorization; validation is reported separately and never used for training.
"""
import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.data import TextAgentFillDataset
from plot.models import GeometryConditionedFillArgs, decode_voxel_prediction
from plot.models.projective_fill import ProjectiveFillArgs
from plot.models.projection import raycast_voxel_targets
from plot.models.visible_supervision import visible_voxel_masks


def inputs(batch):
    return {k: batch[k] for k in ("voxel_context", "known_mask", "fill_mask", "images", "agent_mask",
                                  "camera_position", "camera_direction", "fov_x", "fov_y")}


@torch.no_grad()
def metrics(pred, batch, air):
    surface, free = batch["surface"], batch["free"]
    occupied = pred != air
    correct = (pred == batch["target"]) & surface
    visible_pred = occupied & (surface | free)
    result = {
        "surface_exact_recall": float(correct.sum() / surface.sum().clamp_min(1)),
        "visible_exact_precision": float(correct.sum() / visible_pred.sum().clamp_min(1)),
        "surface_occupancy_recall": float((occupied & surface).sum() / surface.sum().clamp_min(1)),
        "free_accuracy": float((~occupied & free).sum() / free.sum().clamp_min(1)),
    }
    volume = (occupied & batch["target_valid"]).float()
    for prefix, pos, direction in (("gt_pose", batch["gt_position"], batch["gt_direction"]),
                                    ("pred_pose", batch["camera_position"], batch["camera_direction"])):
        hits = errors = count = 0
        for view in range(batch["images"].shape[1]):
            ph, pd, _ = raycast_voxel_targets(volume, pos[:, view], direction[:, view],
                batch["fov_x"][:, view], batch["fov_y"][:, view],
                height=60, width=106, samples=192, max_distance=32.)
            gh, gd = batch["ray_gt"][view]
            active = gh.bool().clone()
            active[:, round(60 * .82):] = False
            active &= (batch["camera_valid"] & batch["agent_mask"])[:, view, None, None]
            delta = (torch.where(ph.bool(), pd, 32.) - gd).abs()
            hits += int(((delta <= .5) & ph.bool() & active).sum())
            errors += float(delta[active].sum())
            count += int(active.sum())
        result[prefix + "/half_block_hit"] = hits / max(count, 1)
        result[prefix + "/hit_or_missing_mae"] = errors / max(count, 1)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default="../textagent/data/batches/s01_v1_mask_20260907")
    p.add_argument("--vocabulary", default="datasets/s01_block_vocabulary.json")
    p.add_argument("--episode-index", default="datasets/s01_episode_index.json")
    p.add_argument("--pose-checkpoint", default="outputs/m1_direct_visible_v8/checkpoint_00001500.pt")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--train-samples", type=int, default=8)
    p.add_argument("--stream-train", action="store_true",
                   help="Sample all training data online; train-samples then controls only the audit subset")
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--unseen-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260910)
    args = p.parse_args()
    if (args.output_dir / "manifest.json").exists():
        raise FileExistsError("Choose a new output directory to preserve existing experiments")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    checkpoint = torch.load(args.pose_checkpoint, map_location="cpu", weights_only=False)
    old = GeometryConditionedFillArgs(**checkpoint["model_args"]).build().cuda().eval()
    old.load_state_dict(checkpoint["model"])
    old.requires_grad_(False)
    datasets, indices, baseline = {}, {}, {}
    air = checkpoint["model_args"]["air_class"]
    for split, n in (("train", args.train_samples), ("val_id", args.val_samples)):
        ds = TextAgentFillDataset(args.dataset_root, args.vocabulary, split=split,
            samples_per_agent=1, max_agents=2, num_views=2, initial_only=True,
            episode_index=args.episode_index, canonical_yaw=True)
        if split == "train":
            train_dataset = ds
        if n < 1 or n > len(ds):
            raise ValueError("sample count out of range")
        indices[split] = torch.linspace(0, len(ds) - 1, n).round().long().tolist()
        datasets[split], baseline[split] = [], []
        for i in indices[split]:
            batch = {k: v[None].cuda() for k, v in ds[i].items()}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                output = old(batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
                             batch["images"], batch["agent_mask"], return_aux=True)
                pred = decode_voxel_prediction(output, air)
            batch["gt_position"] = batch["camera_position"].clone()
            batch["gt_direction"] = batch["camera_direction"].clone()
            batch["surface"], batch["free"] = visible_voxel_masks(batch["target"], batch["target_valid"],
                batch["gt_position"], batch["gt_direction"], batch["camera_valid"] & batch["agent_mask"],
                batch["fov_x"], batch["fov_y"], air)
            batch["camera_position"] = output["camera_position"].float().detach()
            batch["camera_direction"] = output["camera_direction"].float().detach()
            batch["ray_gt"] = []
            for view in range(2):
                gh, gd, _ = raycast_voxel_targets(((batch["target"] != air) & batch["target_valid"]).float(),
                    batch["gt_position"][:, view], batch["gt_direction"][:, view],
                    batch["fov_x"][:, view], batch["fov_y"][:, view],
                    height=60, width=106, samples=192, max_distance=32.)
                batch["ray_gt"].append((gh, gd))
            baseline[split].append(metrics(pred, batch, air))
            datasets[split].append(batch)
        print(json.dumps({"cached": split, "samples": n}), flush=True)
    del output, checkpoint
    if not args.stream_train:
        del old
    torch.cuda.empty_cache()
    model_args = ProjectiveFillArgs(num_block_classes=ds.vocabulary.size)
    model = model_args.build().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    starting_step = 0
    if args.resume:
        previous = torch.load(args.resume, map_location="cpu", weights_only=False)
        if previous["model_args"] != asdict(model_args) or previous["manifest"]["indices"] != indices:
            raise ValueError("Resume requires identical model and sample indices")
        if previous["manifest"]["pose_checkpoint"] != args.pose_checkpoint:
            raise ValueError("Resume requires the same camera provider")
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
        starting_step = previous["step"]
    manifest = {**vars(args), "resume": str(args.resume) if args.resume else None,
        "starting_step": starting_step, "output_dir": str(args.output_dir), "indices": indices,
        "training_population": len(train_dataset) if args.stream_train else args.train_samples,
        "model_args": asdict(model_args), "pose_source": "frozen V8 predictions; dataset FOV",
        "loss": "surface CE + free CE + unseen_weight * unseen CE; region means",
        "instance_masks": False, "parameter_count": sum(t.numel() for t in model.parameters())}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output_dir / "baseline.json").write_text(json.dumps(baseline, indent=2))
    history = []
    for step in range(1, args.steps + 1):
        model.train()
        batch = random.choice(datasets["train"])
        if args.stream_train:
            batch = {k: v[None].cuda() for k, v in train_dataset[random.randrange(len(train_dataset))].items()}
            batch["surface"], batch["free"] = visible_voxel_masks(batch["target"], batch["target_valid"],
                batch["camera_position"], batch["camera_direction"], batch["camera_valid"] & batch["agent_mask"],
                batch["fov_x"], batch["fov_y"], air)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                geometry = old.geometry(batch["images"], batch["agent_mask"])
                position, direction = old._anchor_pose(geometry)
                position, direction = old._camera_poses(geometry, position, direction)
            batch["camera_position"], batch["camera_direction"] = position.float(), direction.float()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs(batch))
            error = F.cross_entropy(logits.float(), batch["target"], reduction="none")
            writable = batch["target_valid"] & batch["fill_mask"] & ~batch["known_mask"]
            surface, free = batch["surface"] & writable, batch["free"] & writable
            unseen = writable & ~surface & ~free
            loss = sum(weight * (error * mask).sum() / mask.sum().clamp_min(1)
                       for mask, weight in ((surface, 1.), (free, 1.), (unseen, args.unseen_weight)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step % 25 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            record = {"step": starting_step + step}
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for split, samples in datasets.items():
                    rows = [metrics(model.predict(**inputs(b)), b, air) for b in samples]
                    record[split] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
                    record[split + "_per_sample"] = rows
            history.append(record)
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
            print(json.dumps({k: v for k, v in record.items() if "per_sample" not in k}), flush=True)
    torch.save({"model": model.state_dict(), "model_args": asdict(model_args),
                "optimizer": optimizer.state_dict(), "step": starting_step + args.steps, "manifest": manifest},
               args.output_dir / "checkpoint_final.pt")


if __name__ == "__main__":
    main()
