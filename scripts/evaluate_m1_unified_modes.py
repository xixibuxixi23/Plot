from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plot.data import TextAgentFillDataset  # noqa: E402
from plot.models import FillNetworkArgs  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unified M1 by conditioning mode")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, required=True)
    parser.add_argument("--episode-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-mode", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def evenly_spaced(indices: list[int], count: int) -> list[int]:
    if len(indices) <= count:
        return indices
    positions = np.linspace(0, len(indices) - 1, count)
    return [indices[int(round(position))] for position in positions]


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def evaluate(model, dataset, indices, *, batch_size, workers, device, air_class, ablate_images=False):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
    )
    classes = model.num_block_classes
    confusion = torch.zeros((classes, classes), dtype=torch.int64)
    totals = {
        "examples": 0,
        "valid_voxels": 0,
        "correct_voxels": 0,
        "cross_entropy_sum": 0.0,
        "occupancy_tp": 0,
        "occupancy_fp": 0,
        "occupancy_fn": 0,
        "target_non_air_voxels": 0,
        "correct_non_air_material_voxels": 0,
        "known_voxels": 0,
        "fill_voxels": 0,
        "valid_target_voxels": 0,
    }
    sample_accuracies = []
    started = time.time()
    for batch in loader:
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        if ablate_images:
            batch["image_condition_mask"] = torch.zeros_like(batch["image_condition_mask"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(
                batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
                batch["images"], batch["agent_mask"], batch["image_condition_mask"],
            )
        target = batch["target"]
        valid = batch["fill_mask"].bool() & batch["target_valid"].bool()
        prediction = logits.argmax(1)
        losses = F.cross_entropy(logits.float(), target, reduction="none")
        valid_count = int(valid.sum())
        correct = prediction.eq(target) & valid
        target_occupied = target.ne(air_class) & valid
        prediction_occupied = prediction.ne(air_class) & valid

        totals["examples"] += int(target.shape[0])
        totals["valid_voxels"] += valid_count
        totals["correct_voxels"] += int(correct.sum())
        totals["cross_entropy_sum"] += float(losses[valid].sum())
        totals["occupancy_tp"] += int((target_occupied & prediction_occupied).sum())
        totals["occupancy_fp"] += int((~target_occupied & prediction_occupied & valid).sum())
        totals["occupancy_fn"] += int((target_occupied & ~prediction_occupied).sum())
        totals["target_non_air_voxels"] += int(target_occupied.sum())
        totals["correct_non_air_material_voxels"] += int((correct & target_occupied).sum())
        totals["known_voxels"] += int(batch["known_mask"].sum())
        totals["fill_voxels"] += int(batch["fill_mask"].sum())
        totals["valid_target_voxels"] += int(batch["target_valid"].sum())

        for row in range(target.shape[0]):
            row_valid = valid[row]
            sample_accuracies.append(float(correct[row].sum() / row_valid.sum().clamp_min(1)))
        flat = target[valid] * classes + prediction[valid]
        confusion += torch.bincount(flat.cpu(), minlength=classes * classes).reshape(classes, classes)

    elapsed = time.time() - started
    eps = 1e-12
    tp, fp, fn = (totals[key] for key in ("occupancy_tp", "occupancy_fp", "occupancy_fn"))
    precision = tp / max(eps, tp + fp)
    recall = tp / max(eps, tp + fn)
    row_support = confusion.sum(1)
    class_recall = confusion.diag().float() / row_support.clamp_min(1)
    class_mask = row_support.gt(0)
    class_mask[air_class] = False
    result = {
        "examples": totals["examples"],
        "valid_fill_voxels": totals["valid_voxels"],
        "mean_cross_entropy": totals["cross_entropy_sum"] / max(1, totals["valid_voxels"]),
        "voxel_accuracy_micro": totals["correct_voxels"] / max(1, totals["valid_voxels"]),
        "voxel_accuracy_per_sample_mean": float(np.mean(sample_accuracies)),
        "voxel_accuracy_per_sample_median": float(np.median(sample_accuracies)),
        "occupancy_precision": precision,
        "occupancy_recall": recall,
        "occupancy_f1": 2 * precision * recall / max(eps, precision + recall),
        "non_air_material_accuracy": totals["correct_non_air_material_voxels"] / max(1, totals["target_non_air_voxels"]),
        "non_air_macro_recall": float(class_recall[class_mask].mean()) if class_mask.any() else None,
        "non_air_classes_with_support": int(class_mask.sum()),
        "target_non_air_voxels": totals["target_non_air_voxels"],
        "mean_known_fraction": totals["known_voxels"] / max(1, totals["examples"] * 48**3),
        "mean_fill_fraction": totals["fill_voxels"] / max(1, totals["examples"] * 48**3),
        "seconds": elapsed,
        "examples_per_second": totals["examples"] / max(eps, elapsed),
    }
    return result


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    model_args = checkpoint.get("model_args") or {
        "num_block_classes": int(state["classifier.weight"].shape[0]),
        "base_channels": int(state["classifier.weight"].shape[1]),
        "max_views": int(state["view_embedding"].shape[0]),
    }
    model = FillNetworkArgs(**model_args).build()
    model.load_state_dict(state)
    model.to(device).eval()

    dataset = TextAgentFillDataset(
        args.dataset_root, args.vocabulary, split="val_id", samples_per_agent=4,
        max_agents=2, num_views=2, initial_only=False, episode_index=args.episode_index,
        frontier_sampling=True, frontier_image_probability=1 / 3,
    )
    try:
        air_class = dataset.vocabulary.class_to_raw.index(126)
    except ValueError as error:
        raise RuntimeError("vocabulary lacks Minetest CONTENT_AIR=126") from error

    candidates = {"initialization_image": [], "frontier_no_image": [], "frontier_image": []}
    for item, (_, _, slot) in enumerate(dataset.index):
        if slot == 0:
            candidates["initialization_image"].append(item)
        elif slot in (1, 2):
            candidates["frontier_no_image"].append(item)
        else:
            candidates["frontier_image"].append(item)
    selected = {
        mode: evenly_spaced(indices, args.samples_per_mode)
        for mode, indices in candidates.items()
    }

    results = {}
    for mode in ("initialization_image", "frontier_no_image", "frontier_image"):
        print(f"evaluating {mode}: {len(selected[mode])} samples", flush=True)
        results[mode] = evaluate(
            model, dataset, selected[mode], batch_size=args.batch_size, workers=args.workers,
            device=device, air_class=air_class,
        )
        print(json.dumps({mode: results[mode]}, sort_keys=True), flush=True)
    print("evaluating frontier_image_ablate_to_no_image on matched samples", flush=True)
    results["frontier_image_ablate_to_no_image"] = evaluate(
        model, dataset, selected["frontier_image"], batch_size=args.batch_size,
        workers=args.workers, device=device, air_class=air_class, ablate_images=True,
    )

    payload = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "protocol": {
            "split": "val_id",
            "selection": "evenly spaced within deterministic dataset mode slots",
            "requested_samples_per_mode": args.samples_per_mode,
            "batch_size": args.batch_size,
            "precision": "bfloat16 autocast",
            "seed": args.seed,
            "frontier_image_ablation": "same frontier_image examples with image_condition_mask forced false",
        },
        "candidate_counts": {mode: len(indices) for mode, indices in candidates.items()},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
