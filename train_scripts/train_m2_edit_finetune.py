"""Fine-tune only the M2 ordered edit path from an existing checkpoint."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader

from plot.data.transition_dataset import collate_transition
from plot.data.transition_stream import TransitionStream
from plot.kinematics import KinematicsConfig
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.training.transition_runner import evaluate, move
from plot.training.transition_trainer import transition_loss


HEAD_PREFIXES = (
    "event_", "pointer_query.", "pointer_key.", "occurrence_head.",
    "edit_kind_head.", "edit_age_head.", "edit_progress_head.",
    "target_type.", "null_token", "payload.", "block_head.",
)
TRUNK_PREFIXES = (
    "state.", "visual.", "held_condition.", "action.", "proposal.",
    "time_embedding", "blocks.", "item_embedding.", "type_embedding.",
)


def edit_score(metrics):
    correct = metrics.get("tolerant_correct_edits", 0)
    predicted = metrics.get("tolerant_predicted_events", 0)
    labels = metrics.get("tolerant_block_labels", 0)
    precision = correct / max(1, predicted)
    recall = correct / max(1, labels)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("heads", "trunk"), default="heads")
    parser.add_argument("--event-context-frames", type=int, default=0)
    parser.add_argument("--edit-oversample", type=int, default=1)
    parser.add_argument("--event-count-positive-weight", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.event_count_positive_weight <= 0:
        parser.error("--event-count-positive-weight must be positive")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("output directory is not empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    model_cfg = dict(checkpoint["config"]["model"])
    model_cfg["kinematics"] = KinematicsConfig(**model_cfg["kinematics"])
    if args.event_context_frames:
        model_cfg["event_context_frames"] = args.event_context_frames
    model = TransitionNetwork(TransitionArgs(**model_cfg))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    allowed_missing = ({name for name in missing if name.startswith("event_context_")}
                       if args.event_context_frames else set())
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    prefixes = HEAD_PREFIXES + (TRUNK_PREFIXES if args.scope == "trunk" else ())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable_names = []
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
            trainable_names.append(name)
    model.to(args.device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    vocabulary_path = args.output_dir / "block_vocabulary.json"
    vocabulary_path.write_text(json.dumps({"class_to_raw": checkpoint["config"]["class_to_raw"]}))
    stream_kwargs = dict(
        index=args.index, vocabulary=vocabulary_path, cache_root=args.cache_root,
        seed=args.seed, items=checkpoint["config"]["items"])
    train = TransitionStream(split="train", edit_oversample=args.edit_oversample,
                             **stream_kwargs)
    val = TransitionStream(split="val_id", fixed_order=True, **stream_kwargs)
    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.workers,
        collate_fn=collate_transition, pin_memory=True,
        persistent_workers=args.workers > 0)
    if args.workers:
        loader_kwargs["prefetch_factor"] = 1
    train_loader = DataLoader(train, **loader_kwargs)
    val_loader = DataLoader(val, **loader_kwargs)

    config = {
        "schema": "m2-edit-finetune-v1", "training": vars(args),
        "model": asdict(model.cfg), "items": checkpoint["config"]["items"],
        "class_to_raw": checkpoint["config"]["class_to_raw"],
        "source_checkpoint": str(args.resume), "objective": "no_hp",
        "selection": "val_id edit_f1_tolerance1",
        "trainable_parameters": sum(p.numel() for p in trainable),
        "trainable_names": trainable_names,
        "frozen_player_attack_and_shared_geometry": True,
    }
    config["training"] = {
        k: str(v) if isinstance(v, Path) else v for k, v in config["training"].items()}
    config["model"]["kinematics"] = asdict(model.cfg.kinematics)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))

    def log(step, split, metrics):
        precision, recall, f1 = edit_score(metrics)
        row = {"step": step, "split": split, **metrics,
               "edit_precision_tolerance1_aggregate": precision,
               "edit_recall_tolerance1_aggregate": recall,
               "edit_f1_tolerance1_aggregate": f1}
        with (args.output_dir / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return f1

    initial, _ = evaluate(model, val_loader, args.device, max_batches=args.eval_batches,
                          objective="no_hp")
    best_f1 = log(0, "val", initial)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": 0, "config": config, "val_metrics": initial},
               args.output_dir / "best-val-edit-f1.pt")
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader); batch = next(iterator)
        values = move(batch, args.device)
        model.train(); optimizer.zero_grad(set_to_none=True)
        output = model(values["inputs"])
        loss, metrics = transition_loss(
            model, output, **values, objective="no_hp",
            event_count_positive_weight=args.event_count_positive_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite edit loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("non-finite gradient")
        optimizer.step()
        if step % 20 == 0:
            log(step, "train", {**metrics, "gradient_norm": float(norm)})
        if step % args.save_every == 0 or step == args.steps:
            val_metrics, _ = evaluate(model, val_loader, args.device,
                                      max_batches=args.eval_batches, objective="no_hp")
            score = log(step, "val", val_metrics)
            payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "step": step, "config": config, "val_metrics": val_metrics}
            torch.save(payload, args.output_dir / "checkpoint.pt")
            if score > best_f1:
                best_f1 = score
                torch.save(payload, args.output_dir / "best-val-edit-f1.pt")
    (args.output_dir / "COMPLETED.json").write_text(json.dumps({
        "steps": args.steps, "best_val_edit_f1_tolerance1": best_f1}, indent=2))


if __name__ == "__main__":
    main()
