"""Two-node regularized text-conditioned M4 v2 training."""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from plot.data.state_policy_dataset import (
    BalancedUnifiedStatePolicyDataset, CachedStatePolicyDataset,
    RoleStatePolicyDataset, collate_state_policy,
)
from plot.models.state_policy_text_v2 import (
    IndependentStatePolicyV4, TextStatePolicyV2, TextStatePolicyV2Args,
    UnifiedStatePolicyV4, UnifiedStatePolicyV4Args,
)
from train_scripts.train_state_zombie_v2 import (
    capture_rng, hard_marginal_baseline, init_distributed, move, restore_rng)


COMPONENTS = ("move_keys", "attack", "place", "hotbar", "mouse_x", "mouse_y",
              "mouse_x_gate", "mouse_y_gate")


def binary_result(counts):
    result = {}
    for index, threshold in enumerate((0.1, 0.3, 0.5)):
        tp, fp, fn = counts[index]
        result[str(threshold)] = dict(
            tp=int(tp), fp=int(fp), fn=int(fn),
            precision=float(tp / (tp + fp).clamp_min(1)),
            recall=float(tp / (tp + fn).clamp_min(1)),
            f1=float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    return result


@torch.no_grad()
def evaluate(model, loader, device, world, args):
    model.eval()
    sums = torch.zeros(2 + len(COMPONENTS), device=device, dtype=torch.float64)
    horizon_sums = torch.zeros(len(COMPONENTS), 8, device=device, dtype=torch.float64)
    horizon_valid = torch.zeros(8, device=device, dtype=torch.float64)
    mouse_absolute = torch.zeros(2, 8, device=device, dtype=torch.float64)
    binary_counts = torch.zeros(2, 3, 3, device=device, dtype=torch.float64)
    binary_horizon = torch.zeros(2, 8, 3, device=device, dtype=torch.float64)
    target_stats = torch.zeros(4, device=device, dtype=torch.float64)
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["inputs"])
            loss, parts = model.loss(
                logits, batch["target_actions"], batch["valid_mask"],
                positive_weight=args.positive_weight,
                place_positive_weight=args.place_positive_weight,
                attack_weight=args.attack_weight, place_weight=args.place_weight,
                mouse_smoothing=args.mouse_smoothing,
                mouse_move_positive_weight=args.mouse_move_positive_weight,
                target_address=batch.get("target_address"), target_weight=args.target_weight)
        size = len(batch["target_actions"])
        valid = batch["valid_mask"].double()
        sums[0] += size
        sums[1] += loss.double() * size
        horizon_valid += valid.sum(0)
        for index, name in enumerate(COMPONENTS):
            values = parts[name].double()
            sums[index + 2] += (values * valid).sum() / valid.sum().clamp_min(1) * size
            horizon_sums[index] += (values * valid).sum(0)
        decoded = model.decode(logits)
        for index, action_index in enumerate((21, 22)):
            mouse_absolute[index] += (
                (decoded[..., action_index] - batch["target_actions"][..., action_index]).abs()
                * valid).sum(0)
        for branch, action_index in (("attack", 8), ("place", 9)):
            branch_index = 0 if branch == "attack" else 1
            truth = batch["target_actions"][..., action_index] > 0
            probability = logits[branch].sigmoid()
            for threshold_index, threshold in enumerate((0.1, 0.3, 0.5)):
                predicted = probability >= threshold
                binary_counts[branch_index, threshold_index] += torch.stack((
                    (predicted & truth).sum(), (predicted & ~truth).sum(),
                    (~predicted & truth).sum()))
            predicted = probability >= 0.5
            binary_horizon[branch_index, :, 0] += (predicted & truth).sum(0)
            binary_horizon[branch_index, :, 1] += (predicted & ~truth).sum(0)
            binary_horizon[branch_index, :, 2] += (~predicted & truth).sum(0)
        if "target_address" in batch:
            labels = batch["target_address"].long()
            valid_target = labels >= 0
            mapped = torch.where(
                labels == 2197, labels.new_full((), logits["target"].shape[-1] - 1), labels)
            predicted = logits["target"].argmax(-1)
            target_stats[0] += parts["target"].double() * valid_target.sum()
            target_stats[1] += valid_target.sum()
            target_stats[2] += ((predicted == mapped) & valid_target).sum()
            target_stats[3] += ((labels != 2197) & valid_target).sum()
    if world > 1:
        for value in (sums, horizon_sums, horizon_valid, mouse_absolute,
                      binary_counts, binary_horizon, target_stats):
            dist.all_reduce(value)
    windows = sums[0].clamp_min(1)
    result = dict(
        windows=int(sums[0]), loss=float(sums[1] / windows),
        parts={name: float(sums[index + 2] / windows)
               for index, name in enumerate(COMPONENTS)},
        attack=binary_result(binary_counts[0]), place=binary_result(binary_counts[1]))
    per_horizon = []
    for horizon in range(8):
        denominator = horizon_valid[horizon].clamp_min(1)
        row = dict(
            horizon=horizon + 1,
            parts={name: float(horizon_sums[index, horizon] / denominator)
                   for index, name in enumerate(COMPONENTS)},
            mouse_x_mae=float(mouse_absolute[0, horizon] / denominator),
            mouse_y_mae=float(mouse_absolute[1, horizon] / denominator))
        for branch_index, branch in enumerate(("attack", "place")):
            tp, fp, fn = binary_horizon[branch_index, horizon]
            row[branch + "_05"] = dict(
                precision=float(tp / (tp + fp).clamp_min(1)),
                recall=float(tp / (tp + fn).clamp_min(1)),
                f1=float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
        per_horizon.append(row)
    result["per_horizon"] = per_horizon
    if target_stats[1] > 0:
        result["target"] = dict(
            loss=float(target_stats[0] / target_stats[1]),
            labeled=int(target_stats[1]), accuracy=float(target_stats[2] / target_stats[1]),
            voxel_labels=int(target_stats[3]))
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--zombie-cache", type=Path)
    parser.add_argument("--independent-v4", action="store_true",
                        help="train one full, task-private V4 parameter set")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validate-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=640)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text-hidden-size", type=int, default=768)
    parser.add_argument("--target-pointer", action="store_true")
    parser.add_argument("--target-weight", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--place-positive-weight", type=float, default=4.0)
    parser.add_argument("--move-key-weight", type=float, default=1.0)
    parser.add_argument("--attack-weight", type=float, default=2.0)
    parser.add_argument("--place-weight", type=float, default=2.0)
    parser.add_argument("--mouse-smoothing", type=float, default=0.2)
    parser.add_argument("--mouse-move-positive-weight", type=float, default=2.0)
    parser.add_argument("--horizon-min-weight", type=float, default=0.25)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-step", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--validation-limit", type=int, default=0,
                        help="limit total validation windows for pipeline checks only")
    args = parser.parse_args()
    rank, world, device = init_distributed()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    text_summary = json.loads((args.cache / "summary.json").read_text())
    supported = ("language_builder", "zombie_melee") if args.independent_v4 else ("language_builder",)
    if text_summary["profile"] not in supported or not text_summary.get("m3_state"):
        raise ValueError("training requires a full supported M3-state cache")
    if not text_summary["selection"]["all_data"]:
        raise ValueError("production requires full eligible data")
    unified = args.zombie_cache is not None
    if unified and args.independent_v4:
        raise ValueError("independent-v4 accepts exactly one task cache")
    zombie_summary = None
    if unified:
        zombie_summary = json.loads((args.zombie_cache / "summary.json").read_text())
        if zombie_summary["profile"] != "zombie_melee" or not zombie_summary.get("m3_state"):
            raise ValueError("v4 requires a full zombie_melee M3-state cache")
        for key in ("num_block_classes", "item_vocabulary"):
            if zombie_summary[key] != text_summary[key]:
                raise ValueError("unified cache mismatch: " + key)
    cfg_class = UnifiedStatePolicyV4Args if unified else TextStatePolicyV2Args
    model_class = (UnifiedStatePolicyV4 if unified else
                   IndependentStatePolicyV4 if args.independent_v4 else TextStatePolicyV2)
    cfg = cfg_class(
        text_summary["num_block_classes"], len(text_summary["item_vocabulary"]),
        profile=("unified" if unified else text_summary["profile"]),
        text_hidden_size=text_summary.get("text_hidden_size", args.text_hidden_size), hidden=args.hidden,
        heads=args.heads, depth=args.depth, dropout=args.dropout,
        target_pointer=args.target_pointer)
    dataset_summary = ({"language_builder": text_summary, "zombie_melee": zombie_summary}
                       if unified else text_summary)
    raw = model_class(cfg).to(device)
    if args.initialize and not args.resume:
        initialized = torch.load(args.initialize, map_location="cpu", weights_only=True)
        incompatible = raw.load_state_dict(initialized["model"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError("unexpected initialization tensors: "
                             + ", ".join(incompatible.unexpected_keys))
        allowed = ("policy_role.", "role_fusion.", "target_", "ego_memory_norm.")
        def allowed_new_tensor(key):
            return (key.startswith(allowed)
                    or ".ego_norm." in key or ".ego_attn." in key
                    or ".target_norm." in key or ".target_attn." in key)
        if any(not allowed_new_tensor(key) for key in incompatible.missing_keys):
            raise ValueError("incompatible initialization tensors: "
                             + ", ".join(incompatible.missing_keys))
        if rank == 0:
            print(json.dumps(dict(initialized_from=str(args.initialize),
                                  new_tensors=incompatible.missing_keys)), flush=True)
    optimizer = torch.optim.AdamW(raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    best = float("inf")
    bad_validations = 0
    saved = None
    if args.resume:
        saved = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=True)
        if saved["config"] != asdict(cfg) or saved["world_size"] != world:
            raise ValueError("resume topology/config mismatch")
        raw.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        best = saved["best_val_loss"]
        bad_validations = saved.get("bad_validations", 0)
    model = DDP(raw, device_ids=[device.index], broadcast_buffers=False) if world > 1 else raw
    if unified:
        train = BalancedUnifiedStatePolicyDataset(
            args.zombie_cache / "train.jsonl", args.cache / "train.jsonl")
        template = train.text.text_template
        text_val = RoleStatePolicyDataset(
            CachedStatePolicyDataset(args.cache / "val_id.jsonl", "language_builder"),
            BalancedUnifiedStatePolicyDataset.BUILDER_ROLE, template)
        zombie_val = RoleStatePolicyDataset(
            CachedStatePolicyDataset(args.zombie_cache / "val_id.jsonl", "zombie_melee"),
            BalancedUnifiedStatePolicyDataset.ZOMBIE_ROLE, template)
        val_datasets = {"language_builder": text_val, "zombie_melee": zombie_val}
    else:
        train = CachedStatePolicyDataset(args.cache / "train.jsonl", cfg.profile)
        val_datasets = {cfg.profile: CachedStatePolicyDataset(
            args.cache / "val_id.jsonl", cfg.profile)}
    sampler = DistributedSampler(train, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        drop_last=True, pin_memory=True, collate_fn=collate_state_policy)
    val_loaders = {}
    for name, val in val_datasets.items():
        val_stop = min(len(val), args.validation_limit) if args.validation_limit else len(val)
        val_loaders[name] = DataLoader(
            Subset(val, list(range(rank, val_stop, world))), batch_size=args.batch_size,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_state_policy)
    if rank == 0:
        if not args.resume:
            args.output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[1]
        sources = ["plot/models/state_policy_text_v2.py", "plot/models/state_policy_large_v2.py",
                   "plot/models/state_policy_large.py", "plot/models/structured_action.py",
                   "plot/data/state_policy_dataset.py", "train_scripts/train_state_text_v2.py"]
        if not args.resume:
            for path in sources:
                destination = args.output / "source" / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / path, destination)
            config = dict(
                model=asdict(cfg), training=vars(args), dataset=dataset_summary, world_size=world,
                global_batch=args.batch_size * world,
                parameters=sum(parameter.numel() for parameter in raw.parameters()),
                source_sha256={path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                               for path in sources})
            config["training"] = {key: str(value) if isinstance(value, Path) else value
                                  for key, value in config["training"].items()}
            (args.output / "config.json").write_text(json.dumps(config, indent=2))
            if not args.skip_baseline:
                (args.output / "baseline.json").write_text(
                    json.dumps(hard_marginal_baseline(args.cache), indent=2))
            print(json.dumps(dict(configured=config["parameters"], world_size=world,
                                  global_batch=config["global_batch"])), flush=True)
    if world > 1:
        dist.barrier()

    def save(step, improved=False):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, capture_rng())
        else:
            states = [capture_rng()]
        if rank == 0:
            payload = dict(model=raw.state_dict(), optimizer=optimizer.state_dict(),
                           config=asdict(cfg), dataset=dataset_summary, step=step,
                           best_val_loss=best, bad_validations=bad_validations,
                           rng=states, world_size=world, batch_size=args.batch_size)
            temporary = args.output / "latest.tmp"
            torch.save(payload, temporary)
            temporary.replace(args.output / "latest.pt")
            if improved:
                shutil.copy2(args.output / "latest.pt", args.output / "best-val-loss.tmp")
                (args.output / "best-val-loss.tmp").replace(args.output / "best-val-loss.pt")
            if step and step % 10000 == 0:
                shutil.copy2(args.output / "latest.pt", args.output / f"step-{step:06d}.pt")
        if world > 1:
            dist.barrier()

    def validate(step):
        nonlocal best, bad_validations
        role_results = {name: evaluate(raw, loader, device, world, args)
                        for name, loader in val_loaders.items()}
        if unified:
            result = dict(
                loss=sum(value["loss"] for value in role_results.values()) / len(role_results),
                roles=role_results,
                selection="equal mean of per-role validation losses")
        else:
            result = role_results[cfg.profile]
        improved = result["loss"] < best
        best = min(best, result["loss"])
        bad_validations = 0 if improved else bad_validations + int(step >= args.early_stop_min_step)
        stop = (args.early_stop_patience > 0 and step >= args.early_stop_min_step
                and bad_validations >= args.early_stop_patience)
        if rank == 0:
            record = dict(step=step, val=result, improved=improved,
                          bad_validations=bad_validations, early_stop=stop)
            with (args.output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        save(step, improved)
        return stop

    epoch = start // len(loader)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(start % len(loader)):
        next(iterator)
    if saved is not None:
        restore_rng(saved["rng"][rank])
    horizon_weights = torch.linspace(1.0, args.horizon_min_weight, 8, device=device)
    started = time.monotonic()
    final_step = start
    early_stopped = False
    for step in range(start + 1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        warm = min(1.0, step / max(args.warmup, 1))
        progress = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        lr = args.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["inputs"])
            loss, parts = raw.loss(
                logits, batch["target_actions"], batch["valid_mask"],
                positive_weight=args.positive_weight,
                place_positive_weight=args.place_positive_weight,
                move_key_weight=args.move_key_weight, attack_weight=args.attack_weight,
                place_weight=args.place_weight, mouse_smoothing=args.mouse_smoothing,
                mouse_move_positive_weight=args.mouse_move_positive_weight,
                horizon_weights=horizon_weights,
                target_address=batch.get("target_address"), target_weight=args.target_weight)
        finite = torch.isfinite(loss).int()
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("nonfinite training loss")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final_step = step
        if step == start + 1 or step % 20 == 0:
            metrics = torch.stack((loss.detach(), *[
                parts[name].detach().mean() for name in COMPONENTS])).float()
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            if rank == 0:
                row = dict(step=step, epoch=epoch, loss=float(metrics[0]),
                           parts={name: float(metrics[index + 1])
                                  for index, name in enumerate(COMPONENTS)},
                           grad_norm=float(grad), lr=lr,
                           elapsed_seconds=time.monotonic() - started,
                           gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        if step == 1000 or step % args.validate_every == 0 or step == args.steps:
            if validate(step):
                early_stopped = True
                break
        elif step % args.checkpoint_every == 0:
            save(step)
    if rank == 0:
        (args.output / "COMPLETE.json").write_text(json.dumps(dict(
            planned_steps=args.steps, final_step=final_step,
            early_stopped=early_stopped, best_val_loss=best, profile=cfg.profile)))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
