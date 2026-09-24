"""Two-node regularized zombie M4 v2 training with early stopping."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
from datetime import timedelta
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

from plot.data.state_policy_dataset import CachedStatePolicyDataset, collate_state_policy
from plot.models.state_policy_large_v2 import (
    ZombieStatePolicyV2, ZombieStatePolicyV2Args,
    ZombieStatePolicyV3, ZombieStatePolicyV3Args,
)
from plot.models.structured_action import StructuredActionHead


V2_COMPONENTS = ("move_keys", "attack", "hotbar", "mouse_x", "mouse_y",
                 "mouse_x_gate", "mouse_y_gate")
V3_COMPONENTS = V2_COMPONENTS + ("locomotion", "combat_range", "combat_bearing")


def move(batch, device):
    return {key: move(value, device) if isinstance(value, dict)
            else value.to(device, non_blocking=True) for key, value in batch.items()}


def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=15))
    return rank, world, torch.device("cuda", local)


def hard_marginal_baseline(cache):
    """Reference hard-label NLL; training/validation use the v2 soft mouse objective."""
    head = StructuredActionHead(1, 8)
    labels = np.load(cache / "train.actions.npy", mmap_mode="r")
    counts = {
        "keys": torch.zeros(8, 10, dtype=torch.float64),
        "hotbar": torch.zeros(8, 10, dtype=torch.float64),
        "mouse_x": torch.zeros(8, 17, dtype=torch.float64),
        "mouse_y": torch.zeros(8, 17, dtype=torch.float64),
    }
    for offset in range(0, len(labels), 4096):
        targets = head.targets(torch.tensor(np.array(labels[offset:offset + 4096])))
        counts["keys"] += targets["keys"].sum(0)
        for name, size in (("hotbar", 10), ("mouse_x", 17), ("mouse_y", 17)):
            counts[name] += torch.nn.functional.one_hot(targets[name], size).sum(0)
    probability = (counts["keys"] + 0.5) / (len(labels) + 1)
    logits = {"keys": torch.logit(probability).float()}
    for name, size in (("hotbar", 10), ("mouse_x", 17), ("mouse_y", 17)):
        logits[name] = ((counts[name] + 0.5) / (len(labels) + size * 0.5)).log().float()
    val = np.load(cache / "val_id.actions.npy", mmap_mode="r")
    total = 0.0
    for offset in range(0, len(val), 4096):
        actions = torch.tensor(np.array(val[offset:offset + 4096]))
        batch = len(actions)
        loss, _ = head.loss({key: value[None].expand(batch, -1, -1)
                             for key, value in logits.items()}, actions)
        total += float(loss) * batch
    return dict(hard_label_loss=total / len(val), train_windows=len(labels),
                val_windows=len(val), note="not directly comparable to v2 soft/gated validation loss")


@torch.no_grad()
def evaluate(model, loader, device, world, components, loss_kwargs):
    model.eval()
    sums = torch.zeros(2 + len(components), device=device, dtype=torch.float64)
    horizon_sums = torch.zeros(len(components), 8, device=device, dtype=torch.float64)
    horizon_valid = torch.zeros(8, device=device, dtype=torch.float64)
    mouse_absolute = torch.zeros(2, 8, device=device, dtype=torch.float64)
    attack_counts = torch.zeros(3, 3, device=device, dtype=torch.float64)
    attack_horizon = torch.zeros(8, 3, device=device, dtype=torch.float64)
    for batch in loader:
        batch = move(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["inputs"])
            loss, parts = model.loss(
                logits, batch["target_actions"], batch["valid_mask"],
                **loss_kwargs)
        size = len(batch["target_actions"])
        valid = batch["valid_mask"].to(torch.float64)
        sums[0] += size
        sums[1] += loss.double() * size
        horizon_valid += valid.sum(0)
        for index, name in enumerate(components):
            values = parts[name].double()
            sums[index + 2] += (values * valid).sum() / valid.sum().clamp_min(1) * size
            horizon_sums[index] += (values * valid).sum(0)

        decoded = model.decode(logits)
        decoded_x, decoded_y = decoded[..., 21], decoded[..., 22]
        mouse_absolute[0] += ((decoded_x - batch["target_actions"][..., 21]).abs() * valid).sum(0)
        mouse_absolute[1] += ((decoded_y - batch["target_actions"][..., 22]).abs() * valid).sum(0)

        truth = batch["target_actions"][..., 8] > 0
        probability = logits["attack"].sigmoid()
        for threshold_index, threshold in enumerate((0.1, 0.3, 0.5)):
            predicted = probability >= threshold
            attack_counts[threshold_index] += torch.stack((
                (predicted & truth).sum(), (predicted & ~truth).sum(), (~predicted & truth).sum()))
        predicted = probability >= 0.5
        attack_horizon[:, 0] += (predicted & truth).sum(0)
        attack_horizon[:, 1] += (predicted & ~truth).sum(0)
        attack_horizon[:, 2] += (~predicted & truth).sum(0)
    if world > 1:
        for value in (sums, horizon_sums, horizon_valid, mouse_absolute,
                      attack_counts, attack_horizon):
            dist.all_reduce(value)
    windows = sums[0].clamp_min(1)
    result = dict(
        windows=int(sums[0]), loss=float(sums[1] / windows),
        parts={name: float(sums[index + 2] / windows)
               for index, name in enumerate(components)})
    result["attack"] = {}
    for index, threshold in enumerate((0.1, 0.3, 0.5)):
        tp, fp, fn = attack_counts[index]
        result["attack"][str(threshold)] = dict(
            tp=int(tp), fp=int(fp), fn=int(fn),
            precision=float(tp / (tp + fp).clamp_min(1)),
            recall=float(tp / (tp + fn).clamp_min(1)),
            f1=float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
    per_horizon = []
    for horizon in range(8):
        denominator = horizon_valid[horizon].clamp_min(1)
        tp, fp, fn = attack_horizon[horizon]
        per_horizon.append(dict(
            horizon=horizon + 1,
            parts={name: float(horizon_sums[index, horizon] / denominator)
                   for index, name in enumerate(components)},
            mouse_x_mae=float(mouse_absolute[0, horizon] / denominator),
            mouse_y_mae=float(mouse_absolute[1, horizon] / denominator),
            attack_05=dict(
                precision=float(tp / (tp + fp).clamp_min(1)),
                recall=float(tp / (tp + fn).clamp_min(1)),
                f1=float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))))
    result["per_horizon"] = per_horizon
    model.train()
    return result


def capture_rng():
    state = np.random.get_state()
    return dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                python=random.getstate(),
                numpy=(state[0], state[1].tolist(), state[2], state[3], state[4]))


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state((value[0], np.array(value[1], dtype=np.uint32),
                         value[2], value[3], value[4]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-version", choices=("v2", "v3"), default="v2")
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--validate-every", type=int, default=2000)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--hidden", type=int, default=640)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=10)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--move-key-weight", type=float, default=1.0)
    parser.add_argument("--attack-weight", type=float, default=2.0)
    parser.add_argument("--mouse-smoothing", type=float, default=0.2)
    parser.add_argument("--mouse-move-positive-weight", type=float, default=2.0)
    parser.add_argument("--locomotion-weight", type=float, default=1.0)
    parser.add_argument("--combat-range-weight", type=float, default=0.5)
    parser.add_argument("--combat-bearing-weight", type=float, default=0.25)
    parser.add_argument("--attack-range", type=float, default=3.25)
    parser.add_argument("--horizon-min-weight", type=float, default=0.25)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-step", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-pilot", action="store_true")
    args = parser.parse_args()
    rank, world, device = init_distributed()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    summary = json.loads((args.cache / "summary.json").read_text())
    if summary["profile"] != "zombie_melee":
        raise ValueError("zombie v2 trainer requires the zombie_melee cache")
    if not summary.get("m3_state"):
        raise ValueError("cache must include M3 camera/resident actions")
    if not args.allow_pilot and not summary["selection"]["all_data"]:
        raise ValueError("production requires full eligible data")
    components = V3_COMPONENTS if args.model_version == "v3" else V2_COMPONENTS
    cfg_class = ZombieStatePolicyV3Args if args.model_version == "v3" else ZombieStatePolicyV2Args
    model_class = ZombieStatePolicyV3 if args.model_version == "v3" else ZombieStatePolicyV2
    cfg_values = dict(hidden=args.hidden, heads=args.heads, depth=args.depth,
                      dropout=args.dropout)
    if args.model_version == "v3":
        cfg_values["attack_range"] = args.attack_range
    cfg = cfg_class(summary["num_block_classes"], len(summary["item_vocabulary"]),
                    summary["profile"], **cfg_values)
    raw = model_class(cfg).to(device)
    if args.initialize and not args.resume:
        initialized = torch.load(args.initialize, map_location="cpu", weights_only=True)
        incompatible = raw.load_state_dict(initialized["model"], strict=False)
        if incompatible.unexpected_keys:
            raise ValueError("unexpected initialization tensors: "
                             + ", ".join(incompatible.unexpected_keys))
        allowed = ("combat_encoder.", "combat_fusion.", "range_head.", "bearing_head.")
        if any(not name.startswith(allowed) for name in incompatible.missing_keys):
            raise ValueError("incompatible initialization tensors: "
                             + ", ".join(incompatible.missing_keys))
        if rank == 0:
            print(json.dumps(dict(initialized_from=str(args.initialize),
                                  new_tensors=incompatible.missing_keys)), flush=True)
    optimizer = torch.optim.AdamW(
        raw.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    saved = None
    start = 0
    best = float("inf")
    bad_validations = 0
    if args.resume:
        saved = torch.load(args.output / "latest.pt", map_location="cpu", weights_only=True)
        if (saved["config"] != asdict(cfg) or saved["world_size"] != world
                or saved["batch_size"] != args.batch_size):
            raise ValueError("resume topology/config mismatch")
        if saved["dataset"] != summary:
            raise ValueError("resume dataset mismatch")
        schedule_keys = ["steps", "lr", "weight_decay", "warmup", "positive_weight",
                         "move_key_weight", "attack_weight", "mouse_smoothing",
                         "mouse_move_positive_weight", "horizon_min_weight", "seed"]
        if args.model_version == "v3":
            schedule_keys += ["locomotion_weight", "combat_range_weight",
                              "combat_bearing_weight", "attack_range"]
        for key in schedule_keys:
            if saved["training"][key] != getattr(args, key):
                raise ValueError("resume optimizer schedule mismatch: " + key)
        raw.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
        best = saved["best_val_loss"]
        bad_validations = saved.get("bad_validations", 0)
    model = DDP(raw, device_ids=[device.index], broadcast_buffers=False) if world > 1 else raw

    train = CachedStatePolicyDataset(args.cache / "train.jsonl", cfg.profile)
    val = CachedStatePolicyDataset(args.cache / "val_id.jsonl", cfg.profile)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank,
                                 shuffle=True, seed=args.seed)
    loader = DataLoader(
        train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
        persistent_workers=args.workers > 0, drop_last=True, pin_memory=True,
        collate_fn=collate_state_policy)
    val_ids = list(range(rank, len(val), world))
    val_loader = DataLoader(
        Subset(val, val_ids), batch_size=args.batch_size, num_workers=args.workers,
        pin_memory=True, collate_fn=collate_state_policy)
    if not len(loader):
        raise ValueError("too few training samples for batch/world size")
    if rank == 0:
        if not args.resume:
            args.output.mkdir(parents=True, exist_ok=False)
        source_paths = [
            "plot/models/state_policy_large_v2.py", "plot/models/state_policy_large.py",
            "plot/models/structured_action.py", "plot/data/state_policy_dataset.py",
            "plot/models/renderer_backbone/embeddings.py",
            "plot/models/renderer_backbone/voxel_rasterizer.py",
            "plot/models/renderer_backbone/camera_util.py",
            "plot/data/renderer_dataset.py", "train_scripts/train_state_zombie_v2.py"]
        root = Path(__file__).resolve().parents[1]
        hashes = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                  for path in source_paths}
        if not args.resume:
            for path in source_paths:
                destination = args.output / "source" / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(root / path, destination)
            config = dict(
                model=asdict(cfg),
                training={key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()},
                dataset=summary, world_size=world,
                global_batch=args.batch_size * world,
                model_version=args.model_version,
                parameters=sum(parameter.numel() for parameter in raw.parameters()),
                source_sha256=hashes)
            (args.output / "config.json").write_text(json.dumps(config, indent=2))
            (args.output / "baseline.json").write_text(
                json.dumps(hard_marginal_baseline(args.cache), indent=2))
            print(json.dumps(dict(configured=config["parameters"], world_size=world,
                                  global_batch=args.batch_size * world)), flush=True)
    if world > 1:
        dist.barrier()

    eval_loss_kwargs = dict(mouse_smoothing=args.mouse_smoothing)
    if args.model_version == "v3":
        eval_loss_kwargs.update(
            locomotion_weight=args.locomotion_weight,
            combat_range_weight=args.combat_range_weight,
            combat_bearing_weight=args.combat_bearing_weight)

    def save(step, improved=False):
        states = [None] * world
        if world > 1:
            dist.all_gather_object(states, capture_rng())
        else:
            states = [capture_rng()]
        if rank == 0:
            payload = dict(
                model=raw.state_dict(), optimizer=optimizer.state_dict(),
                config=asdict(cfg), dataset=summary, step=step,
                best_val_loss=best, bad_validations=bad_validations,
                rng=states, world_size=world, batch_size=args.batch_size,
                training={key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()})
            temporary = args.output / "latest.tmp"
            torch.save(payload, temporary)
            temporary.replace(args.output / "latest.pt")
            if improved:
                shutil.copy2(args.output / "latest.pt", args.output / "best-val-loss.tmp")
                (args.output / "best-val-loss.tmp").replace(args.output / "best-val-loss.pt")
            if step and step % 5000 == 0:
                shutil.copy2(args.output / "latest.pt", args.output / f"step-{step:06d}.tmp")
                (args.output / f"step-{step:06d}.tmp").replace(
                    args.output / f"step-{step:06d}.pt")
        if world > 1:
            dist.barrier()

    def validate(step):
        nonlocal best, bad_validations
        result = evaluate(raw, val_loader, device, world, components, eval_loss_kwargs)
        improved = result["loss"] < best
        best = min(best, result["loss"])
        if improved:
            bad_validations = 0
        elif step >= args.early_stop_min_step:
            bad_validations += 1
        stop = (args.early_stop_patience > 0 and step >= args.early_stop_min_step
                and bad_validations >= args.early_stop_patience)
        record = dict(step=step, val=result, improved=improved,
                      bad_validations=bad_validations, early_stop=stop)
        if rank == 0:
            with (args.output / "validation.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        save(step, improved)
        return stop

    epoch = start // len(loader)
    skip = start % len(loader)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(skip):
        next(iterator)
    if saved is not None:
        restore_rng(saved["rng"][rank])
    started = time.monotonic()
    final_step = start
    early_stopped = False
    horizon_weights = torch.linspace(1.0, args.horizon_min_weight, 8, device=device)
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
        learning_rate = args.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["inputs"])
            loss_kwargs = dict(
                positive_weight=args.positive_weight,
                move_key_weight=args.move_key_weight,
                attack_weight=args.attack_weight,
                mouse_smoothing=args.mouse_smoothing,
                mouse_move_positive_weight=args.mouse_move_positive_weight,
                horizon_weights=horizon_weights)
            if args.model_version == "v3":
                loss_kwargs.update(
                    locomotion_weight=args.locomotion_weight,
                    combat_range_weight=args.combat_range_weight,
                    combat_bearing_weight=args.combat_bearing_weight)
            loss, parts = raw.loss(
                logits, batch["target_actions"], batch["valid_mask"], **loss_kwargs)
        finite = torch.isfinite(loss).int()
        if world > 1:
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError("nonfinite training loss on a rank")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        final_step = step
        if step == start + 1 or step % 20 == 0:
            metrics = torch.stack((loss.detach(), *[
                parts[name].detach().mean() for name in components])).float()
            if world > 1:
                dist.all_reduce(metrics)
                metrics /= world
            if rank == 0:
                row = dict(
                    step=step, epoch=epoch, loss=float(metrics[0]),
                    parts={name: float(metrics[index + 1])
                           for index, name in enumerate(components)},
                    grad_norm=float(grad), lr=learning_rate,
                    elapsed_seconds=time.monotonic() - started,
                    gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
                with (args.output / "train.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        should_validate = (step == 1000 or step % args.validate_every == 0
                           or step == args.steps)
        if should_validate:
            if validate(step):
                early_stopped = True
                break
        elif step % args.checkpoint_every == 0:
            save(step)
    if rank == 0:
        (args.output / "COMPLETE.json").write_text(json.dumps(dict(
            planned_steps=args.steps, final_step=final_step,
            early_stopped=early_stopped, best_val_loss=best,
            profile=cfg.profile)))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
