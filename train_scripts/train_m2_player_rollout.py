"""Fine-tune M2 player dynamics through differentiable 16--64 frame rollouts."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from plot.data.player_rollout_stream import PlayerRolloutStream, collate_player_rollouts
from plot.kinematics import KinematicsConfig, wrap_angle
from plot.models.transition import TransitionArgs, TransitionNetwork


def masked_mean(values, mask):
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(values)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / mask.sum().clamp_min(1)


def move(batch, device):
    return {group: ({key: value.to(device, non_blocking=True) for key, value in values.items()}
                    if isinstance(values, dict) else values)
            for group, values in batch.items()}


def player_attack_loss(output, inputs, targets):
    state = targets["state_valid"].bool() & inputs["active"][:, None]
    terms = {
        "position": masked_mean(F.smooth_l1_loss(
            output["pose"][..., :3], targets["pose"][..., :3], reduction="none"), state),
        "angle": masked_mean(1 - torch.cos(
            output["pose"][..., 3:] - targets["pose"][..., 3:]), state),
        "camera_position": masked_mean(F.smooth_l1_loss(
            output["camera_relative"], targets["camera_relative"], reduction="none"), state),
        "camera_direction": masked_mean(1 - F.cosine_similarity(
            output["camera_direction"], targets["camera_direction"], dim=-1), state),
    }
    held_valid = state & targets["held_item_valid"].bool()
    terms["held_item"] = masked_mean(F.cross_entropy(
        output["held_logits"].flatten(0, 2), targets["held_item"].flatten(),
        reduction="none").reshape_as(state), held_valid)

    count_valid = targets["attack_count_valid"].bool() & inputs["active"].bool()
    terms["attack_count"] = masked_mean(F.cross_entropy(
        output["attack_count_logits"].flatten(0, 1),
        targets["attack_count"].flatten(), reduction="none").reshape_as(count_valid), count_valid)
    slot_valid = targets["attack_slot_valid"].bool()
    if slot_valid.any():
        terms["attack_time"] = F.cross_entropy(
            output["attack_time_logits"][slot_valid], targets["attack_time"][slot_valid])
        terms["attack_target"] = F.cross_entropy(
            output["attack_target_logits"][slot_valid], targets["attack_target"][slot_valid])
    else:
        zero = output["attack_time_logits"].sum() * 0
        terms["attack_time"] = terms["attack_target"] = zero
    damage_valid = targets["attack_damage_valid"].bool() & slot_valid
    terms["attack_damage"] = masked_mean(F.smooth_l1_loss(
        output["attack_damage"], targets["attack_damage"], reduction="none"), damage_valid)
    attack = (terms["attack_count"] + terms["attack_time"] + terms["attack_target"]
              + 0.25 * terms["attack_damage"])
    dynamics = (terms["position"] + 0.25 * terms["angle"]
                + 0.25 * terms["camera_position"] + 0.25 * terms["camera_direction"]
                + 0.1 * terms["held_item"])
    return dynamics + 0.25 * attack, terms, state


def rollout(model, sequence, device, *, train=True, short_horizon_weight=0.0,
            stationary_weight=0.0, stationary_threshold=0.05):
    batches = [move(batch, device) for batch in sequence]
    pose = camera_relative = camera_direction = velocity = None
    losses, metrics = [], {}
    final_errors = []
    stationary_errors = []
    for block, batch in enumerate(batches):
        inputs = dict(batch["inputs"])
        if block:
            if "player_voxel_relative_xyz" in inputs:
                inputs["player_voxel_relative_xyz"] = (
                    inputs["player_voxel_relative_xyz"]
                    + inputs["initial_pose"][:, :, None, :3] - pose[:, :, None, :3])
            inputs["initial_pose"] = pose
            inputs["initial_velocity"] = velocity
            inputs["camera_relative"] = camera_relative
            inputs["camera_direction"] = camera_direction
            # HP, held item, inventory and active masks remain observed boundary
            # conditions; this experiment isolates continuous pose drift.
        output = model.forward_player(inputs)
        loss, terms, state = player_attack_loss(output, inputs, batch["targets"])
        # Later endpoints matter more because deployment feeds them back.
        endpoint = masked_mean(F.smooth_l1_loss(
            output["pose"][:, -1, :, :3], batch["targets"]["pose"][:, -1, :, :3],
            reduction="none"), state[:, -1])
        endpoint_weight = 0.5 + block / max(1, len(batches) - 1)
        if block == 0:
            endpoint_weight += short_horizon_weight

        # Classify motion from the observed boundary and ground-truth endpoint,
        # never from the autoregressive prediction.  This makes a drifting
        # prediction remain a stationary training example instead of escaping
        # the stationary loss after it has already moved.
        true_initial_pose = batch["inputs"]["initial_pose"]
        true_displacement = (batch["targets"]["pose"][:, -1, :, :3]
                             - true_initial_pose[..., :3]).norm(dim=-1)
        stationary = state[:, -1] & (true_displacement <= stationary_threshold)
        stationary_endpoint = masked_mean(F.smooth_l1_loss(
            output["pose"][:, -1, :, :3], batch["targets"]["pose"][:, -1, :, :3],
            reduction="none"), stationary)
        loss = loss + endpoint_weight * endpoint + stationary_weight * stationary_endpoint
        losses.append(loss)
        pose = output["pose"][:, -1]
        velocity = output["velocity"][:, -1]
        camera_relative = output["camera_relative"][:, -1]
        camera_direction = output["camera_direction"][:, -1]
        error = (pose[..., :3] - batch["targets"]["pose"][:, -1, :, :3]).norm(dim=-1)
        valid = state[:, -1]
        final_errors.append((torch.where(valid, error, torch.zeros_like(error)).sum(), valid.sum()))
        stationary_errors.append((torch.where(
            stationary, error, torch.zeros_like(error)).sum(), stationary.sum()))
        for name, value in terms.items():
            metrics[name] = metrics.get(name, 0.0) + float(value.detach())
    total = torch.stack(losses).mean()
    result = {name: value / len(batches) for name, value in metrics.items()}
    result["loss"] = float(total.detach())
    for block, (error, count) in enumerate(final_errors, 1):
        result[f"position_error_{block * 8}f"] = float((error / count.clamp_min(1)).detach())
    for block, (error, count) in enumerate(stationary_errors, 1):
        result[f"stationary_error_{block * 8}f"] = float((error / count.clamp_min(1)).detach())
        result[f"stationary_count_{block * 8}f"] = float(count.detach())
    return total, result


class RolloutModule(nn.Module):
    """Expose the multi-call rollout as one DDP forward transaction."""
    def __init__(self, m2, device, *, short_horizon_weight=0.0,
                 stationary_weight=0.0, stationary_threshold=0.05):
        super().__init__()
        self.m2 = m2
        self.device = device
        self.loss_kwargs = dict(short_horizon_weight=short_horizon_weight,
                                stationary_weight=stationary_weight,
                                stationary_threshold=stationary_threshold)

    def forward(self, sequence):
        return rollout(self.m2, sequence, self.device, **self.loss_kwargs)


@torch.no_grad()
def evaluate(model, loader, device, batches, **loss_kwargs):
    model.eval()
    totals = {}; count = 0
    for sequence in loader:
        _, metrics = rollout(model, sequence, device, train=False, **loss_kwargs)
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if count >= batches:
            break
    return {key: value / max(1, count) for key, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--rollout-blocks", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--player-velocity", action="store_true",
                        help="Condition the player branch on boundary velocity.")
    parser.add_argument("--player-voxel", action="store_true",
                        help="Cross-attend to a compact local 7-cube in the player branch.")
    parser.add_argument("--geometry-only", action="store_true",
                        help="Freeze the established player branch and train only new voxel modules.")
    parser.add_argument("--short-horizon-weight", type=float, default=0.0,
                        help="Extra weight for the first 8-frame endpoint.")
    parser.add_argument("--stationary-weight", type=float, default=0.0,
                        help="Extra endpoint loss for ground-truth stationary blocks.")
    parser.add_argument("--stationary-threshold", type=float, default=0.05,
                        help="Maximum 8-frame displacement considered stationary.")
    parser.add_argument("--selection-short-weight", type=float, default=0.0,
                        help="8-frame validation weight used to select the best checkpoint.")
    parser.add_argument("--selection-stationary-weight", type=float, default=0.0,
                        help="Stationary 8-frame validation weight used for checkpoint selection.")
    args = parser.parse_args()

    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    torch.set_num_threads(2)

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    model_cfg = dict(checkpoint["config"]["model"])
    model_cfg["kinematics"] = KinematicsConfig(**model_cfg["kinematics"])
    if args.player_velocity:
        model_cfg["player_velocity"] = True
    if args.player_voxel:
        model_cfg["player_voxel"] = True
    model = TransitionNetwork(TransitionArgs(**model_cfg))
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    source_has_velocity = checkpoint["config"]["model"].get("player_velocity", False)
    allowed_missing = ({"player_velocity.weight", "player_velocity.bias"}
                       if model_cfg.get("player_velocity") and not source_has_velocity else set())
    if model_cfg.get("player_voxel") and not checkpoint["config"]["model"].get("player_voxel",False):
        allowed_missing.update(name for name in missing if name.startswith("player_voxel_"))
    if set(missing) != allowed_missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if args.geometry_only and not args.player_voxel:
        raise ValueError("--geometry-only requires --player-voxel")
    prefixes = (("player_voxel_",) if args.geometry_only else
                ("player_", "pose_head", "camera_head", "held_head", "attack_"))
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
    model.to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    index = json.loads(args.index.read_text())
    items = checkpoint["config"]["items"]
    geometry_vocab = checkpoint["config"]["class_to_raw"] if args.player_voxel else None
    train = PlayerRolloutStream(index, "train", items, args.rollout_blocks, args.seed,
                                rank, world, class_to_raw=geometry_vocab)
    val = PlayerRolloutStream(index, "val_id", items, args.rollout_blocks, args.seed,
                              rank, world, fixed_order=True, class_to_raw=geometry_vocab)
    loader_args = dict(batch_size=args.batch_size, num_workers=args.workers,
                       collate_fn=collate_player_rollouts, pin_memory=True,
                       persistent_workers=args.workers > 0)
    if args.workers:
        loader_args["prefetch_factor"] = 1
    train_loader = DataLoader(train, **loader_args)
    val_loader = DataLoader(val, **loader_args)
    raw_model = model
    loss_kwargs = dict(short_horizon_weight=args.short_horizon_weight,
                       stationary_weight=args.stationary_weight,
                       stationary_threshold=args.stationary_threshold)
    rollout_model = RolloutModule(raw_model, device, **loss_kwargs)
    if distributed:
        # Attack-positive slots are sparse and can be absent on an individual
        # rank even when other ranks have positives in the same global step.
        rollout_model = DDP(rollout_model, device_ids=[local_rank], find_unused_parameters=True)

    out = args.output_dir
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        if (out / "metrics.jsonl").exists():
            raise ValueError("output contains an existing run")
        config = dict(
            schema="m2-player-rollout-v1", training=vars(args), model=asdict(raw_model.cfg),
            items=items, class_to_raw=checkpoint["config"]["class_to_raw"],
            source_checkpoint=str(args.resume), rollout_frames=8 * args.rollout_blocks,
            trainable_parameters=sum(parameter.numel() for parameter in trainable),
            frozen_event_branch=True,
            supervision=["autoregressive_pose", "angle", "camera", "held_item", "attack"])
        config["training"] = {key: str(value) if isinstance(value, Path) else value
                              for key, value in config["training"].items()}
        (out / "config.json").write_text(json.dumps(config, indent=2))

    def log(step, split, values):
        if rank != 0:
            return
        row = {"step": step, "split": split, **values}
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    rollout_frames = args.rollout_blocks * 8
    best_rollout_error = float("inf")
    if rank == 0:
        initial_validation = evaluate(raw_model, val_loader, device, args.eval_batches,
                                      **loss_kwargs)
        log(0, "val", initial_validation)
        best_rollout_error = (initial_validation[f"position_error_{rollout_frames}f"]
                              + args.selection_short_weight
                              * initial_validation["position_error_8f"]
                              + args.selection_stationary_weight
                              * initial_validation["stationary_error_8f"])
    if distributed:
        torch.distributed.barrier()
    iterator = iter(train_loader)
    for step in range(1, args.steps + 1):
        try:
            sequence = next(iterator)
        except StopIteration:
            iterator = iter(train_loader); sequence = next(iterator)
        rollout_model.train(); optimizer.zero_grad(set_to_none=True)
        loss, metrics = rollout_model(sequence)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite rollout loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step % 20 == 0:
            log(step, "train", {**metrics, "gradient_norm": float(norm)})
        if step % args.save_every == 0 or step == args.steps:
            if distributed:
                torch.distributed.barrier()
            if rank == 0:
                validation = evaluate(raw_model, val_loader, device, args.eval_batches,
                                      **loss_kwargs)
                log(step, "val", validation)
                payload = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                           "step": step, "config": json.loads((out / "config.json").read_text())}
                temporary = out / "checkpoint.tmp"
                torch.save(payload, temporary); temporary.replace(out / "latest.pt")
                score = (validation[f"position_error_{rollout_frames}f"]
                         + args.selection_short_weight * validation["position_error_8f"]
                         + args.selection_stationary_weight
                         * validation["stationary_error_8f"])
                if score < best_rollout_error:
                    best_rollout_error = score
                    temporary = out / "best.tmp"
                    torch.save(payload, temporary); temporary.replace(out / f"best-val-{rollout_frames}f.pt")
                    (out / "best.json").write_text(json.dumps(
                        {"step": step, "selection_score": score,
                         f"position_error_{rollout_frames}f":
                         validation[f"position_error_{rollout_frames}f"],
                         "position_error_8f": validation["position_error_8f"],
                         "stationary_error_8f": validation["stationary_error_8f"]}, indent=2))
            if distributed:
                torch.distributed.barrier()
    if rank == 0:
        (out / "COMPLETE.json").write_text(json.dumps({"steps": args.steps}))
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
