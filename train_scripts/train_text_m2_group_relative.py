"""Critic-free group-relative TextAgent training with frozen-M2 rewards.

All candidates are hard, executable 8-frame actions.  M2 is used only under
no_grad to rank the candidates; policy gradients come from their M4 log-probs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.nn import functional as F

from plot.data.fill_dataset import BlockVocabulary
from plot.data.state_policy_dataset import CachedStatePolicyDataset, collate_state_policy
from plot.data.transition_dataset import VOXEL_COUNT, episode_windows, event_position
from plot.kinematics import KinematicsConfig
from plot.models.state_policy_text_v2 import IndependentStatePolicyV4, TextStatePolicyV2Args
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.training.transition_trainer import decode_ordered_events


def move_group(group, device):
    return {key: value.to(device, non_blocking=True) for key, value in group.items()}


def read_placements(episode):
    placements = []
    manifest = json.loads((episode / "manifest.json").read_text())
    event_file = episode / manifest.get("event_file", "events.jsonl")
    for line in event_file.read_text().splitlines():
        event = json.loads(line)
        if event.get("initialization_event") or event.get("event") not in (
                "block_placed", "scaffold_placed"):
            continue
        coordinate = event_position(event)
        transition = event.get("transition_index")
        if coordinate is not None and transition is not None:
            placements.append((int(transition), tuple(map(int, coordinate))))
    placements.sort()
    return placements


def infer_placement_classes(samples):
    classes = []
    for sample in samples:
        targets = sample["targets"]
        mask = targets["block_valid"].bool() & (targets["address"] < VOXEL_COUNT)
        classes.extend(targets["block"][mask].tolist())
    if not classes:
        raise ValueError("episode contains no representable placement payload")
    return Counter(map(int, classes)).most_common(1)[0][0]


def load_examples(cache, m2_index, m2_checkpoint, episodes, windows_per_episode,
                  seed, episode_substring=None):
    base = CachedStatePolicyDataset(cache / "train.jsonl", "language_builder")
    by_key = defaultdict(list)
    for index, row in enumerate(base.rows):
        if episode_substring and episode_substring not in row["episode_id"]:
            continue
        by_key[(row["episode_id"], int(row["anchor"]))].append(index)
    available_episodes = {episode_id for episode_id, _ in by_key}

    index = json.loads(m2_index.read_text())
    root = Path(index["dataset_root"])
    vocabulary = BlockVocabulary(tuple(m2_checkpoint["config"]["class_to_raw"]))
    records = list(index["splits"]["train"])
    random.Random(seed).shuffle(records)
    examples = []
    used_episodes = []
    for record in records:
        episode = root / record["path"]
        episode_id = episode.name
        if episode_substring and episode_substring not in episode_id:
            continue
        if episode_id not in available_episodes:
            continue
        samples, _ = episode_windows(
            episode, vocabulary, m2_checkpoint["config"]["items"],
            windows_per_episode=windows_per_episode, split="train", stride=8)
        if not samples:
            continue
        placements = read_placements(episode)
        if not placements:
            continue
        block_class = infer_placement_classes(samples)
        episode_examples = []
        for m2_sample in samples:
            start = int(m2_sample["metadata"]["start"])
            for m4_index in by_key.get((episode_id, start), ()):
                m4_sample = base[m4_index]
                target = int(base.rows[m4_index]["agent_slot"])
                anchor = np.asarray(
                    m2_sample["metadata"]["anchors"][target], np.int64)
                addresses = []
                for transition, coordinate in placements:
                    if transition < start:
                        continue
                    local = np.asarray(coordinate, np.int64) - (anchor - 6)
                    if (local >= 0).all() and (local < 13).all():
                        addresses.append(int(np.ravel_multi_index(
                            tuple(local), (13, 13, 13))))
                addresses = sorted(set(addresses))
                if addresses:
                    episode_examples.append(dict(
                        m4=m4_sample, m2=m2_sample, target=target,
                        goal_addresses=addresses, goal_block=block_class,
                        episode=episode_id, start=start))
        if episode_examples:
            examples.extend(episode_examples)
            used_episodes.append(episode_id)
        if episodes > 0 and len(used_episodes) >= episodes:
            break
    if not examples:
        raise RuntimeError("no aligned M2/M4 examples with local unfinished goals")
    return examples, used_episodes


def collate_examples(rows):
    m4 = collate_state_policy([row["m4"] for row in rows])
    inputs = {key: torch.stack([row["m2"]["inputs"][key] for row in rows])
              for key in rows[0]["m2"]["inputs"]}
    targets = {key: torch.stack([row["m2"]["targets"][key] for row in rows])
               for key in rows[0]["m2"]["targets"]}
    mask = torch.zeros(len(rows), VOXEL_COUNT, dtype=torch.bool)
    for index, row in enumerate(rows):
        mask[index, row["goal_addresses"]] = True
    return dict(
        m4=m4, m2_inputs=inputs, m2_targets=targets,
        target=torch.tensor([row["target"] for row in rows]),
        goal_mask=mask,
        goal_block=torch.tensor([row["goal_block"] for row in rows]),
        metadata=[(row["episode"], row["start"]) for row in rows])


def expand_logits(logits, group):
    return {key: value.expand(group, *value.shape[1:]) for key, value in logits.items()}


def sample_action_group(model, logits, group, proposal_actions=None):
    expanded = expand_logits(logits, group)
    sample = {}
    sample["keys"] = torch.distributions.Bernoulli(
        logits=expanded["keys"].float()).sample()
    sample["hotbar"] = torch.distributions.Categorical(
        logits=expanded["hotbar"].float()).sample()
    for axis in ("mouse_x", "mouse_y"):
        sample[axis + "_move"] = torch.distributions.Bernoulli(
            logits=expanded[axis + "_move"].float()).sample()
        direction = expanded[axis].float().clone()
        bins = getattr(model.loss_head, "mouse_bins_" + axis[-1])
        zero = int((bins == 0).nonzero(as_tuple=False)[0])
        direction[..., zero] = -1e4
        sample[axis] = torch.distributions.Categorical(logits=direction).sample()

    # Candidate zero is deterministic greedy M4, making every group at least
    # as informative as the current inference policy.
    sample["keys"][0] = (logits["keys"][0].sigmoid() >= .5)
    sample["hotbar"][0] = logits["hotbar"][0].argmax(-1)
    for axis in ("mouse_x", "mouse_y"):
        sample[axis + "_move"][0] = (logits[axis + "_move"][0].sigmoid() >= .5)
        direction = logits[axis][0].clone()
        bins = getattr(model.loss_head, "mouse_bins_" + axis[-1])
        zero = int((bins == 0).nonzero(as_tuple=False)[0])
        direction[..., zero] = -torch.inf
        sample[axis][0] = direction.argmax(-1)

    # One recorded action can be used as an exploration proposal.  It is not a
    # label here: frozen M2 still decides its reward relative to every sample.
    if proposal_actions is not None and group > 1:
        targets = model.loss_head.targets(proposal_actions.float())
        sample["keys"][1] = targets["keys"][0]
        sample["hotbar"][1] = targets["hotbar"][0]
        for axis in ("mouse_x", "mouse_y"):
            bins = getattr(model.loss_head, "mouse_bins_" + axis[-1])
            zero = int((bins == 0).nonzero(as_tuple=False)[0])
            sample[axis][1] = targets[axis][0]
            sample[axis + "_move"][1] = (targets[axis][0] != zero).float()

    actions = logits["keys"].new_zeros(group, 8, 23).float()
    actions[..., model.loss_head.key_indices] = sample["keys"]
    for slot in range(1, 10):
        actions[..., 11 + slot] = (sample["hotbar"] == slot).float()
    for axis, index in (("mouse_x", 21), ("mouse_y", 22)):
        bins = getattr(model.loss_head, "mouse_bins_" + axis[-1]).to(actions)
        actions[..., index] = bins[sample[axis]] * sample[axis + "_move"]
    return actions, sample


def action_log_probability(model, logits, sample):
    group = sample["keys"].shape[0]
    expanded = expand_logits(logits, group)
    terms = [torch.distributions.Bernoulli(
        logits=expanded["keys"].float()).log_prob(sample["keys"]).sum((-1, -2))]
    terms.append(torch.distributions.Categorical(
        logits=expanded["hotbar"].float()).log_prob(sample["hotbar"]).sum(-1))
    for axis in ("mouse_x", "mouse_y"):
        terms.append(torch.distributions.Bernoulli(
            logits=expanded[axis + "_move"].float()).log_prob(
                sample[axis + "_move"]).sum(-1))
        direction = expanded[axis].float().clone()
        bins = getattr(model.loss_head, "mouse_bins_" + axis[-1])
        zero = int((bins == 0).nonzero(as_tuple=False)[0])
        direction[..., zero] = -1e4
        direction_log_probability = torch.distributions.Categorical(
            logits=direction).log_prob(sample[axis])
        # Direction is undefined when the mouse-move gate is off.  In
        # particular, recorded no-move actions use the zero bin, which is
        # intentionally excluded from the conditional moving distribution.
        terms.append((direction_log_probability * sample[axis + "_move"]).sum(-1))
    # Normalize by time so beta and learning rate do not depend on horizon.
    return torch.stack(terms).sum(0) / 8.0


def distribution_kl(model, current, reference):
    terms = []
    for name in ("keys", "mouse_x_move", "mouse_y_move"):
        p = current[name].float().sigmoid()
        q = reference[name].float().sigmoid()
        terms.append((p * (torch.log(p.clamp_min(1e-6)) - torch.log(q.clamp_min(1e-6)))
                      + (1 - p) * (torch.log((1 - p).clamp_min(1e-6))
                                   - torch.log((1 - q).clamp_min(1e-6)))).mean())
    for name in ("hotbar", "mouse_x", "mouse_y"):
        log_p = current[name].float().log_softmax(-1)
        log_q = reference[name].float().log_softmax(-1)
        terms.append((log_p.exp() * (log_p - log_q)).sum(-1).mean())
    return torch.stack(terms).mean()


@torch.no_grad()
def score_group(m2, output, m2_inputs, target, goal_mask, goal_block, actions,
                wrong_weight, hard_correct_bonus, hard_wrong_penalty,
                distance_weight, aim_weight):
    rows = torch.arange(len(target), device=target.device)
    count = output["event_count_logits"][rows, target].float().softmax(-1)
    active = 1.0 - count.cumsum(-1)[..., :-1]
    address_logits = torch.nan_to_num(
        output["event_address_logits"].float(), neginf=-1e4, posinf=1e4)
    address_probability = address_logits[rows, target].softmax(-1)
    voxel_probability = address_probability[..., :VOXEL_COUNT]
    address_correct = (voxel_probability * goal_mask[:, None]).sum(-1)
    all_probability = address_logits.softmax(-1)
    selected = torch.einsum(
        "baqn,band->baqd", all_probability, output["candidates"][:, :, :-1].float())
    hidden = m2.payload(torch.cat((output["event_hidden"].float(), selected), -1))
    block_probability = m2.block_head(hidden)[rows, target].softmax(-1)
    block_correct = block_probability.gather(
        -1, goal_block[:, None, None].expand(-1, block_probability.shape[1], 1)).squeeze(-1)
    slot_correct = active * address_correct * block_correct
    success = -torch.expm1(torch.log1p(-slot_correct.clamp(max=1 - 1e-6)).sum(-1))
    expected_correct = slot_correct.sum(-1)
    expected_writes = (active * voxel_probability.sum(-1)).sum(-1)
    expected_wrong = (expected_writes - expected_correct).clamp_min(0)

    address, block, _ = decode_ordered_events(
        m2, output, VOXEL_COUNT + output["event_count_logits"].shape[1], 8)
    address = address[rows, :, target]
    block = block[rows, :, target]
    is_write = address < VOXEL_COUNT
    safe = address.clamp(0, VOXEL_COUNT - 1)
    hard_correct = (is_write & goal_mask.gather(1, safe)
                    & (block == goal_block[:, None])).sum(-1).float()
    hard_wrong = (is_write.sum(-1).float() - hard_correct).clamp_min(0)

    # Dense task progress for eight-frame blocks that do not place yet.
    relative_xyz = m2_inputs["voxel_relative_xyz"][rows, target].float()
    initial_position = m2_inputs["initial_pose"][rows, target, :3].float()
    goal_world = initial_position[:, None] + relative_xyz
    initial_distance = relative_xyz.norm(dim=-1).masked_fill(~goal_mask, 1e6).amin(-1)
    final_position = output["pose"][rows, -1, target, :3].float()
    final_delta = goal_world - final_position[:, None]
    final_distance = final_delta.norm(dim=-1).masked_fill(~goal_mask, 1e6).amin(-1)
    distance_progress = (initial_distance - final_distance).clamp(-1, 1)

    initial_camera = (initial_position
                      + m2_inputs["camera_relative"][rows, target].float())
    initial_direction = F.normalize(
        m2_inputs["camera_direction"][rows, target].float(), dim=-1)
    initial_goal_direction = F.normalize(goal_world - initial_camera[:, None], dim=-1)
    initial_alignment = (initial_goal_direction * initial_direction[:, None]).sum(-1)
    initial_alignment = initial_alignment.masked_fill(~goal_mask, -1).amax(-1)
    final_camera = final_position + output["camera_relative"][rows, -1, target].float()
    final_direction = F.normalize(
        output["camera_direction"][rows, -1, target].float(), dim=-1)
    final_goal_direction = F.normalize(goal_world - final_camera[:, None], dim=-1)
    final_alignment = (final_goal_direction * final_direction[:, None]).sum(-1)
    final_alignment = final_alignment.masked_fill(~goal_mask, -1).amax(-1)
    aim_progress = (final_alignment - initial_alignment).clamp(-1, 1)
    reward = (success - wrong_weight * expected_wrong
              + hard_correct_bonus * hard_correct - hard_wrong_penalty * hard_wrong
              + distance_weight * distance_progress + aim_weight * aim_progress)
    return reward, dict(success=success, expected_wrong=expected_wrong,
                        hard_correct=hard_correct, hard_wrong=hard_wrong,
                        distance_progress=distance_progress, aim_progress=aim_progress,
                        place_frames=(actions[..., 9] > .5).sum(-1).float())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--m2-index", type=Path, required=True)
    parser.add_argument("--m2-checkpoint", type=Path, required=True)
    parser.add_argument("--m4-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument(
        "--episode-substring", default=None,
        help="Restrict the training pool to episode IDs containing this text.")
    parser.add_argument("--windows-per-episode", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--weight-decay", type=float, default=.05)
    parser.add_argument("--wrong-weight", type=float, default=5.0)
    parser.add_argument("--hard-correct-bonus", type=float, default=2.0)
    parser.add_argument("--hard-wrong-penalty", type=float, default=5.0)
    parser.add_argument("--distance-weight", type=float, default=1.0)
    parser.add_argument("--aim-weight", type=float, default=.25)
    parser.add_argument("--kl-weight", type=float, default=.05)
    parser.add_argument("--bc-weight", type=float, default=.05)
    parser.add_argument("--pointer-weight", type=float, default=.05)
    parser.add_argument("--min-reward-std", type=float, default=.01)
    parser.add_argument("--include-gt-proposal", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    checkpoint = torch.load(args.m4_checkpoint, map_location="cpu", weights_only=True)
    m4 = IndependentStatePolicyV4(TextStatePolicyV2Args(**checkpoint["config"]))
    m4.load_state_dict(checkpoint["model"], strict=True); m4.to(device).train()
    reference = IndependentStatePolicyV4(TextStatePolicyV2Args(**checkpoint["config"]))
    reference.load_state_dict(checkpoint["model"], strict=True); reference.to(device).eval()
    for parameter in reference.parameters(): parameter.requires_grad_(False)

    m2_checkpoint = torch.load(args.m2_checkpoint, map_location="cpu", weights_only=True)
    cfg = dict(m2_checkpoint["config"]["model"])
    cfg["kinematics"] = KinematicsConfig(**cfg["kinematics"])
    m2 = TransitionNetwork(TransitionArgs(**cfg))
    m2.load_state_dict(m2_checkpoint["model"], strict=True); m2.to(device).eval()
    for parameter in m2.parameters(): parameter.requires_grad_(False)
    examples, used = load_examples(args.cache, args.m2_index, m2_checkpoint,
                                   args.episodes, args.windows_per_episode, args.seed,
                                   args.episode_substring)
    optimizer = torch.optim.AdamW(m4.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output.mkdir(parents=True, exist_ok=False)
    config = dict(schema="m4-frozen-m2-group-relative-v1", frozen_m2=True,
                  examples=len(examples), used_episodes=used,
                  training={key: str(value) if isinstance(value, Path) else value
                            for key, value in vars(args).items()}, m4=asdict(m4.cfg))
    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    rng = random.Random(args.seed); started = time.monotonic()
    for step in range(1, args.steps + 1):
        packed = collate_examples([examples[rng.randrange(len(examples))]])
        inputs = move_group(packed["m4"]["inputs"], device)
        target_actions = packed["m4"]["target_actions"].to(device)
        valid = packed["m4"]["valid_mask"].to(device)
        goal_mask = packed["goal_mask"].to(device)
        goal_block = packed["goal_block"].to(device)
        target = packed["target"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m4(inputs)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            reference_logits = reference(inputs)
        actions, sample = sample_action_group(
            m4, logits, args.group_size,
            target_actions if args.include_gt_proposal else None)
        log_probability = action_log_probability(m4, logits, sample)

        m2_inputs = move_group(packed["m2_inputs"], device)
        m2_inputs = {key: value.expand(args.group_size, *value.shape[1:]).clone()
                     for key, value in m2_inputs.items()}
        group_target = target.expand(args.group_size)
        rows = torch.arange(args.group_size, device=device)
        m2_inputs["actions"][rows, :, group_target] = actions
        group_mask = goal_mask.expand(args.group_size, -1)
        group_block = goal_block.expand(args.group_size)
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            output = m2(m2_inputs)
            reward, reward_parts = score_group(
                m2, output, m2_inputs, group_target, group_mask, group_block, actions,
                args.wrong_weight, args.hard_correct_bonus, args.hard_wrong_penalty,
                args.distance_weight, args.aim_weight)
        reward_std = reward.std(unbiased=False)
        hard_variation = (reward_parts["hard_correct"].unique().numel() > 1
                          or reward_parts["hard_wrong"].unique().numel() > 1)
        meaningful_group = bool(reward_std >= args.min_reward_std or hard_variation)
        advantage = ((reward - reward.mean()) / reward_std.clamp_min(1e-4)
                     if meaningful_group else torch.zeros_like(reward))
        policy_loss = -(advantage.detach() * log_probability).mean()
        kl = distribution_kl(m4, logits, reference_logits)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bc_loss, _ = m4.loss(logits, target_actions, valid)
        pointer = torch.nan_to_num(logits["target"].float(), neginf=-1e4).softmax(-1)
        pointer_mass = (pointer[..., :VOXEL_COUNT] * goal_mask).sum(-1)
        pointer_loss = -torch.log(pointer_mass.clamp_min(1e-6)).mean()
        loss = (policy_loss + args.kl_weight * kl + args.bc_weight * bc_loss
                + args.pointer_weight * pointer_loss)
        optimizer.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(m4.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        best = int(reward.argmax()); greedy = 0
        row = dict(step=step, loss=float(loss.detach()), policy_loss=float(policy_loss.detach()),
                   kl=float(kl.detach()), bc_loss=float(bc_loss.detach()),
                   pointer_loss=float(pointer_loss.detach()), pointer_mass=float(pointer_mass.detach()),
                   reward_mean=float(reward.mean()), reward_std=float(reward_std),
                   meaningful_group=meaningful_group,
                   reward_best=float(reward[best]), reward_greedy=float(reward[greedy]),
                   reward_gt_proposal=(float(reward[1]) if args.include_gt_proposal else None),
                   best_is_greedy=(best == greedy), unique_rewards=int(reward.unique().numel()),
                   best_hard_correct=int(reward_parts["hard_correct"][best]),
                   best_hard_wrong=int(reward_parts["hard_wrong"][best]),
                   greedy_hard_correct=int(reward_parts["hard_correct"][greedy]),
                   greedy_hard_wrong=int(reward_parts["hard_wrong"][greedy]),
                   gt_proposal_hard_correct=(int(reward_parts["hard_correct"][1])
                                             if args.include_gt_proposal else None),
                   gt_proposal_hard_wrong=(int(reward_parts["hard_wrong"][1])
                                           if args.include_gt_proposal else None),
                   best_distance_progress=float(reward_parts["distance_progress"][best]),
                   greedy_distance_progress=float(reward_parts["distance_progress"][greedy]),
                   best_aim_progress=float(reward_parts["aim_progress"][best]),
                   grad_norm=float(grad), elapsed_seconds=time.monotonic() - started,
                   gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
        with (args.output / "train.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            torch.save(dict(model=m4.state_dict(), optimizer=optimizer.state_dict(),
                            config=asdict(m4.cfg), dataset=checkpoint["dataset"],
                            step=step, group_relative=config),
                       args.output / f"step-{step:06d}.pt")
    torch.save(dict(model=m4.state_dict(), optimizer=optimizer.state_dict(),
                    config=asdict(m4.cfg), dataset=checkpoint["dataset"],
                    step=args.steps, group_relative=config), args.output / "latest.pt")
    (args.output / "COMPLETE.json").write_text(json.dumps({"steps": args.steps}))


if __name__ == "__main__":
    main()
