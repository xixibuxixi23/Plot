"""Evaluate one TextAgent builder through the learned M2 world transition model.

This is deliberately an offline learned-world test.  Other residents keep their
recorded actions/states and M2 receives the recorded RGB at every eight-step
boundary; the selected builder state, its actions, and voxel writes are rolled
forward autoregressively.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data.fill_dataset import BlockVocabulary
from plot.data.renderer_dataset import raster_camera
from plot.data.state_policy_dataset import assemble_state_inputs, load_state_cache, read_state_sample
from plot.data.transition_dataset import VOXEL_COUNT, episode_windows, event_position
from plot.kinematics import KinematicsConfig
from plot.models.state_policy_text_v2 import IndependentStatePolicyV4, TextStatePolicyV2Args
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.training.transition_trainer import decode_ordered_events
from plot.world_memory import WorldMemory


KINDS = {"human_like": 0, "npc_villager": 1, "npc_zombie": 2, "npc_skeleton": 3}


def move(inputs, device):
    return {key: value[None].to(device) for key, value in inputs.items()}


def load_models(m2_path, m4_path, device, load_m4=True):
    m2_checkpoint = torch.load(m2_path, map_location="cpu", weights_only=True)
    cfg = dict(m2_checkpoint["config"]["model"])
    cfg["kinematics"] = KinematicsConfig(**cfg["kinematics"])
    m2 = TransitionNetwork(TransitionArgs(**cfg))
    m2.load_state_dict(m2_checkpoint["model"], strict=True)
    m2.to(device).eval()

    m4 = m4_checkpoint = None
    if load_m4:
        m4_checkpoint = torch.load(m4_path, map_location="cpu", weights_only=True)
        m4 = IndependentStatePolicyV4(TextStatePolicyV2Args(**m4_checkpoint["config"]))
        m4.load_state_dict(m4_checkpoint["model"], strict=True)
        m4.to(device).eval()
        if m2_checkpoint["config"]["class_to_raw"] != m4_checkpoint["dataset"]["class_to_raw"]:
            raise ValueError("M2 and M4 block vocabularies differ")
    return m2, m2_checkpoint, m4, m4_checkpoint


def load_initial_inputs(cache_path, cache_summary):
    cached = load_state_cache(cache_path)
    inputs = dict(cached["inputs"])
    summary = json.loads(cache_summary.read_text())
    text_cache = load_file(summary["text_cache"], device="cpu")
    for name in ("shared", "current"):
        text_id = int(inputs.pop(name + "_text_id"))
        inputs[name + "_text"] = text_cache["encoder_hidden"][text_id].float()
        inputs[name + "_text_mask"] = text_cache["attention_mask"][text_id].bool()
    return inputs


def build_m4_inputs(history, memory, target, fov, text_inputs):
    position = np.stack([row["position"] for row in history])
    angles = np.stack([row["angles"] for row in history])
    hp = np.stack([row["hp"] for row in history])
    camera_relative = np.stack([row["camera_relative"] for row in history])
    camera_direction = np.stack([row["camera_direction"] for row in history])
    event_cues = np.stack([row["event_cues"] for row in history])
    held_item = np.stack([row["held_item"] for row in history])
    resident_type = np.stack([row["resident_type"] for row in history])
    resident_valid = np.stack([row["resident_valid"] for row in history])
    resident_actions = np.stack([row["resident_actions"] for row in history])
    anchor = np.rint(position[-1, target]).astype(np.int64)
    blocks, known = memory.read_tile(anchor)
    result = assemble_state_inputs(
        blocks=blocks, known=known, anchor=anchor, position=position, angles=angles,
        hp=hp, camera_relative=camera_relative, camera_direction=camera_direction,
        event_cues=event_cues, held_item=held_item, resident_type=resident_type,
        resident_valid=resident_valid, incoming_actions=resident_actions[:, target],
        target=target, history_valid=np.ones(8, bool))
    camera_position = position[-1, target] + camera_relative[-1, target]
    result["raster_camera"] = torch.from_numpy(
        raster_camera(camera_position, camera_direction[-1, target], fov, anchor))
    result["resident_actions"] = torch.from_numpy(resident_actions.astype(np.float32))
    result.update(text_inputs)
    return result


def update_selected_slots(selected, actions):
    selected = np.asarray(selected, np.int64).copy()
    for step in range(actions.shape[0]):
        for slot in range(9):
            selected = np.where(actions[step, :, 12 + slot] > 0, slot, selected)
    return selected


def target_events(episode):
    result = []
    for line in (episode / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") != "block_placed" or event.get("placed_node") != "mcl_core:wood":
            continue
        xyz = event_position(event)
        result.append((int(event["transition_index"]), event.get("actor"), tuple(map(int, xyz))))
    return result


def score_world(memory, coordinates, wood_class):
    values = []
    for coordinate in coordinates:
        block, known = memory.read_region(np.asarray(coordinate), (1, 1, 1))
        values.append(bool(known[0, 0, 0] and int(block[0, 0, 0]) == wood_class))
    return sum(values), values


@torch.no_grad()
def run_mode(mode, samples, initial_inputs, text_inputs, target, m2, m4, device,
             expected, wood_class, max_blocks):
    first = samples[0]["inputs"]
    agents = int(first["active"].numel())
    pose = first["initial_pose"].numpy().copy()
    velocity = first.get("initial_velocity", torch.zeros_like(first["initial_pose"][..., :3])).numpy().copy()
    hp = first["initial_hp"].numpy().copy()
    camera_relative = first["camera_relative"].numpy().copy()
    camera_direction = first["camera_direction"].numpy().copy()
    held = first["held_item"].numpy().copy()
    resident_type = first["resident_type"].numpy().copy()
    selected_slot = first["selected_slot"].numpy().copy()
    hotbar = first["hotbar"].numpy().copy()

    memory = WorldMemory()
    initial_anchor = np.rint(pose[target, :3]).astype(np.int64)
    memory.commit_tile(initial_anchor, initial_inputs["voxel_classes"].numpy(),
                       initial_inputs["voxel_known"].numpy())
    history = None
    trace, writes = [], []
    total_place_actions = 0
    for block, sample in enumerate(samples[:max_blocks]):
        start = int(sample["metadata"]["start"])
        real = sample["inputs"]
        # All non-target residents are observations, not predictions.  This tests
        # one builder without compounding unrelated resident-model errors.
        other = np.arange(agents) != target
        pose[other] = real["initial_pose"].numpy()[other]
        real_velocity = real.get("initial_velocity", torch.zeros_like(real["initial_pose"][..., :3]))
        velocity[other] = real_velocity.numpy()[other]
        hp[other] = real["initial_hp"].numpy()[other]
        camera_relative[other] = real["camera_relative"].numpy()[other]
        camera_direction[other] = real["camera_direction"].numpy()[other]
        held[other] = real["held_item"].numpy()[other]
        hotbar[other] = real["hotbar"].numpy()[other]
        selected_slot[other] = real["selected_slot"].numpy()[other]

        joint_actions = real["actions"].numpy().copy()
        if mode == "m4":
            m4_inputs = initial_inputs if block == 0 else build_m4_inputs(
                history, memory, target, 1.2217304764, text_inputs)
            joint_actions[:, target] = m4.decode(m4(move(m4_inputs, device)))[0].cpu().numpy()
        elif mode == "zero":
            joint_actions[:, target] = 0
        elif mode != "real":
            raise ValueError(mode)
        total_place_actions += int((joint_actions[:, target, 9] > 0).sum())

        anchors = np.rint(pose[:, :3]).astype(np.int64)
        voxels, known, xyz = [], [], []
        offsets = np.stack(np.meshgrid(*([np.arange(13)] * 3), indexing="ij"), -1)
        for agent in range(agents):
            lower = anchors[agent] - 6
            block_ids, block_known = memory.read_region(lower, (13, 13, 13))
            voxels.append(block_ids); known.append(block_known)
            xyz.append(lower + offsets)
        inputs = dict(
            voxels=torch.from_numpy(np.stack(voxels)),
            voxel_known=torch.from_numpy(np.stack(known)),
            voxel_relative_xyz=torch.from_numpy(
                (np.stack(xyz).reshape(agents, VOXEL_COUNT, 3) - pose[:, None, :3]).astype(np.float32)),
            initial_pose=torch.from_numpy(pose), initial_hp=torch.from_numpy(hp),
            initial_velocity=torch.from_numpy(velocity),
            held_item=torch.from_numpy(held), resident_type=torch.from_numpy(resident_type),
            camera_relative=torch.from_numpy(camera_relative),
            camera_direction=torch.from_numpy(camera_direction),
            actions=torch.from_numpy(joint_actions.astype(np.float32)),
            active=real["active"], previous_rgb=real["previous_rgb"],
            hotbar=torch.from_numpy(hotbar), selected_slot=torch.from_numpy(selected_slot))
        output = m2(move(inputs, device))
        dense_address, dense_block, counts = decode_ordered_events(
            m2, output, VOXEL_COUNT + agents, 8)
        addresses = dense_address[0].cpu().numpy()
        blocks = dense_block[0].cpu().numpy()
        predicted_pose = output["pose"][0].float().cpu().numpy()
        predicted_camera = output["camera_relative"][0].float().cpu().numpy()
        predicted_direction = output["camera_direction"][0].float().cpu().numpy()
        predicted_held = output["held_logits"][0].argmax(-1).cpu().numpy()
        history = []
        writes_before = len(writes)
        for step in range(8):
            cues = np.zeros((agents, 4), np.float32)
            for source in range(agents):
                address = int(addresses[step, source])
                if address >= VOXEL_COUNT:
                    continue
                local = np.asarray(np.unravel_index(address, (13, 13, 13)))
                coordinate = anchors[source] - 6 + local
                block_class = int(blocks[step, source])
                memory.commit_points(coordinate[None], np.asarray([block_class]),
                                     allow_overwrite=True)
                cues[source, 0] += 1
                writes.append(dict(transition=start + step, source=source,
                                   coordinate=coordinate.tolist(), block_class=block_class))
            # Recorded states for residents that are not under test.
            next_pose = predicted_pose[step].copy()
            next_camera = predicted_camera[step].copy()
            next_direction = predicted_direction[step].copy()
            next_held = predicted_held[step].copy()
            real_target = sample["targets"]
            next_pose[other] = real_target["pose"][step].numpy()[other]
            next_camera[other] = real_target["camera_relative"][step].numpy()[other]
            next_direction[other] = real_target["camera_direction"][step].numpy()[other]
            next_held[other] = real_target["held_item"][step].numpy()[other]
            history.append(dict(
                position=next_pose[:, :3], angles=next_pose[:, 3:], hp=hp.copy(),
                camera_relative=next_camera, camera_direction=next_direction,
                event_cues=cues, held_item=next_held, resident_type=resident_type.copy(),
                resident_valid=np.ones(agents, bool), resident_actions=joint_actions[step]))
        pose = history[-1]["position"].copy()
        pose = np.concatenate((pose, history[-1]["angles"]), -1).astype(np.float32)
        velocity[target] = predicted_pose[-1, target, :3] - predicted_pose[-2, target, :3]
        camera_relative = history[-1]["camera_relative"].copy()
        camera_direction = history[-1]["camera_direction"].copy()
        held = history[-1]["held_item"].copy()
        selected_slot = update_selected_slots(selected_slot, joint_actions)
        complete, _ = score_world(memory, expected, wood_class)
        trace.append(dict(block=block, start=start, correct_targets=complete,
                          predicted_writes=len(writes) - writes_before,
                          target_place_actions=int((joint_actions[:, target, 9] > 0).sum()),
                          target_position=pose[target, :3].tolist()))
    correct, mask = score_world(memory, expected, wood_class)
    target_writes = [row for row in writes if row["source"] == target]
    correct_write_coordinates = {tuple(row["coordinate"]) for row in target_writes} & set(expected)
    return dict(mode=mode, blocks=len(trace), transitions=8 * len(trace),
                correct_targets=correct, target_count=len(expected), completion=correct / len(expected),
                target_place_actions=total_place_actions, predicted_writes=len(writes),
                target_predicted_writes=len(target_writes),
                target_correct_write_coordinates=len(correct_write_coordinates),
                target_mask=mask, trace=trace, writes=writes)


def infer_wood_class(samples, real_events):
    by_start = {int(sample["metadata"]["start"]): sample for sample in samples}
    classes = []
    for transition, actor, _ in real_events:
        start = transition - ((transition - min(by_start)) % 8)
        sample = by_start.get(start)
        if sample is None:
            continue
        source = int(actor.removeprefix("agent"))
        offset = transition - start
        if bool(sample["targets"]["block_valid"][offset, source]):
            classes.append(int(sample["targets"]["block"][offset, source]))
    if not classes:
        raise ValueError("could not infer wood block class from real placement labels")
    return Counter(classes).most_common(1)[0][0]


@torch.no_grad()
def teacher_forced_m2_diagnostic(samples, m2, device, expected, wood_class, target):
    """Decode every window from exact recorded M2 inputs (no rollout drift)."""
    expected_set = set(expected)
    predicted = []
    per_source = Counter()
    for sample in samples:
        output = m2(move(sample["inputs"], device))
        agents = int(sample["inputs"]["active"].numel())
        address, block, _ = decode_ordered_events(m2, output, VOXEL_COUNT + agents, 8)
        address = address[0].cpu().numpy()
        block = block[0].cpu().numpy()
        anchors = np.asarray(sample["metadata"]["anchors"], np.int64)
        start = int(sample["metadata"]["start"])
        for step in range(8):
            for source in range(agents):
                index = int(address[step, source])
                if index >= VOXEL_COUNT:
                    continue
                local = np.asarray(np.unravel_index(index, (13, 13, 13)))
                coordinate = tuple(map(int, anchors[source] - 6 + local))
                row = dict(transition=start + step, source=source,
                           coordinate=list(coordinate), block_class=int(block[step, source]))
                predicted.append(row)
                per_source[source] += 1
    correct = {tuple(row["coordinate"]) for row in predicted
               if row["block_class"] == wood_class and tuple(row["coordinate"]) in expected_set}
    target_correct = {tuple(row["coordinate"]) for row in predicted
                      if row["source"] == target and row["block_class"] == wood_class
                      and tuple(row["coordinate"]) in expected_set}
    return dict(predicted_writes=len(predicted), writes_by_source=dict(per_source),
                correct_target_coordinates=len(correct),
                selected_builder_correct_coordinates=len(target_correct),
                completion=len(correct) / len(expected), writes=predicted)


@torch.no_grad()
def gt_state_m4_diagnostic(episode, samples, vocabulary, item_vocabulary, model_start,
                           text_inputs, m2, m4, device, expected, wood_class, target,
                           max_blocks):
    """Run M4 actions through M2 while resetting all state to GT every block.

    M4 observes the exact recorded voxel/state history at each anchor.  M2 also
    receives its exact recorded input at that anchor, except that the selected
    builder's eight actions are replaced by M4's decoded actions.  No predicted
    pose, camera, or voxel write is fed into the next block.
    """
    expected_set = set(expected)
    writes = []
    trace = []
    total_place_actions = 0
    target_coordinates = set()
    correct_coordinates = set()
    writes_by_source = Counter()
    predicted_place_transitions = []
    recorded_place_transitions = []
    action_l1_sum = 0.0
    action_value_count = 0
    for block_index, sample in enumerate(samples[:max_blocks]):
        start = int(sample["metadata"]["start"])
        gt_m4 = read_state_sample(
            episode, start, target, vocabulary, model_start, item_vocabulary,
            include_m3_fields=True)["inputs"]
        gt_m4.update(text_inputs)
        actions = m4.decode(m4(move(gt_m4, device)))[0].cpu()
        total_place_actions += int((actions[:, 9] > 0).sum())

        m2_inputs = dict(sample["inputs"])
        joint_actions = m2_inputs["actions"].clone()
        recorded_actions = joint_actions[:, target].clone()
        predicted_place_transitions.extend(
            start + int(step) for step in torch.nonzero(actions[:, 9] > 0).flatten())
        recorded_place_transitions.extend(
            start + int(step) for step in torch.nonzero(recorded_actions[:, 9] > 0).flatten())
        action_l1_sum += float((actions - recorded_actions).abs().sum())
        action_value_count += actions.numel()
        joint_actions[:, target] = actions
        m2_inputs["actions"] = joint_actions
        output = m2(move(m2_inputs, device))
        agents = int(m2_inputs["active"].numel())
        address, block_class, _ = decode_ordered_events(
            m2, output, VOXEL_COUNT + agents, 8)
        address = address[0].cpu().numpy()
        block_class = block_class[0].cpu().numpy()
        anchors = np.asarray(sample["metadata"]["anchors"], np.int64)
        writes_before = len(writes)
        target_before = len(target_coordinates)
        correct_before = len(correct_coordinates)
        for step in range(8):
            for source in range(agents):
                index = int(address[step, source])
                if index >= VOXEL_COUNT:
                    continue
                local = np.asarray(np.unravel_index(index, (13, 13, 13)))
                coordinate = tuple(map(int, anchors[source] - 6 + local))
                predicted_class = int(block_class[step, source])
                writes.append(dict(transition=start + step, source=source,
                                   coordinate=list(coordinate),
                                   block_class=predicted_class))
                writes_by_source[source] += 1
                if source == target:
                    target_coordinates.add(coordinate)
                    if predicted_class == wood_class and coordinate in expected_set:
                        correct_coordinates.add(coordinate)
        trace.append(dict(
            block=block_index, start=start,
            target_place_actions=int((actions[:, 9] > 0).sum()),
            predicted_writes=len(writes) - writes_before,
            new_target_write_coordinates=len(target_coordinates) - target_before,
            new_correct_target_coordinates=len(correct_coordinates) - correct_before,
            cumulative_correct_targets=len(correct_coordinates)))

    target_writes = [row for row in writes if row["source"] == target]
    return dict(
        mode="m4_gt_state", blocks=len(trace), transitions=8 * len(trace),
        correct_targets=len(correct_coordinates), target_count=len(expected),
        completion=len(correct_coordinates) / len(expected),
        target_place_actions=total_place_actions, predicted_writes=len(writes),
        target_predicted_writes=len(target_writes),
        target_correct_write_coordinates=len(correct_coordinates),
        predicted_place_transitions=predicted_place_transitions,
        recorded_place_transitions=recorded_place_transitions,
        place_transition_overlap=len(set(predicted_place_transitions) &
                                     set(recorded_place_transitions)),
        mean_action_l1=action_l1_sum / action_value_count,
        writes_by_source=dict(writes_by_source), trace=trace, writes=writes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--initial-cache", type=Path, required=True)
    parser.add_argument("--cache-summary", type=Path, required=True)
    parser.add_argument("--m2-checkpoint", type=Path, required=True)
    parser.add_argument("--m4-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", type=int, default=0)
    parser.add_argument("--max-blocks", type=int, default=60)
    parser.add_argument("--modes", nargs="+", choices=("real", "zero", "m4", "m4_gt_state"),
                        default=("real", "zero", "m4"))
    parser.add_argument("--model-start", type=int, default=0,
                        help="First valid frame of the episode segment used by M4 history")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device)
    needs_m4 = any(mode.startswith("m4") for mode in args.modes)
    m2, m2_checkpoint, m4, m4_checkpoint = load_models(
        args.m2_checkpoint, args.m4_checkpoint, device, load_m4=needs_m4)
    cache_summary = json.loads(args.cache_summary.read_text())
    initial_inputs = load_initial_inputs(args.initial_cache, args.cache_summary)
    text_inputs = {key: initial_inputs[key] for key in (
        "shared_text", "shared_text_mask", "current_text", "current_text_mask")}
    vocabulary = BlockVocabulary(tuple(m2_checkpoint["config"]["class_to_raw"]))
    split = json.loads((args.episode / "manifest.json").read_text())["split"]
    samples, stats = episode_windows(
        args.episode, vocabulary, m2_checkpoint["config"]["items"],
        windows_per_episode=None, split=split, stride=8)
    first_start = int(samples[0]["metadata"]["start"])
    samples = [sample for sample in samples
               if (int(sample["metadata"]["start"]) - first_start) % 8 == 0]
    real_events = target_events(args.episode)
    expected = sorted({coordinate for _, _, coordinate in real_events})
    wood_class = infer_wood_class(samples, real_events)
    teacher_forced = teacher_forced_m2_diagnostic(
        samples, m2, device, expected, wood_class, args.target)
    results = []
    for mode in args.modes:
        if mode == "m4_gt_state":
            results.append(gt_state_m4_diagnostic(
                args.episode, samples, vocabulary, cache_summary["item_vocabulary"],
                args.model_start, text_inputs, m2, m4, device, expected, wood_class,
                args.target, args.max_blocks))
        else:
            results.append(run_mode(
                mode, samples, initial_inputs, text_inputs, args.target,
                m2, m4, device, expected, wood_class, args.max_blocks))
    payload = dict(
        schema="text-m2-closed-loop-v1", episode=args.episode.name,
        task="Fill the 3 by 3 hole with wooden planks.", target_agent=args.target,
        expected_coordinates=expected, wood_class=wood_class,
        semantics={
            "target_action": "recorded, all-zero, or TextAgent depending on mode",
            "other_actions": "recorded ground truth",
            "target_state": "M2 autoregressive after initial observation",
            "other_state": "recorded at every frame",
            "world": "M2 ordered voxel writes applied autoregressively",
            "rgb": "recorded RGB at each 8-transition boundary (teacher-forced visual context)",
            "m4_gt_state": (
                "M4 and M2 both reset to exact recorded state/world every 8 frames; "
                "only the selected builder actions are replaced by M4, and predicted "
                "state/writes never feed the next block")},
        cache_stats=dict(stats), teacher_forced_m2=teacher_forced,
        model=dict(m2=str(args.m2_checkpoint), m4=str(args.m4_checkpoint)),
        results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps([{key: row[key] for key in (
        "mode", "blocks", "correct_targets", "target_count", "completion",
        "target_place_actions", "predicted_writes", "target_predicted_writes",
        "target_correct_write_coordinates")} for row in results], indent=2))
    print(json.dumps({"teacher_forced_m2": {key: teacher_forced[key] for key in (
        "predicted_writes", "writes_by_source", "correct_target_coordinates",
        "selected_builder_correct_coordinates", "completion")}}, indent=2))


if __name__ == "__main__":
    main()
