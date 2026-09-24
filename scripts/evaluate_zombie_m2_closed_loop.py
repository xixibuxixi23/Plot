"""Closed-loop M2 combat evaluation with real human and generated zombie actions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data.renderer_dataset import raster_camera
from plot.data.player_transition_stream import episode_player_windows
from plot.data.state_policy_dataset import assemble_state_inputs, load_state_cache
from plot.kinematics import KinematicsConfig
from plot.models.state_policy_large_v2 import (
    ZombieStatePolicyV2, ZombieStatePolicyV2Args,
    ZombieStatePolicyV3, ZombieStatePolicyV3Args,
)
from plot.models.state_policy_text_v2 import (
    IndependentStatePolicyV4, TextStatePolicyV2Args,
    UnifiedStatePolicyV4, UnifiedStatePolicyV4Args,
)
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.world_memory import WorldMemory


KINDS = {"human_like": 0, "npc_villager": 1, "npc_zombie": 2, "npc_skeleton": 3}


def move(inputs, device):
    return {key: value[None].to(device) for key, value in inputs.items()}


def remap_local_items(values, local_vocabulary, target_vocabulary):
    inverse = {int(index): name for name, index in local_vocabulary.items()}
    flat = np.asarray(values).reshape(-1)
    mapped = [target_vocabulary.get(inverse.get(int(value), ""), 0) for value in flat]
    return np.asarray(mapped, np.int64).reshape(np.asarray(values).shape)


def update_selected_slots(selected, actions):
    selected = np.asarray(selected, np.int64).copy()
    for step in range(actions.shape[0]):
        for slot in range(9):
            selected = np.where(actions[step, :, 12 + slot] > 0, slot, selected)
    return selected


def load_models(m2_path, m4_path, device):
    m2_checkpoint = torch.load(m2_path, map_location="cpu", weights_only=True)
    m2_cfg = dict(m2_checkpoint["config"]["model"])
    m2_cfg["kinematics"] = KinematicsConfig(**m2_cfg["kinematics"])
    m2 = TransitionNetwork(TransitionArgs(**m2_cfg))
    m2.load_state_dict(m2_checkpoint["model"], strict=True)
    m2 = m2.to(device).eval()

    m4_checkpoint = torch.load(m4_path, map_location="cpu", weights_only=True)
    if m4_checkpoint["config"].get("profile") == "unified":
        m4_cfg = UnifiedStatePolicyV4Args(**m4_checkpoint["config"])
        m4 = UnifiedStatePolicyV4(m4_cfg)
    elif "text_projection.weight" in m4_checkpoint["model"]:
        m4_cfg = TextStatePolicyV2Args(**m4_checkpoint["config"])
        m4 = IndependentStatePolicyV4(m4_cfg)
    elif "attack_range" in m4_checkpoint["config"]:
        m4_cfg = ZombieStatePolicyV3Args(**m4_checkpoint["config"])
        m4 = ZombieStatePolicyV3(m4_cfg)
    else:
        m4_cfg = ZombieStatePolicyV2Args(**m4_checkpoint["config"])
        m4 = ZombieStatePolicyV2(m4_cfg)
    m4.load_state_dict(m4_checkpoint["model"], strict=True)
    m4 = m4.to(device).eval()
    return m2, m2_checkpoint, m4, m4_checkpoint


def build_m4_inputs(history, memory, target, fov):
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
    return result


def decoded_attacks(m2, output):
    decoded = m2.decode_attacks(output)
    rows = []
    for source in range(decoded["valid"].shape[1]):
        for query in range(decoded["valid"].shape[2]):
            if not bool(decoded["valid"][0, source, query]):
                continue
            rows.append(dict(
                source=source,
                target=int(decoded["target"][0, source, query]),
                time=int(decoded["time"][0, source, query]),
                raw_damage=float(decoded["damage"][0, source, query])))
    return sorted(rows, key=lambda row: (row["time"], row["source"]))


@torch.no_grad()
def teacher_forced_attack_diagnostic(m2, m2_checkpoint, episode, start, max_blocks, device):
    split = json.loads((episode / "manifest.json").read_text())["split"]
    totals = {name: [0, 0, 0] for name in ("human", "zombie")}
    windows = 0
    for sample in episode_player_windows(
            episode, m2_checkpoint["config"]["items"], split, stride=8):
        frame = int(sample["metadata"]["start"])
        if frame < start:
            continue
        if windows >= max_blocks:
            break
        predicted = decoded_attacks(m2, m2.forward_player(move(sample["inputs"], device)))
        targets = sample["targets"]
        for source, name in enumerate(("human", "zombie")):
            labels = [
                (int(targets["attack_time"][source, query]),
                 int(targets["attack_target"][source, query]))
                for query in range(targets["attack_slot_valid"].shape[1])
                if bool(targets["attack_slot_valid"][source, query])]
            guesses = [(row["time"], row["target"]) for row in predicted
                       if row["source"] == source]
            used = set()
            matched = 0
            for label_time, label_target in labels:
                candidates = [index for index, (guess_time, guess_target) in enumerate(guesses)
                              if index not in used and guess_target == label_target
                              and abs(guess_time - label_time) <= 1]
                if candidates:
                    chosen = min(candidates, key=lambda index: abs(guesses[index][0] - label_time))
                    used.add(chosen)
                    matched += 1
            totals[name][0] += matched
            totals[name][1] += len(guesses)
            totals[name][2] += len(labels)
        windows += 1
    result = {"windows": windows}
    for name, (matched, predicted, labels) in totals.items():
        precision = matched / predicted if predicted else 0.0
        recall = matched / labels if labels else 0.0
        result[name] = dict(
            matched=matched, predicted=predicted, labels=labels,
            precision=precision, recall=recall,
            f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return result


@torch.no_grad()
def run_mode(*, mode, episode, initial_cache, start, m2, m2_checkpoint,
             m4, m4_checkpoint, device, max_blocks, anchor_human_state,
             anchor_human_every_frame):
    manifest = json.loads((episode / "manifest.json").read_text())
    metadata = json.loads((episode / "training_metadata.json").read_text())
    with np.load(episode / manifest.get("training_data_file", "data.npz"), allow_pickle=False) as stored:
        actions = stored["action_continuous"].astype(np.float32)
        position = stored["player_pos"].astype(np.float32)
        yaw = stored["player_yaw"].astype(np.float32)
        pitch = stored["player_pitch"].astype(np.float32)
        health = stored["player_health"].astype(np.float32)
        camera_position = stored["cam_pos"].astype(np.float32)
        camera_direction = stored["cam_dir"].astype(np.float32)
        wielded = stored["wielded_item_id"].astype(np.int64)
        hotbar = stored["inventory_item_ids"].astype(np.int64)
        selected_slot = stored["selected_slot"].astype(np.int64)
        fov_value = stored["fov_x"][start]
        fov = float(fov_value if np.ndim(fov_value) == 0 else fov_value[1])

    agents = int(manifest["num_agents"])
    if agents != 2 or manifest["agent_kinds"].get("agent1") != "npc_zombie":
        raise ValueError("evaluation requires agent0 human and agent1 zombie")
    types = np.asarray([KINDS[manifest["agent_kinds"][f"agent{i}"]] for i in range(agents)])
    pose = np.concatenate((
        position[start], np.deg2rad(np.stack((yaw[start], pitch[start]), -1))), -1).astype(np.float32)
    hp = health[start].copy()
    camera_relative = (camera_position[start] - position[start]).astype(np.float32)
    direction = camera_direction[start].copy()

    local_items = metadata["item_vocabulary"]
    m2_items = m2_checkpoint["config"]["items"]
    m4_dataset = m4_checkpoint["dataset"]
    if m4_checkpoint["config"].get("profile") == "unified":
        m4_dataset = m4_dataset["zombie_melee"]
    m4_items = m4_dataset["item_vocabulary"]
    m2_inverse = {int(index): name for name, index in m2_items.items()}
    held_m2 = remap_local_items(wielded[start, :agents], local_items, m2_items)
    held_m2_all = remap_local_items(wielded[:, :agents], local_items, m2_items)
    held_m4_all = remap_local_items(wielded[:, :agents], local_items, m4_items)
    current_hotbar = remap_local_items(hotbar[start, :agents], local_items, m2_items)
    current_slot = selected_slot[start, :agents].copy()

    cached = load_state_cache(initial_cache)
    initial_inputs = cached["inputs"]
    if int(initial_inputs["target_agent"]) != 1:
        raise ValueError("initial cache must target agent1")
    memory = WorldMemory()
    initial_anchor = np.rint(position[start, 1]).astype(np.int64)
    memory.commit_tile(
        initial_anchor, initial_inputs["voxel_classes"].numpy(),
        initial_inputs["voxel_known"].numpy())

    history = None
    events = []
    trace = []
    terminated = None
    final_step = start
    block_limit = min(max_blocks, (len(actions) - start) // 8)
    for block in range(block_limit):
        transition = start + block * 8
        if (anchor_human_state or anchor_human_every_frame) and block > 0:
            history_frames = range(transition - 7, transition + 1)
            for row, frame in zip(history, history_frames):
                row["position"][0] = position[frame, 0]
                row["angles"][0] = np.deg2rad([yaw[frame, 0], pitch[frame, 0]])
                row["camera_relative"][0] = camera_position[frame, 0] - position[frame, 0]
                row["camera_direction"][0] = camera_direction[frame, 0]
                row["held_item"][0] = held_m4_all[frame, 0]
                row["resident_valid"][0] = hp[0] > 0
            pose[0, :3] = position[transition, 0]
            pose[0, 3:] = np.deg2rad([yaw[transition, 0], pitch[transition, 0]])
            camera_relative[0] = camera_position[transition, 0] - position[transition, 0]
            direction[0] = camera_direction[transition, 0]
            held_m2[0] = held_m2_all[transition, 0]
            current_slot[0] = selected_slot[transition, 0]
            current_hotbar[0] = remap_local_items(
                hotbar[transition, 0], local_items, m2_items)
        if mode == "m4":
            m4_inputs = initial_inputs if block == 0 else build_m4_inputs(history, memory, 1, fov)
            zombie_actions = m4.decode(m4(move(m4_inputs, device)))[0].float().cpu().numpy()
        elif mode == "real":
            zombie_actions = actions[transition:transition + 8, 1].copy()
        elif mode == "zero":
            zombie_actions = np.zeros((8, 23), np.float32)
        else:
            raise ValueError(mode)

        joint_actions = np.zeros((8, 2, 23), np.float32)
        joint_actions[:, 0] = actions[transition:transition + 8, 0]
        joint_actions[:, 1] = zombie_actions
        inputs = dict(
            initial_pose=torch.from_numpy(pose), initial_hp=torch.from_numpy(hp),
            held_item=torch.from_numpy(held_m2), resident_type=torch.from_numpy(types),
            camera_relative=torch.from_numpy(camera_relative),
            camera_direction=torch.from_numpy(direction), actions=torch.from_numpy(joint_actions),
            active=torch.from_numpy(hp > 0), hotbar=torch.from_numpy(current_hotbar),
            selected_slot=torch.from_numpy(current_slot))
        batch = move(inputs, device)
        output = m2.forward_player(batch)
        attacks = decoded_attacks(m2, output)
        predicted_pose = output["pose"][0].float().cpu().numpy()
        predicted_camera = output["camera_relative"][0].float().cpu().numpy()
        predicted_direction = output["camera_direction"][0].float().cpu().numpy()
        predicted_held = output["held_logits"][0].argmax(-1).cpu().numpy()

        # Keep agent0 on the recorded trajectory at every observation while agent1
        # remains fully autoregressive. M2 still evaluates each eight-action window,
        # but M4 and the following M2 window see exact human state on every frame.
        if anchor_human_every_frame:
            frames = np.arange(transition + 1, transition + 9)
            predicted_pose[:, 0, :3] = position[frames, 0]
            predicted_pose[:, 0, 3] = np.deg2rad(yaw[frames, 0])
            predicted_pose[:, 0, 4] = np.deg2rad(pitch[frames, 0])
            predicted_camera[:, 0] = camera_position[frames, 0] - position[frames, 0]
            predicted_direction[:, 0] = camera_direction[frames, 0]
            predicted_held[:, 0] = held_m2_all[frames, 0]

        hp_before = hp.copy()
        history = []
        for step in range(8):
            cues = np.zeros((2, 4), np.float32)
            previous_hp = hp.copy()
            for event in [row for row in attacks if row["time"] == step]:
                source, target = event["source"], event["target"]
                if source == target or hp[source] <= 0 or hp[target] <= 0:
                    event["applied_damage"] = 0.0
                    continue
                damage = float(np.clip(event["raw_damage"], 0.0, 20.0))
                applied = min(float(hp[target]), damage)
                hp[target] -= applied
                event["applied_damage"] = applied
                cues[source, 1] += 1
                cues[target, 2] += 1
                events.append(dict(block=block, transition=transition + step, **event))
            cues[:, 3] = hp - previous_hp
            held_names = [m2_inverse.get(int(value), "") for value in predicted_held[step]]
            held_m4 = np.asarray([m4_items.get(name, 0) for name in held_names], np.int64)
            history.append(dict(
                position=predicted_pose[step, :, :3].copy(),
                angles=predicted_pose[step, :, 3:].copy(), hp=hp.copy(),
                camera_relative=predicted_camera[step].copy(),
                camera_direction=predicted_direction[step].copy(), event_cues=cues,
                held_item=held_m4, resident_type=types.copy(), resident_valid=hp > 0,
                resident_actions=joint_actions[step].copy()))
            final_step = transition + step + 1
            if np.any(hp <= 0):
                terminated = int(np.flatnonzero(hp <= 0)[0])
                break

        pose = predicted_pose[min(7, final_step - transition - 1)].copy()
        camera_relative = predicted_camera[min(7, final_step - transition - 1)].copy()
        direction = predicted_direction[min(7, final_step - transition - 1)].copy()
        held_m2 = predicted_held[min(7, final_step - transition - 1)].copy()
        current_slot = update_selected_slots(current_slot, joint_actions)
        if anchor_human_every_frame:
            current_slot[0] = selected_slot[final_step, 0]
            current_hotbar[0] = remap_local_items(
                hotbar[final_step, 0], local_items, m2_items)
        distance = float(np.linalg.norm(pose[0, :3] - pose[1, :3]))
        trace.append(dict(
            block=block, start_transition=transition, end_transition=final_step,
            hp_before=hp_before.tolist(), hp_after=hp.tolist(), distance=distance,
            human_attack_actions=int((joint_actions[:, 0, 8] > 0).sum()),
            zombie_attack_actions=int((joint_actions[:, 1, 8] > 0).sum()),
            predicted_attacks=sum(1 for row in attacks if row.get("applied_damage", 0) > 0)))
        if terminated is not None:
            break

    return dict(
        mode=mode, episode=episode.name, start=start, final_transition=final_step,
        anchor_human_state=anchor_human_state,
        anchor_human_every_frame=anchor_human_every_frame,
        blocks=len(trace), initial_hp=health[start].tolist(), final_hp=hp.tolist(),
        dead_agent=terminated, dead_name=None if terminated is None else f"agent{terminated}",
        total_attacks=len(events), human_damage=float(sum(
            row["applied_damage"] for row in events if row["source"] == 0)),
        zombie_damage=float(sum(
            row["applied_damage"] for row in events if row["source"] == 1)),
        trace=trace, events=events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--initial-cache", type=Path, required=True)
    parser.add_argument("--m2-checkpoint", type=Path, required=True)
    parser.add_argument("--m4-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=19)
    parser.add_argument("--max-blocks", type=int, default=25)
    parser.add_argument("--modes", nargs="+", choices=("real", "zero", "m4"),
                        default=("real", "zero", "m4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-human-state", action="store_true",
                        help="replace human pose/camera/held state with recorded data each block")
    parser.add_argument("--anchor-human-every-frame", action="store_true",
                        help="expose recorded human state to M4 on every frame")
    args = parser.parse_args()
    device = torch.device(args.device)
    m2, m2_checkpoint, m4, m4_checkpoint = load_models(
        args.m2_checkpoint, args.m4_checkpoint, device)
    results = [run_mode(
        mode=mode, episode=args.episode, initial_cache=args.initial_cache,
        start=args.start, m2=m2, m2_checkpoint=m2_checkpoint, m4=m4,
        m4_checkpoint=m4_checkpoint, device=device, max_blocks=args.max_blocks,
        anchor_human_state=args.anchor_human_state,
        anchor_human_every_frame=args.anchor_human_every_frame)
        for mode in args.modes]
    teacher_forced = teacher_forced_attack_diagnostic(
        m2, m2_checkpoint, args.episode, args.start, args.max_blocks, device)
    payload = dict(
        schema="zombie-m2-closed-loop-v1",
        semantics={
            "human_action": "recorded ground truth for every transition",
            "zombie_action": "selected by mode; m4 and zero never read recorded zombie action",
            "state": ("M2 zombie state is autoregressive; human pose/camera/held are recorded every frame"
                      if args.anchor_human_every_frame else
                      "M2 autoregressive; human pose/camera/held are recorded at block boundaries"
                      if args.anchor_human_state else
                      "M2 player branch autoregressive after the initial observation"),
            "hp": "only decoded M2 ordered attack events change HP",
            "world": "static initial M4 voxel memory; selected S08 has no block edits"},
        model=dict(m2=str(args.m2_checkpoint), m4=str(args.m4_checkpoint)),
        teacher_forced_attack=teacher_forced,
        results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps([{key: row[key] for key in (
        "mode", "blocks", "final_transition", "final_hp", "dead_name",
        "total_attacks", "human_damage", "zombie_damage")} for row in results], indent=2))
    print(json.dumps({"teacher_forced_attack": teacher_forced}, indent=2))


if __name__ == "__main__":
    main()
