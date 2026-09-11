"""Executable eight-step M4 -> M2 -> memory/M1 -> M3 closed loop."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from plot.pipelines.renderer_pipeline import RendererMemoryBlock


@dataclass(frozen=True)
class ClosedLoopBlock:
    actions: torch.Tensor
    latents: torch.Tensor
    rgb: torch.Tensor
    snapshots: tuple
    events: tuple


@torch.no_grad()
def decode_transition(model, inputs, occurrence_threshold=0.5):
    """Decode the typed sparse M2 distribution into committer tensors."""
    output = model(inputs)
    nonnull = output["address_logits"][..., :-1].argmax(-1)
    null = output["address_logits"].shape[-1] - 1
    address = torch.where(
        output["occurrence_logits"].sigmoid() >= occurrence_threshold,
        nonnull, torch.full_like(nonnull, null),
    )
    block, damage = model.payloads(output, address)
    return {
        "pose": output["pose"], "address": address,
        "block_payload": block.argmax(-1), "hp_payload": damage,
        "held_item": output["held_logits"].argmax(-1),
        "camera_relative": output["camera_relative"],
        "camera_direction": output["camera_direction"],
    }


def _numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def build_renderer_conditions(*, resident_ids, snapshots, events, actions, geometry,
                              fov_x, skins, appearance_valid, device, initial_hp=None):
    """Assemble one shared-state M3 batch with one target view per resident."""
    a, t = len(resident_ids), 8
    if len(snapshots) != t or len(events) != t or len(geometry) != a:
        raise ValueError("M3 requires eight committed snapshots and one crop per target")
    rows = [[snapshot[name] for name in resident_ids] for snapshot in snapshots]
    position = np.asarray([[r.position_xyz for r in step] for step in rows], np.float32)
    camera_relative = np.asarray([[r.camera_relative for r in step] for step in rows], np.float32)
    camera_direction = np.asarray([[r.camera_direction for r in step] for step in rows], np.float32)
    hp = np.asarray([[r.hp for r in step] for step in rows], np.float32)
    angles = np.asarray([[[r.yaw, r.pitch] for r in step] for step in rows], np.float32)
    held = np.asarray([[r.held_item for r in step] for step in rows], np.int64)
    kinds = np.asarray([[r.resident_type for r in step] for step in rows], np.int64)
    cues = np.zeros((t, a, 4), np.float32)
    previous_hp = None if initial_hp is None else np.asarray(initial_hp, np.float32)
    for step, current_events in enumerate(events):
        for event in current_events:
            source = resident_ids.index(event.source)
            if event.target_kind == "voxel":
                cues[step, source, 0] += 1
            else:
                cues[step, source, 1] += 1
                cues[step, resident_ids.index(event.target), 2] += 1
        current_hp = hp[step]
        if previous_hp is not None:
            cues[step, :, 3] = current_hp - previous_hp
        previous_hp = current_hp
    fov = np.asarray(fov_x, np.float32)
    if fov.shape == (a,):
        fov = np.broadcast_to(fov, (t, a)).copy()
    if fov.shape != (t, a):
        raise ValueError("fov_x must be [A] or [8,A]")
    shared = {
        "player_position": position, "camera_relative": camera_relative,
        "camera_direction": camera_direction, "fov_x": fov, "hp": hp,
        "yaw_pitch": angles, "held_item": held, "resident_type": kinds,
        "event_cues": cues, "action": _numpy(actions),
        "player_valid": np.ones((t, a), bool),
    }
    batch = a
    result = {
        key: torch.as_tensor(np.broadcast_to(value, (batch, *value.shape)).copy(), device=device)
        for key, value in shared.items()
    }
    result.update(
        voxel_classes=torch.stack([g["voxel_classes"][0] for g in geometry]).to(device),
        voxel_known=torch.stack([g["voxel_known"][0] for g in geometry]).to(device),
        raster_camera=torch.stack([g["raster_camera"][0] for g in geometry]).to(device),
        target_agent=torch.arange(a, device=device),
        player_skin=torch.as_tensor(skins, device=device)[None].expand(batch, -1, -1, -1, -1, -1),
        player_appearance_valid=torch.as_tensor(appearance_valid, device=device)[None].expand(batch, -1, -1),
        condition_mask=torch.zeros(batch, t, dtype=torch.bool, device=device),
        action_prefix_mask=torch.zeros(batch, t, dtype=torch.bool, device=device),
    )
    return result


class ClosedLoopPipeline:
    """Own the authoritative order of one generated eight-transition block.

    Renderer cache is batched across residents because one shared Renderer has
    one cache object. ``fill_pipeline`` is invoked after every M2 commit, before
    the corresponding M3 memory snapshot is captured.
    """

    def __init__(self, *, policy, transition, committer, renderer_rollout, codec,
                 fill_pipeline, resident_ids, skins, appearance_valid, fov_x, device="cuda"):
        self.policy, self.transition, self.committer = policy, transition, committer
        self.renderer_rollout, self.codec, self.fill_pipeline = renderer_rollout, codec, fill_pipeline
        self.resident_ids = tuple(resident_ids)
        self.skins, self.appearance_valid = skins, appearance_valid
        self.fov_x = np.asarray(fov_x, np.float32)
        if self.fov_x.shape != (len(self.resident_ids),):
            raise ValueError("closed-loop fov_x must contain one value per resident")
        self.device = torch.device(device)
        self.policy_layers = None
        self.policy_conditions = None
        self.latest_rgb = None

    def prime(self, *, first_latents, first_conditions, action_history, latest_rgb):
        """Prime M3; one external frame implies one external bootstrap chunk."""
        condition = {key: value.to(self.device) if torch.is_tensor(value) else value
                     for key, value in first_conditions.items()}
        self.renderer_rollout.start(first_latents.to(self.device), condition)
        history = torch.as_tensor(action_history, dtype=torch.float32, device=self.device)
        if history.shape != (len(self.resident_ids), 8, 23):
            raise ValueError("bootstrap action chunk must be [A,8,23]")
        if first_latents.shape[1] != 1:
            raise ValueError("closed-loop initialization accepts exactly one external frame")
        self.latest_rgb = torch.as_tensor(latest_rgb, device=self.device)

    @torch.no_grad()
    def plan_actions(self, policy_state, external_actions, controlled):
        external = torch.as_tensor(external_actions, dtype=torch.float32, device=self.device)
        a = len(self.resident_ids)
        if external.shape != (a, 8, 23):
            raise ValueError("external actions must be [A,8,23]")
        controlled = torch.as_tensor(controlled, dtype=torch.bool, device=self.device)
        if controlled.shape != (a,):
            raise ValueError("controlled must be [A]")
        if not controlled.any() or self.policy_layers is None:
            return external
        policy_state = {key: value.to(self.device) if torch.is_tensor(value) else value
                        for key, value in policy_state.items()}
        ids = controlled.nonzero().flatten()
        condition = {key: value[ids] if torch.is_tensor(value) and value.shape[:1] == (a,) else value
                     for key, value in self.policy_conditions.items()}
        # Reindex target slots because selecting batch rows does not change the
        # resident axis inside shared conditions.
        condition["target_agent"] = self.policy_conditions["target_agent"][ids]
        _, logits = self.policy(
            None, condition, policy_state["family_id"][ids],
            policy_state["profile_id"][ids],
            shared_text=policy_state["shared_text"][ids],
            shared_text_mask=policy_state["shared_text_mask"][ids],
            current_text=policy_state["current_text"][ids],
            current_text_mask=policy_state["current_text_mask"][ids],
            video_layers=tuple(layer[ids] for layer in self.policy_layers))
        result = external.clone()
        result[ids] = self.policy.decode(logits)
        return result

    @torch.no_grad()
    def run_block(self, *, policy_state, external_actions, controlled,
                  transition_inputs, anchors):
        actions = self.plan_actions(policy_state, external_actions, controlled)
        inputs = {key: value.to(self.device) if torch.is_tensor(value) else value
                  for key, value in transition_inputs.items()}
        inputs["actions"] = actions.transpose(0, 1)[None]
        decoded = decode_transition(self.transition, inputs)
        decoded = {key: _numpy(value[0]) for key, value in decoded.items()}
        anchors = np.asarray(anchors, np.int64)
        initial_hp = [self.committer.chars[name].hp for name in self.resident_ids]
        geometry = [RendererMemoryBlock(anchor) for anchor in anchors]
        step_events, snapshots = [], []

        def after_step(index, memory, chars, events):
            centers = np.rint(np.asarray([chars[name].position_xyz for name in self.resident_ids])).astype(np.int64)
            self.fill_pipeline.fill_resident_windows(centers, self.latest_rgb)
            for target, block in enumerate(geometry):
                row = chars[self.resident_ids[target]]
                block.append(
                    memory, transition_index=index,
                    camera_world=np.asarray(row.position_xyz) + np.asarray(row.camera_relative),
                    camera_direction=row.camera_direction, fov_x=np.asarray(self.fov_x)[target],
                )
            snapshots.append(chars); step_events.append(events)

        self.committer.commit(
            resident_ids=self.resident_ids, anchors=anchors, after_step=after_step, **decoded)
        condition = build_renderer_conditions(
            resident_ids=self.resident_ids, snapshots=snapshots, events=step_events,
            actions=actions.transpose(0, 1), geometry=[g.conditions(device=self.device) for g in geometry],
            fov_x=self.fov_x, skins=self.skins, appearance_valid=self.appearance_valid,
            device=self.device, initial_hp=initial_hp,
        )
        cfg = self.renderer_rollout.model.cfg
        noise = torch.randn(len(self.resident_ids), 8, cfg.in_channels, cfg.input_h, cfg.input_w,
                            device=self.device)
        latent = self.renderer_rollout.generate(noise, condition)
        rgb = self.codec.decode(latent)
        completed = dict(condition)
        completed["condition_mask"] = torch.ones(
            latent.shape[:2], dtype=torch.bool, device=self.device)
        completed["action_prefix_mask"] = torch.zeros_like(completed["condition_mask"])
        layers = self.renderer_rollout.last_policy_layers
        if layers is None:
            raise RuntimeError("M3 rollout did not expose its final insertion layers")
        self.policy_layers, self.policy_conditions = layers, completed
        self.latest_rgb = rgb[:, -1]
        return ClosedLoopBlock(actions, latent, rgb, tuple(snapshots), tuple(step_events))
