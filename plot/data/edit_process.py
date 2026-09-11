"""Derive persistent edit attempts from recorded actions and interaction rays.

The engine-side dig timer was not stored in the released episodes.  This module
therefore exposes *observable* edit age rather than pretending to recover that
timer: consecutive actions on the same voxel accumulate frames and Minetest
time, and a one-transition pending state accounts for delayed server events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .transition_dataset import event_position, source_slot


DIG_ACTION = 8
PLACE_ACTION = 9
DIG_EVENTS = {"block_dug", "scaffold_dug"}
PLACE_EVENTS = {"block_placed", "scaffold_placed"}


@dataclass
class _Run:
    kind: int
    target: tuple[int, int, int]
    frames: int = 0
    seconds: float = 0.0
    last_action: int = -1


def derive_edit_process(
    actions: np.ndarray,
    pointed_type: np.ndarray,
    pointed_under: np.ndarray,
    pointed_above: np.ndarray,
    dt_minetest: np.ndarray,
    events_by_step: Mapping[int, Sequence[dict]],
    agents: Sequence[str],
) -> dict[str, np.ndarray]:
    """Return transition-aligned edit age, target and retrospective progress.

    ``kind`` is 0/1/2 for inactive/dig/place. ``age_*`` is available for every
    observed attempt. ``progress`` is supervised only for attempts that end in
    a matching world-edit event; failed or truncated attempts remain invalid.
    The target coordinates use the dataset's ENU convention.
    """
    actions = np.asarray(actions)
    transitions, residents = actions.shape[:2]
    if actions.shape != (transitions, residents, 23):
        raise ValueError("actions must have shape [T,A,23]")
    if len(agents) != residents:
        raise ValueError("agent count does not match action slots")
    if np.asarray(pointed_type).shape != (transitions + 1, residents):
        raise ValueError("pointed_type must have shape [T+1,A]")

    kind = np.zeros((transitions, residents), np.int8)
    target = np.zeros((transitions, residents, 3), np.int32)
    target_valid = np.zeros((transitions, residents), bool)
    age_frames = np.zeros((transitions, residents), np.int16)
    age_seconds = np.zeros((transitions, residents), np.float32)
    completed = np.zeros((transitions, residents), bool)
    progress = np.zeros((transitions, residents), np.float32)
    progress_valid = np.zeros((transitions, residents), bool)
    runs: list[_Run | None] = [None] * residents
    run_steps: list[list[int]] = [[] for _ in range(residents)]

    dt = np.asarray(dt_minetest)
    if dt.ndim == 1:
        dt = np.broadcast_to(dt[:, None], (transitions + 1, residents))
    if dt.shape != (transitions + 1, residents):
        raise ValueError("dt_minetest must have shape [T+1,A] or [T+1]")

    def matching_event(step: int, slot: int, run: _Run) -> bool:
        expected = DIG_EVENTS if run.kind == 1 else PLACE_EVENTS
        for event in events_by_step.get(step, ()):
            xyz = event_position(event)
            if (event.get("event") in expected and source_slot(event, agents) == slot
                    and xyz is not None and tuple(map(int, xyz)) == run.target):
                return True
        return False

    def finish(slot: int, success: bool) -> None:
        steps = run_steps[slot]
        if success and steps:
            action_steps = [step for step in steps
                            if actions[step, slot, DIG_ACTION] > .5
                            or actions[step, slot, PLACE_ACTION] > .5]
            denominator = max(1, len(action_steps))
            ordinal = 0
            for step in steps:
                if step in action_steps:
                    ordinal += 1
                progress[step, slot] = ordinal / denominator
                progress_valid[step, slot] = True
            completed[steps[-1], slot] = True
        runs[slot] = None
        run_steps[slot] = []

    for step in range(transitions):
        for slot in range(residents):
            dig = actions[step, slot, DIG_ACTION] > 0.5
            place = actions[step, slot, PLACE_ACTION] > 0.5
            action_kind = 1 if dig and not place else 2 if place and not dig else 0
            valid_ray = bool(pointed_type[step, slot] == 1)
            ray = pointed_under[step, slot] if action_kind == 1 else pointed_above[step, slot]
            ray_key = tuple(map(int, ray)) if action_kind and valid_ray else None
            run = runs[slot]

            continues = run is not None and action_kind == run.kind and ray_key == run.target
            if run is not None and not continues:
                # Server callbacks may arrive one transition after the final key.
                if step == run.last_action + 1 and matching_event(step, slot, run):
                    kind[step, slot] = run.kind
                    target[step, slot] = run.target
                    target_valid[step, slot] = True
                    age_frames[step, slot] = run.frames
                    age_seconds[step, slot] = run.seconds
                    run_steps[slot].append(step)
                    finish(slot, True)
                else:
                    finish(slot, False)
                run = runs[slot]

            if action_kind and valid_ray:
                if run is None:
                    run = _Run(action_kind, ray_key)
                    runs[slot] = run
                run.frames += 1
                run.seconds += max(0.0, float(dt[step + 1, slot]))
                run.last_action = step
                kind[step, slot] = run.kind
                target[step, slot] = run.target
                target_valid[step, slot] = True
                age_frames[step, slot] = run.frames
                age_seconds[step, slot] = run.seconds
                run_steps[slot].append(step)
                if matching_event(step, slot, run):
                    finish(slot, True)

    for slot in range(residents):
        finish(slot, False)
    return {
        "kind": kind,
        "target": target,
        "target_valid": target_valid,
        "age_frames": age_frames,
        "age_seconds": age_seconds,
        "completed": completed,
        "progress": progress,
        "progress_valid": progress_valid,
    }
