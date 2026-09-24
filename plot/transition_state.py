"""Resident state and immutable event records for M2 commits."""
from dataclasses import dataclass

import torch


def held_item_trajectory(inputs):
    """Derive held items from the known hotbar and eight-step action block.

    The highest selected slot wins, matching TextAgent. Dropping an item makes
    the fixed-hotbar approximation invalid for all later steps.
    """
    actions = inputs["actions"]
    batch, steps, agents, _ = actions.shape
    slot = inputs["selected_slot"].clone()
    hotbar = inputs["hotbar"]
    valid = torch.ones(batch, agents, dtype=torch.bool, device=actions.device)
    items = []
    masks = []
    for step in range(steps):
        valid &= actions[:, step, :, 10] == 0
        for index in range(9):
            slot = torch.where(actions[:, step, :, 12 + index] > 0, index, slot)
        items.append(hotbar.gather(-1, slot[..., None]).squeeze(-1))
        masks.append(valid.clone())
    return torch.stack(items, 1), torch.stack(masks, 1)


@dataclass
class CharRow:
    resident_id: str
    slot_order: int
    position_xyz: tuple[float,float,float]
    yaw: float
    pitch: float
    hp: float
    held_item: int
    resident_type: int
    camera_relative: tuple[float,float,float]
    camera_direction: tuple[float,float,float]
    # Per-transition displacement in world XYZ.  Kept in the authoritative
    # resident row so M2 rollouts do not need to reconstruct motion history.
    velocity_xyz: tuple[float,float,float] = (0., 0., 0.)


@dataclass(frozen=True)
class WriteEvent:
    transition_index: int
    within_transition_order: int
    source: str
    target_kind: str
    target: tuple[int,int,int] | str
    payload: float | int
