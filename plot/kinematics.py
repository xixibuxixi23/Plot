"""Action-only coarse trajectories; no collision or combat rules."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KinematicsConfig:
    # Pilot calibration knobs, in world units and radians per control step.
    distance_per_step: float = 0.4
    radians_per_mouse_unit: float = 1.0


def wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def propose_trajectory(initial_pose, actions, cfg=KinematicsConfig()):
    """[B,A,5], [B,8,A,23] -> [B,8,A,5], relative xyz and absolute angles.

    TextAgent ENU yaw: 0 faces north (+Y); positive yaw rotates toward +X. Data mouse
    signs and gain are explicit configuration, not estimated from future poses.
    Vertical movement is left to the learned residual in this first pilot.
    """
    if actions.shape[1] != 8 or actions.shape[-1] != 23:
        raise ValueError("M2 expects eight continuous 23-D actions")
    angle_delta = actions[..., 21:23] * cfg.radians_per_mouse_unit
    angles = initial_pose[:, None, :, 3:5] + angle_delta.cumsum(1)
    yaw = angles[..., 0]
    forward = actions[..., 0] - actions[..., 1]
    right = actions[..., 3] - actions[..., 2]
    # Clamp the squared norm before sqrt.  Clamping after sqrt has the same
    # forward value, but sqrt(0)'s infinite derivative turns the zero gradient
    # from clamp_min into NaN when a world-model loss differentiates actions.
    norm = (forward.square() + right.square()).clamp_min(1.).sqrt()
    dx = (yaw.sin()*forward + yaw.cos()*right) / norm
    dy = (yaw.cos()*forward - yaw.sin()*right) / norm
    delta = torch.stack((dx,dy,torch.zeros_like(dx)), -1) * cfg.distance_per_step
    return torch.cat((delta.cumsum(1), wrap_angle(angles)), -1)
