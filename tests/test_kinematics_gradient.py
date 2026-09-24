import torch

from plot.kinematics import propose_trajectory


def test_zero_movement_actions_have_finite_gradients():
    pose = torch.zeros(1, 2, 5)
    actions = torch.zeros(1, 8, 2, 23, requires_grad=True)
    trajectory = propose_trajectory(pose, actions)
    trajectory.square().sum().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
