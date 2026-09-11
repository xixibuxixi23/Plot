import torch

from plot.models.visible_supervision import visible_voxel_masks
from plot.training.geometry_fill_trainer import GeometryFillTrainerConfig, geometry_fill_loss


def test_masks_deduplicate_and_exclude_hidden_wall():
    target = torch.zeros(1, 48, 48, 48, dtype=torch.long)
    target[:, 28:29] = 1
    target[:, 35:36] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    pos = torch.zeros(1, 2, 3)
    direction = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    fov = torch.full((1, 2), 0.3)
    single = visible_voxel_masks(
        target, valid, pos, direction, torch.tensor([[True, False]]), fov, fov, 0, height=8, width=8
    )
    double = visible_voxel_masks(
        target,
        valid,
        pos,
        direction,
        torch.ones(1, 2, dtype=torch.bool),
        fov,
        fov,
        0,
        height=8,
        width=8,
    )
    assert all(torch.equal(a, b) for a, b in zip(single, double))
    assert double[0].sum() > 0 and double[1].sum() > 0
    assert not double[0][:, 35:].any()


def test_invisible_logits_receive_no_supervision_gradient():
    shape = (1, 4, 4, 4)
    target = torch.zeros(shape, dtype=torch.long)
    target[:, 2, 2, 2] = 1
    surface = target.bool()
    free = torch.zeros_like(surface)
    free[:, 1, 2, 2] = True
    logits = torch.randn(1, 3, 4, 4, 4, requires_grad=True)
    occupancy = torch.randn(1, 1, 4, 4, 4, requires_grad=True)
    pos = torch.zeros(1, 1, 3)
    direction = torch.tensor([[[1.0, 0.0, 0.0]]])
    out = dict(
        voxel_logits=logits,
        occupancy_logits=occupancy,
        surface_support=torch.ones_like(occupancy),
        camera_position=pos,
        camera_direction=direction,
    )
    batch = dict(
        target=target,
        fill_mask=torch.ones_like(surface),
        target_valid=torch.ones_like(surface),
        visible_surface_mask=surface,
        visible_free_mask=free,
        camera_position=pos,
        camera_direction=direction,
        camera_valid=torch.ones(1, 1, dtype=torch.bool),
        agent_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    loss, _ = geometry_fill_loss(out, batch, GeometryFillTrainerConfig(visible_only=True))
    loss.backward()
    assert occupancy.grad[:, 0][~(surface | free)].abs().sum() == 0
    assert logits.grad.permute(0, 2, 3, 4, 1)[~surface].abs().sum() == 0
    assert occupancy.grad.abs().sum() > 0


def test_ray_weights_normalize_and_only_cover_free_voxels():
    target = torch.zeros(1, 48, 48, 48, dtype=torch.long)
    target[:, 28] = 1
    surface, free, weights = visible_voxel_masks(
        target, torch.ones_like(target, dtype=torch.bool), torch.zeros(1, 1, 3),
        torch.tensor([[[1., 0., 0.]]]), torch.ones(1, 1, dtype=torch.bool),
        torch.full((1, 1), .3), torch.full((1, 1), .3), 0,
        height=8, width=8, return_ray_weights=True,
    )
    assert torch.allclose(weights.sum(), torch.tensor(float(round(8 * .82) * 8)))
    assert not weights[~free].any()
    assert not weights[surface].any()
    logits = torch.zeros_like(weights, requires_grad=True)
    loss = (torch.nn.functional.softplus(logits) * weights).sum() / weights.sum()
    loss.backward()
    assert (logits.grad[free] > 0).all()  # Gradient descent reduces false occupancy.
    assert not logits.grad[~free].any()
