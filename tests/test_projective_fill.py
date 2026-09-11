import torch

from plot.models.projective_fill import ProjectiveFillArgs, project_voxels


def test_projection_orientation_and_behind_camera():
    # At the grid centre looking +X: screen-right is -Y and screen-up is +Z.
    pos = torch.zeros(1, 1, 3)
    direction = torch.tensor([[[1., 0., 0.]]])
    fov = torch.full((1, 1), torch.pi / 2)
    grid, depth, valid = project_voxels(pos, direction, fov, fov, 4)
    grid = grid.reshape(4, 4, 4, 2)
    valid = valid.reshape(4, 4, 4)
    assert torch.allclose(grid[3, 1, 2], torch.tensor([1/3, -1/3]))
    assert valid[3, 1, 2]
    assert not valid[:2].any()
    assert depth.isfinite().all()


def test_padding_gradients_and_known_cells():
    torch.manual_seed(1)
    model = ProjectiveFillArgs(5, voxel_size=8, image_channels=8, channels=8).build()
    images = torch.randn(1, 2, 3, 32, 32, requires_grad=True)
    known = torch.zeros(1, 8, 8, 8, dtype=torch.bool)
    known[:, 0] = True
    args = dict(voxel_context=torch.zeros_like(known, dtype=torch.long), known_mask=known,
        fill_mask=~known, images=images, agent_mask=torch.tensor([[True, False]]),
        camera_position=torch.tensor([[[-.6, 0., 0.], [-.6, 0., 0.]]]),
        camera_direction=torch.tensor([[[1., 0., 0.], [1., 0., 0.]]]),
        fov_x=torch.ones(1, 2) * 1.5, fov_y=torch.ones(1, 2) * 1.5)
    output = model(**args)
    output.square().mean().backward()
    assert output.shape == (1, 5, 8, 8, 8)
    assert images.grad[:, 0].abs().sum() > 0
    assert images.grad[:, 1].abs().sum() == 0
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())
    assert (model.predict(**args)[known] == 0).all()
    args["agent_mask"][:] = False
    assert model(**args).isfinite().all()
