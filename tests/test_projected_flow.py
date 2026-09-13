import torch
from experiments.m1.projected_flow import project_points, projected_features, ProjectedFlow
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from plot.models.projection import camera_rays


def test_projection_inverts_existing_rays_including_vertical_camera():
    for direction in ([0.0, 1.0, 0.0], [1.0, 0.0, 0.3], [0.0, 0.0, 1.0]):
        d = torch.tensor([direction])
        f = torch.tensor([1.2])
        g = torch.tensor([0.8])
        points = camera_rays(d, f, g, 6, 8).reshape(-1, 3) * 0.2
        grid, valid = project_points(
            points, torch.zeros(1, 1, 3), d[:, None], f[:, None], g[:, None]
        )
        yy, xx = torch.meshgrid(
            (torch.arange(6) + 0.5) * 2 / 6 - 1, (torch.arange(8) + 0.5) * 2 / 8 - 1, indexing="ij"
        )
        assert valid.all()
        torch.testing.assert_close(
            grid, torch.stack([xx, yy], -1).reshape(1, 1, -1, 2), atol=2e-6, rtol=2e-6
        )


def test_back_facing_points_invalid():
    _, v = project_points(
        torch.tensor([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [3.0, 1.0, 0.0]]),
        torch.zeros(1, 1, 3),
        torch.tensor([[[0.0, 1.0, 0.0]]]),
        torch.ones(1, 1),
        torch.ones(1, 1),
    )
    assert v.tolist() == [[[False, True, False]]]


def test_projection_view_order_and_missing_views():
    data = dict(
        image_rays=torch.randn(1, 2, 22, 36, 64),
        camera_position=torch.tensor([[[0.0, -0.8, 0.0], [0.0, 0.8, 0.0]]]),
        camera_direction=torch.tensor([[[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]]),
        fov_x=torch.ones(1, 2) * 1.5,
        fov_y=torch.ones(1, 2) * 1.2,
        agent_mask=torch.ones(1, 2, dtype=torch.bool),
        camera_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    torch.testing.assert_close(
        projected_features(data), projected_features({k: v.flip(1) for k, v in data.items()})
    )
    data["camera_valid"].zero_()
    assert torch.count_nonzero(projected_features(data)) == 0


def test_zero_adapter_preserves_baseline_and_receives_gradient():
    base = MultiViewVoxelDiT(width=32, depth=1, heads=4)
    model = ProjectedFlow(width=32, depth=1, heads=4)
    assert model.load_state_dict(base.state_dict(), strict=False).missing_keys == [
        "projection_adapter.weight"
    ]
    x = torch.randn(1, 48, 12, 12, 12)
    t = torch.ones(1) * 0.5
    cond = torch.randn(1, 2, 22, 36, 64)
    valid = torch.ones(1, 2, dtype=torch.bool)
    out = model(x, t, cond, valid, torch.randn(1, 216, 16))
    torch.testing.assert_close(out, base(x, t, cond, valid), atol=0, rtol=0)
    out.square().mean().backward()
    assert model.projection_adapter.weight.grad.abs().sum() > 0
