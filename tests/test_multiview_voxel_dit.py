import pytest
import torch

from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT


def test_player_permutation_and_padding_are_invariant():
    torch.manual_seed(1)
    m = MultiViewVoxelDiT(width=32, depth=1, heads=4).eval()
    x = torch.randn(1, 48, 12, 12, 12)
    t = torch.tensor([0.5])
    views = torch.randn(1, 2, 22, 36, 64)
    with torch.no_grad():
        y = m(x, t, views, torch.ones(1, 2, dtype=torch.bool))
        swapped = m(x, t, views.flip(1), torch.ones(1, 2, dtype=torch.bool))
        assert torch.allclose(y, swapped, atol=2e-6)
        one = m(x, t, views[:, :1], torch.ones(1, 1, dtype=torch.bool))
        views[:, 1] = float("nan")
        padded = m(x, t, views, torch.tensor([[True, False]]))
        assert torch.allclose(one, padded, atol=2e-6)
        with pytest.raises(ValueError, match="valid input view"):
            m(x, t, views, torch.zeros(1, 2, dtype=torch.bool))


def test_invalid_view_has_no_gradient_and_output_layout():
    torch.manual_seed(2)
    m = MultiViewVoxelDiT(width=32, depth=1, heads=4)
    views = torch.randn(1, 2, 22, 36, 64, requires_grad=True)
    y = m(torch.randn(1, 48, 12, 12, 12), torch.tensor([0.3]), views, torch.tensor([[True, False]]))
    assert y.shape == (1, 48, 12, 12, 12)
    y.square().mean().backward()
    assert views.grad[:, 0].abs().sum() > 0
    assert views.grad[:, 1].abs().sum() == 0
