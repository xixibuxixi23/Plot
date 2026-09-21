import pytest
import torch

from plot.training.renderer_trainer import select_renderer_pixel_frames


def select(mask, count=2, seed=19):
    return select_renderer_pixel_frames(
        torch.ones_like(mask),
        frames_per_sample=count,
        player_region_mask=mask,
        selection_mode="player_unique",
        generator=torch.Generator().manual_seed(seed),
    )[0]


def test_player_unique_prefers_distinct_visible_future_frames():
    mask = torch.zeros(32, 9, 1, 4, 4)
    mask[:, 0] = 1  # The observed prefix must never be selected.
    mask[:, 2, :, :1] = 1
    mask[:, 7, :, :3] = 1
    indices = select(mask)
    assert torch.equal(indices.sort(1).values, torch.tensor([[2, 7]]).expand(32, -1))
    assert torch.equal(indices, select(mask))


def test_player_unique_falls_back_without_duplicate_or_fake_supervision():
    mask = torch.zeros(32, 9, 1, 4, 4)
    mask[:, 5] = 1
    indices = select(mask)
    assert (indices[:, 0] == 5).all()
    assert (indices[:, 1] != 5).all()
    assert ((indices > 0) & (indices < 9)).all()

    empty = select(torch.zeros_like(mask), count=8)
    assert torch.equal(empty.sort(1).values, torch.arange(1, 9).expand(32, -1))


def test_player_selection_requires_rgb_player_mask():
    mask = torch.zeros(1, 9, 1, 4, 4)
    with pytest.raises(ValueError, match="requires player_region_mask"):
        select_renderer_pixel_frames(
            mask, frames_per_sample=1, selection_mode="player_unique"
        )
