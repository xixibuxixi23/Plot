"""Structured eight-horizon TextAgent action head, adapted from 2DAction Stage 2."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class StructuredActionHead(nn.Module):
    """Predict compatible keys, one hotbar choice and categorical mouse deltas."""

    def __init__(self, hidden_size=1024, horizons=8, mouse_bins_x=None, mouse_bins_y=None):
        super().__init__()
        self.horizons = horizons
        # Exclude zoom=7 and inventory=11, matching the original 2DAction head.
        self.register_buffer("key_indices", torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 9, 10]))
        self.keys = nn.Linear(hidden_size, horizons * 10)
        self.hotbar = nn.Linear(hidden_size, horizons * 10)  # none + slots 1..9
        default = [-1., -.16, -.08, -.05, -.025, -.016, -.008, -.004,
                   0., .004, .008, .016, .025, .05, .08, .16, 1.]
        self.register_buffer("mouse_bins_x", torch.tensor(mouse_bins_x or default))
        self.register_buffer("mouse_bins_y", torch.tensor(mouse_bins_y or default))
        self.mouse_x = nn.Linear(hidden_size, horizons * len(self.mouse_bins_x))
        self.mouse_y = nn.Linear(hidden_size, horizons * len(self.mouse_bins_y))

    def forward(self, hidden):
        shape = (*hidden.shape[:-1], self.horizons)
        return {
            "keys": self.keys(hidden).view(*shape, 10),
            "hotbar": self.hotbar(hidden).view(*shape, 10),
            "mouse_x": self.mouse_x(hidden).view(*shape, len(self.mouse_bins_x)),
            "mouse_y": self.mouse_y(hidden).view(*shape, len(self.mouse_bins_y)),
        }

    @staticmethod
    def _nearest(values, bins):
        return (values.unsqueeze(-1) - bins.to(values)).abs().argmin(-1)

    def targets(self, actions):
        hotbar_values = actions[..., 12:21]
        hotbar = hotbar_values.argmax(-1) + 1
        hotbar = torch.where(hotbar_values.sum(-1) > 0, hotbar, 0)
        return {
            "keys": actions.index_select(-1, self.key_indices),
            "hotbar": hotbar,
            "mouse_x": self._nearest(actions[..., 21], self.mouse_bins_x),
            "mouse_y": self._nearest(actions[..., 22], self.mouse_bins_y),
        }

    def loss(self, logits, actions, valid_mask=None, horizon_weights=None,
             sample_weight=None):
        target = self.targets(actions)
        valid = (torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
                 if valid_mask is None else valid_mask.bool())
        keys = F.binary_cross_entropy_with_logits(
            logits["keys"], target["keys"], reduction="none").mean(-1)
        hotbar = F.cross_entropy(logits["hotbar"].flatten(0, -2), target["hotbar"].flatten(),
                                 reduction="none").view_as(valid)
        mouse_x = F.cross_entropy(logits["mouse_x"].flatten(0, -2), target["mouse_x"].flatten(),
                                  reduction="none").view_as(valid)
        mouse_y = F.cross_entropy(logits["mouse_y"].flatten(0, -2), target["mouse_y"].flatten(),
                                  reduction="none").view_as(valid)
        combined = keys + hotbar + mouse_x + mouse_y
        if horizon_weights is None:
            horizon_weights = torch.ones(self.horizons, device=actions.device)
        weights = valid.to(combined.dtype) * horizon_weights.to(combined)[None]
        denominator = weights.sum(-1)
        per_sample = (combined * weights).sum(-1) / denominator.clamp_min(1e-8)
        sample_valid = (denominator > 0).to(combined.dtype)
        if sample_weight is not None:
            sample_valid = sample_valid * sample_weight.to(combined)
        total = (per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1)
        return total, {"keys": keys, "hotbar": hotbar,
                       "mouse_x": mouse_x, "mouse_y": mouse_y}

    @torch.no_grad()
    def decode(self, logits, key_threshold=.5):
        keys = logits["keys"].sigmoid() >= key_threshold
        hotbar = logits["hotbar"].argmax(-1)
        mouse_x = self.mouse_bins_x[logits["mouse_x"].argmax(-1)]
        mouse_y = self.mouse_bins_y[logits["mouse_y"].argmax(-1)]
        output = logits["keys"].new_zeros((*keys.shape[:-1], 23))
        output[..., self.key_indices] = keys.to(output.dtype)
        for slot in range(1, 10):
            output[..., 11 + slot] = (hotbar == slot).to(output.dtype)
        output[..., 21], output[..., 22] = mouse_x, mouse_y
        return output


def constrain_peaceful_logits(logits, peaceful):
    """Peaceful residents may move/turn but cannot edit, drop or select items."""
    if not peaceful.any():
        return logits
    result = {key: value.clone() for key, value in logits.items()}
    # Structured key positions 7,8,9 are dig/place/drop.
    result["keys"][peaceful, :, 7:] = -20.
    result["hotbar"][peaceful] = -20.
    result["hotbar"][peaceful, :, 0] = 20.
    return result
