"""Additive, detached player diagnostics for multi-rank/interval aggregation."""
from __future__ import annotations

import torch


PLAYER_DIAGNOSTIC_KEYS = tuple(
    f"player_{kind}_{level}_{stat}"
    for kind in ("flow", "pixel")
    for level in ("low", "mid", "high")
    for stat in ("sum", "count")
) + ("player_pixel_valid_count", "player_pixel_selected_count")


def pack_player_diagnostics(diagnostics, *, device):
    return torch.stack([
        torch.as_tensor(diagnostics.get(key, 0.), device=device, dtype=torch.float32).detach()
        for key in PLAYER_DIAGNOSTIC_KEYS
    ])


def summarize_player_diagnostics(totals):
    """Call after SUM-reducing all ranks, never average rank-local averages."""
    values = dict(zip(PLAYER_DIAGNOSTIC_KEYS, totals.detach().cpu().tolist(), strict=True))
    result = dict(values)
    for kind in ("flow", "pixel"):
        for level in ("low", "mid", "high"):
            prefix = f"player_{kind}_{level}"
            count = values[prefix + "_count"]
            if count > 0:
                result[prefix + "_mean"] = values[prefix + "_sum"] / count
            # An empty bin is missing, not a measured zero error.
    selected = values["player_pixel_selected_count"]
    if selected > 0:
        result["player_pixel_supervised_fraction"] = values["player_pixel_valid_count"] / selected
    return result
