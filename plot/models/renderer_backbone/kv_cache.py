"""Functional helpers for final-denoise-commit temporal KV caches."""

from __future__ import annotations

from typing import TypedDict

import torch


class TemporalKVCache(TypedDict):
    k: torch.Tensor
    v: torch.Tensor
    global_end_index: torch.Tensor
    local_end_index: torch.Tensor


def valid_history(cache: TemporalKVCache) -> tuple[torch.Tensor, torch.Tensor]:
    """Return only committed entries without modifying the cache."""
    length = int(cache["local_end_index"].item())
    return cache["k"][:, :, :length], cache["v"][:, :, :length]


@torch.no_grad()
def commit_current_kv(
    cache: TemporalKVCache,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    frame_index: int,
) -> None:
    """Commit a contiguous block of final-denoise K/V into a sliding window.

    The candidate must come from the final denoiser forward. This function is
    deliberately separate from attention so intermediate denoising steps cannot
    mutate persistent state accidentally.
    """
    if current_k.shape != current_v.shape or current_k.shape[2] < 1:
        raise ValueError("final-denoise commit expects matching non-empty K/V")
    expected = int(cache["global_end_index"].item())
    if frame_index != expected:
        raise ValueError(f"Expected frame_index={expected}, got {frame_index}")

    length = int(cache["local_end_index"].item())
    capacity = cache["k"].shape[2]
    block = current_k.shape[2]
    for offset in range(block):
        if length == capacity:
            cache["k"][:, :, :-1].copy_(cache["k"][:, :, 1:].clone())
            cache["v"][:, :, :-1].copy_(cache["v"][:, :, 1:].clone())
            length -= 1
        cache["k"][:, :, length : length + 1].copy_(
            current_k[:, :, offset : offset + 1]
        )
        cache["v"][:, :, length : length + 1].copy_(
            current_v[:, :, offset : offset + 1]
        )
        length += 1
    cache["local_end_index"].fill_(length)
    cache["global_end_index"].fill_(frame_index + block)


def clone_metadata(cache: TemporalKVCache) -> tuple[int, int]:
    """Small immutable snapshot used by cache-invariance tests."""
    return (
        int(cache["global_end_index"].item()),
        int(cache["local_end_index"].item()),
    )
