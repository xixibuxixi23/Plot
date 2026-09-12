"""Continuous latent flow matching at multiple causal eight-frame anchors."""
from __future__ import annotations

import torch


STATIC_KEYS = {"target_agent", "player_skin", "player_appearance_valid"}


def slice_conditions(cond, start, end):
    return {key: value if key in STATIC_KEYS else value[:, start:end]
            for key, value in cond.items()}


def renderer_flow_loss(model, clean, conditions, *, region_weight=None, generator=None):
    """Causal diffusion-forcing loss with frame zero known and frames 1:65 noisy.

    All 64 unknown frames are scored in one forward pass. Causal attention
    prevents a frame from seeing later frames. Supervision masks only weight
    the loss and never enter the neural conditions.
    """
    base = model.module if hasattr(model, "module") else model
    if clean.ndim != 5 or clean.shape[1] > base.cfg.context_frames:
        raise ValueError("clean must be [B,T,C,H,W] within the configured context")
    if clean.shape[1] != base.cfg.context_frames:
        raise ValueError("training clips must fill the configured causal context")
    if base.core.kv_caches is not None:
        raise RuntimeError("clear rollout caches before training")
    c = dict(conditions)
    c["condition_mask"] = (torch.arange(clean.shape[1], device=clean.device)[None] == 0)
    c["condition_mask"] = c["condition_mask"].expand(len(clean), -1)
    c["action_prefix_mask"] = c["condition_mask"]
    time = torch.zeros(clean.shape[:2], device=clean.device)
    time[:, 1:] = torch.rand(time[:, 1:].shape, device=clean.device, generator=generator)
    noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    tau = time[..., None, None, None]
    noisy = (1 - tau) * clean + tau * noise
    prediction = model(noisy, time, c)
    error = (prediction[:, 1:].float() - (noise - clean)[:, 1:].float()).square()
    if region_weight is None:
        return error.mean()
    weight = region_weight[:, 1:].float()
    expected = (len(clean), clean.shape[1] - 1, 1, *clean.shape[-2:])
    if weight.shape != expected:
        raise ValueError("region_weight must be [B,T,1,H,W]")
    return (error * weight).sum() / (weight.sum().clamp_min(1) * clean.shape[2])


class RendererRollout:
    """Persistent per-view history; each call adds exactly eight completed frames."""

    def __init__(self, model, *, denoising_steps=20):
        if denoising_steps < 1:
            raise ValueError("denoising_steps must be positive")
        self.model, self.denoising_steps = model, denoising_steps
        self.next_frame = None
        self.last_policy_features = None
        self.last_policy_layers = None

    @torch.no_grad()
    def start(self, first_latent, conditions, *, global_start_idx=0):
        if first_latent.shape[1] != 1:
            raise ValueError("initialization requires exactly one completed observation")
        self.model.eval()
        self.model.init_kv_cache(len(first_latent), dtype=first_latent.dtype)
        self.model.set_kv_cache_start(global_start_idx)
        self.last_policy_features = self._commit(first_latent, conditions, global_start_idx)
        self.next_frame = global_start_idx + 1

    def _commit(self, frames, conditions, index):
        cond = dict(conditions, condition_mask=torch.ones(frames.shape[:2],
                                                         device=frames.device, dtype=torch.bool))
        layers = []
        first = max(0, self.model.core.depth - 4)
        def capture(block_index, hidden):
            if block_index >= first:
                layers.append(hidden.detach())
        _, features, candidates = self.model(
            frames, torch.zeros(frames.shape[:2], device=frames.device), cond,
            global_start_idx=index, cache_write=False, return_features=True,
            return_kv_candidates=True, block_callback=capture,
        )
        self.model.commit_kv_candidates(candidates, index)
        self.last_policy_layers = tuple(layers)
        return self.model.compress_policy_features(features)

    @torch.no_grad()
    def generate(self, noise, conditions):
        if self.next_frame is None:
            raise RuntimeError("start must prefill the external first observation")
        if noise.shape[1] != 8:
            raise ValueError("M3 rollout must generate exactly eight new frames")
        x = noise.clone()
        cond = dict(conditions, condition_mask=torch.zeros(x.shape[:2],
                                                         device=x.device, dtype=torch.bool))
        # Geometry/appearance is invariant across the denoising iterations.
        encoded = self.model.encode_conditions(cond)
        for step in range(self.denoising_steps):
            tau = 1.0 - step / self.denoising_steps
            velocity = self.model.core(x, torch.full(x.shape[:2], tau, device=x.device),
                                       encoded, global_start_idx=self.next_frame, cache_write=False)
            x = x - velocity / self.denoising_steps
        # Recompute at clean t=0; never commit an intermediate noisy candidate.
        self.last_policy_features = self._commit(x, conditions, self.next_frame)
        self.next_frame += 8
        return x

    @torch.no_grad()
    def generate_64(self, noise, conditions):
        """Generate the deployment horizon as eight cached eight-frame chunks."""
        if noise.shape[1] != 64:
            raise ValueError("M3 deployment horizon requires exactly 64 noise frames")
        chunks = []
        for start in range(0, 64, 8):
            chunks.append(self.generate(
                noise[:, start:start + 8], slice_conditions(conditions, start, start + 8)
            ))
        return torch.cat(chunks, dim=1)
