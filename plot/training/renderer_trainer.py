"""Continuous latent flow matching at multiple causal eight-frame anchors."""
from __future__ import annotations

import torch
import torch.nn.functional as F


STATIC_KEYS = {"target_agent", "player_skin", "player_appearance_valid"}


def slice_conditions(cond, start, end):
    return {key: value if key in STATIC_KEYS else value[:, start:end]
            for key, value in cond.items()}


def renderer_flow_loss(
    model, clean, conditions, *, region_weight=None, generator=None,
    return_clean_prediction=False,
):
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
        loss = error.mean()
    else:
        weight = region_weight[:, 1:].float()
        expected = (len(clean), clean.shape[1] - 1, 1, *clean.shape[-2:])
        if weight.shape != expected:
            raise ValueError("region_weight must be [B,T,1,H,W]")
        loss = (error * weight).sum() / (weight.sum().clamp_min(1) * clean.shape[2])
    if return_clean_prediction:
        # For v = epsilon - x_0 and x_t = (1-t)x_0 + t epsilon,
        # x_0 = x_t - t*v. Pixel supervision decodes only selected frames.
        return loss, noisy - tau * prediction
    return loss


def _masked_mean(value, mask):
    channels = value.shape[-3]
    return (value.float() * mask.float()).sum() / (mask.sum().clamp_min(1) * channels)


def _masked_edge_l1(prediction, target, mask):
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (
        target[..., 1:] - target[..., :-1]
    )
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        target[..., 1:, :] - target[..., :-1, :]
    )
    mask_x = torch.maximum(mask[..., 1:], mask[..., :-1])
    mask_y = torch.maximum(mask[..., 1:, :], mask[..., :-1, :])
    return (_masked_mean(dx.abs(), mask_x) + _masked_mean(dy.abs(), mask_y)) / 2


def renderer_pixel_losses(
    codec,
    clean_prediction,
    target_rgb,
    pixel_region_mask,
    *,
    frames_per_sample=1,
    generator=None,
    health_box=(190 / 640, 300 / 360, 314 / 640, 322 / 360),
):
    """Full-resolution entity and Minecraft heart-bar losses.

    Only sampled future frames are decoded, which keeps the frozen VAE
    backward graph small. ``health_box`` is normalized ``(x0,y0,x1,y1)``.
    """
    if clean_prediction.ndim != 5 or target_rgb.ndim != 5:
        raise ValueError("predicted latents and target RGB must be [B,T,C,H,W]")
    if clean_prediction.shape[:2] != target_rgb.shape[:2]:
        raise ValueError("predicted latents and target RGB must align in B,T")
    expected_mask = (*target_rgb.shape[:2], 1, *target_rgb.shape[-2:])
    if pixel_region_mask.shape != expected_mask:
        raise ValueError("pixel_region_mask must be [B,T,1,H,W] at RGB resolution")
    future_frames = clean_prediction.shape[1] - 1
    if not 1 <= frames_per_sample <= future_frames:
        raise ValueError("frames_per_sample must be within the future-frame count")
    if len(health_box) != 4 or not (
        0 <= health_box[0] < health_box[2] <= 1
        and 0 <= health_box[1] < health_box[3] <= 1
    ):
        raise ValueError("health_box must be normalized (x0,y0,x1,y1)")

    batch = clean_prediction.shape[0]
    frame_indices = torch.randint(
        1,
        clean_prediction.shape[1],
        (batch, frames_per_sample),
        device=clean_prediction.device,
        generator=generator,
    )
    batch_indices = torch.arange(batch, device=clean_prediction.device)[:, None]
    selected_latent = clean_prediction[batch_indices, frame_indices].reshape(
        batch * frames_per_sample, 1, *clean_prediction.shape[2:]
    )
    selected_target = target_rgb[batch_indices, frame_indices].reshape(
        batch * frames_per_sample, *target_rgb.shape[2:]
    )
    selected_mask = pixel_region_mask[batch_indices, frame_indices].reshape(
        batch * frames_per_sample, 1, *target_rgb.shape[-2:]
    )
    decoded = codec.decode_for_loss(selected_latent, chunk_size=1)[:, 0]

    entity_l1 = _masked_mean((decoded - selected_target).abs(), selected_mask)
    # A five-pixel dilation lets boundary gradients cover anti-aliased edges,
    # while the RGB L1 above remains confined to the exact instance mask.
    edge_mask = F.max_pool2d(selected_mask.float(), kernel_size=5, stride=1, padding=2)
    entity_edge = _masked_edge_l1(decoded, selected_target, edge_mask)

    height, width = target_rgb.shape[-2:]
    x0, y0, x1, y1 = health_box
    left, right = int(round(x0 * width)), int(round(x1 * width))
    top, bottom = int(round(y0 * height)), int(round(y1 * height))
    health_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
    health_mask[..., top:bottom, left:right] = True
    health_l1 = _masked_mean((decoded - selected_target).abs(), health_mask)
    return {
        "entity_pixel_l1": entity_l1,
        "entity_pixel_edge": entity_edge,
        "health_pixel_l1": health_l1,
    }


def renderer_training_losses(
    model,
    codec,
    clean,
    conditions,
    target_rgb,
    pixel_region_mask,
    *,
    region_weight=None,
    frames_per_sample=1,
    entity_pixel_l1_weight=0.1,
    entity_pixel_edge_weight=0.05,
    health_pixel_l1_weight=0.2,
    generator=None,
):
    """Combine latent flow matching with sparse full-resolution supervision."""
    weights = {
        "entity_pixel_l1": float(entity_pixel_l1_weight),
        "entity_pixel_edge": float(entity_pixel_edge_weight),
        "health_pixel_l1": float(health_pixel_l1_weight),
    }
    if min(weights.values()) < 0:
        raise ValueError("pixel loss weights must be nonnegative")
    use_pixels = any(weight > 0 for weight in weights.values())
    result = renderer_flow_loss(
        model,
        clean,
        conditions,
        region_weight=region_weight,
        generator=generator,
        return_clean_prediction=use_pixels,
    )
    if use_pixels:
        flow_loss, clean_prediction = result
        pixels = renderer_pixel_losses(
            codec,
            clean_prediction,
            target_rgb,
            pixel_region_mask,
            frames_per_sample=frames_per_sample,
            generator=generator,
        )
    else:
        flow_loss = result
        pixels = {name: flow_loss.new_zeros(()) for name in weights}
    auxiliary = sum(weights[name] * value for name, value in pixels.items())
    return {
        "total_loss": flow_loss + auxiliary,
        "flow_loss": flow_loss,
        "auxiliary_loss": auxiliary,
        **pixels,
    }


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
