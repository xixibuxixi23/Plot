"""Continuous latent flow matching over causal, bidirectional eight-frame blocks."""
from __future__ import annotations

import torch
import torch.nn.functional as F


STATIC_KEYS = {"target_agent", "player_skin", "player_reference", "player_appearance_valid"}


def slice_conditions(cond, start, end):
    return {key: value if key in STATIC_KEYS else value[:, start:end]
            for key, value in cond.items()}


def _sample_blockwise_train_time(batch_size, total_frames, block_frames, device, generator=None):
    """Keep the observed prefix clean and assign one noise time per future block."""
    future_frames = total_frames - 1
    if future_frames < block_frames or future_frames % block_frames:
        raise ValueError("future training frames must contain complete output blocks")
    block_time = torch.rand(
        batch_size, future_frames // block_frames, device=device, generator=generator,
    )
    time = torch.zeros(batch_size, total_frames, device=device)
    time[:, 1:] = block_time.repeat_interleave(block_frames, dim=1)
    return time


def renderer_flow_loss(
    model, clean, conditions, *, region_weight=None, generator=None,
    return_clean_prediction=False,
):
    """Block-causal diffusion-forcing loss with frame zero known.

    All unknown frames are scored in one forward pass. Frames within an output
    block share one noise time and attend bidirectionally; blocks remain causal.
    Supervision masks only weight the loss and never enter neural conditions.
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
    block_frames = base.cfg.block_frames
    time = _sample_blockwise_train_time(
        len(clean), clean.shape[1], block_frames, clean.device, generator,
    )
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


def select_renderer_pixel_frames(
    pixel_region_mask,
    *,
    frames_per_sample,
    player_region_mask=None,
    hp=None,
    target_agent=None,
    generator=None,
    full_health=20.0,
):
    """Select an entity-rich frame and, when possible, a damaged-HP frame."""
    if pixel_region_mask.ndim != 5 or pixel_region_mask.shape[2] != 1:
        raise ValueError("pixel_region_mask must be [B,T,1,H,W]")
    batch, frames = pixel_region_mask.shape[:2]
    if not 1 <= frames_per_sample < frames:
        raise ValueError("frames_per_sample must be within the future-frame count")

    coverage = pixel_region_mask[:, 1:].flatten(2).sum(-1).float()
    entity_frame = torch.multinomial(
        coverage + 1e-3, 1, replacement=True, generator=generator
    ) + 1
    player_frame = entity_frame
    if player_region_mask is not None:
        if player_region_mask.shape != pixel_region_mask.shape:
            raise ValueError("player_region_mask must match pixel_region_mask")
        player_coverage = player_region_mask[:, 1:].flatten(2).sum(-1).float()
        sampled_player = torch.multinomial(
            player_coverage + 1e-3, 1, replacement=True, generator=generator
        ) + 1
        player_frame = torch.where(
            player_coverage.sum(1, keepdim=True) > 0, sampled_player, entity_frame
        )
    random_frame = torch.randint(
        1, frames, (batch, 1), device=pixel_region_mask.device, generator=generator
    )
    health_frame = random_frame
    has_damage = torch.zeros(batch, dtype=torch.bool, device=pixel_region_mask.device)
    target_hp = None
    if hp is not None or target_agent is not None:
        if hp is None or target_agent is None:
            raise ValueError("hp and target_agent must be supplied together")
        if hp.ndim != 3 or hp.shape[:2] != (batch, frames):
            raise ValueError("hp must be [B,T,A] and align with the masks")
        if target_agent.shape != (batch,):
            raise ValueError("target_agent must be [B]")
        gather = target_agent.long()[:, None, None].expand(-1, frames, 1)
        target_hp = hp.gather(2, gather).squeeze(2)
        future_hp = target_hp[:, 1:]
        has_damage = future_hp.min(1).values < full_health - 1e-3
        minimum = future_hp.min(1, keepdim=True).values
        minimum_frames = (future_hp <= minimum + 1e-3).float()
        sampled_minimum = torch.multinomial(
            minimum_frames, 1, replacement=True, generator=generator
        ) + 1
        health_frame = torch.where(has_damage[:, None], sampled_minimum, random_frame)

    if frames_per_sample == 1:
        indices = torch.where(has_damage[:, None], health_frame, entity_frame)
    elif frames_per_sample >= 3 and player_region_mask is not None:
        extras = torch.randint(
            1,
            frames,
            (batch, frames_per_sample - 3),
            device=pixel_region_mask.device,
            generator=generator,
        )
        indices = torch.cat((entity_frame, health_frame, player_frame, extras), dim=1)
    else:
        extras = torch.randint(
            1,
            frames,
            (batch, frames_per_sample - 2),
            device=pixel_region_mask.device,
            generator=generator,
        )
        indices = torch.cat((entity_frame, health_frame, extras), dim=1)
    return indices, target_hp


def renderer_pixel_losses(
    codec,
    clean_prediction,
    target_rgb,
    pixel_region_mask,
    *,
    player_region_mask=None,
    frames_per_sample=1,
    generator=None,
    hp=None,
    target_agent=None,
    damaged_health_upweight=4.0,
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
    if player_region_mask is not None and player_region_mask.shape != expected_mask:
        raise ValueError("player_region_mask must match the RGB-resolution entity mask")
    future_frames = clean_prediction.shape[1] - 1
    if not 1 <= frames_per_sample <= future_frames:
        raise ValueError("frames_per_sample must be within the future-frame count")
    if len(health_box) != 4 or not (
        0 <= health_box[0] < health_box[2] <= 1
        and 0 <= health_box[1] < health_box[3] <= 1
    ):
        raise ValueError("health_box must be normalized (x0,y0,x1,y1)")

    if damaged_health_upweight < 0:
        raise ValueError("damaged_health_upweight must be nonnegative")
    batch = clean_prediction.shape[0]
    frame_indices, target_hp = select_renderer_pixel_frames(
        pixel_region_mask,
        frames_per_sample=frames_per_sample,
        player_region_mask=player_region_mask,
        hp=hp,
        target_agent=target_agent,
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
    selected_player_mask = (
        player_region_mask[batch_indices, frame_indices].reshape(
            batch * frames_per_sample, 1, *target_rgb.shape[-2:]
        )
        if player_region_mask is not None
        else torch.zeros_like(selected_mask)
    )
    decoded = codec.decode_for_loss(selected_latent, chunk_size=1)[:, 0]

    entity_l1 = _masked_mean((decoded - selected_target).abs(), selected_mask)
    # A five-pixel dilation lets boundary gradients cover anti-aliased edges,
    # while the RGB L1 above remains confined to the exact instance mask.
    edge_mask = F.max_pool2d(selected_mask.float(), kernel_size=5, stride=1, padding=2)
    entity_edge = _masked_edge_l1(decoded, selected_target, edge_mask)
    player_l1 = _masked_mean((decoded - selected_target).abs(), selected_player_mask)
    player_edge_mask = F.max_pool2d(
        selected_player_mask.float(), kernel_size=5, stride=1, padding=2
    )
    player_edge = _masked_edge_l1(decoded, selected_target, player_edge_mask)

    height, width = target_rgb.shape[-2:]
    x0, y0, x1, y1 = health_box
    left, right = int(round(x0 * width)), int(round(x1 * width))
    top, bottom = int(round(y0 * height)), int(round(y1 * height))
    health_mask = torch.zeros_like(selected_mask, dtype=torch.bool)
    health_mask[..., top:bottom, left:right] = True
    if target_hp is not None:
        selected_hp = target_hp[batch_indices, frame_indices].reshape(-1, 1, 1, 1)
        health_mask = health_mask.float() * (
            1 + damaged_health_upweight * (selected_hp < 20 - 1e-3).float()
        )
    health_l1 = _masked_mean((decoded - selected_target).abs(), health_mask)
    return {
        "entity_pixel_l1": entity_l1,
        "entity_pixel_edge": entity_edge,
        "player_pixel_l1": player_l1,
        "player_pixel_edge": player_edge,
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
    player_region_mask=None,
    region_weight=None,
    frames_per_sample=2,
    entity_pixel_l1_weight=0.5,
    entity_pixel_edge_weight=0.2,
    player_pixel_l1_weight=0.0,
    player_pixel_edge_weight=0.0,
    health_pixel_l1_weight=1.0,
    damaged_health_upweight=4.0,
    generator=None,
):
    """Combine latent flow matching with sparse full-resolution supervision."""
    weights = {
        "entity_pixel_l1": float(entity_pixel_l1_weight),
        "entity_pixel_edge": float(entity_pixel_edge_weight),
        "player_pixel_l1": float(player_pixel_l1_weight),
        "player_pixel_edge": float(player_pixel_edge_weight),
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
            player_region_mask=player_region_mask,
            frames_per_sample=frames_per_sample,
            generator=generator,
            hp=conditions.get("hp"),
            target_agent=conditions.get("target_agent"),
            damaged_health_upweight=damaged_health_upweight,
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
        base = model.module if hasattr(model, "module") else model
        self.block_frames = base.cfg.block_frames
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
        if noise.shape[1] != self.block_frames:
            raise ValueError(f"M3 rollout must generate exactly {self.block_frames} new frames")
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
        self.next_frame += self.block_frames
        return x

    @torch.no_grad()
    def generate_64(self, noise, conditions):
        """Generate the deployment horizon as eight cached eight-frame chunks."""
        if noise.shape[1] != 64:
            raise ValueError("M3 deployment horizon requires exactly 64 noise frames")
        chunks = []
        if 64 % self.block_frames:
            raise ValueError("deployment horizon must contain complete output blocks")
        for start in range(0, 64, self.block_frames):
            chunks.append(self.generate(
                noise[:, start:start + self.block_frames],
                slice_conditions(conditions, start, start + self.block_frames)
            ))
        return torch.cat(chunks, dim=1)
