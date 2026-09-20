"""Full M3 fine-tuning with separate appearance and pretrained learning rates."""
from __future__ import annotations

import math

import torch


class AppearanceAdamW(torch.optim.AdamW):
    """Give the appended appearance columns their own AdamW learning rate.

    Gradient scaling alone cannot do this: Adam normalizes gradient magnitude.
    Instead rescale the actual AdamW update (including weight decay) for those
    columns. The model tensor names and inference checkpoint format stay intact.
    """

    @torch.no_grad()
    def step(self, closure=None):
        tails = []
        for group in self.param_groups:
            channels = group.get("appearance_input_channels", 0)
            if channels:
                parameter, = group["params"]
                if parameter.grad is not None:
                    tail = parameter[:, -channels:]
                    tails.append((tail, tail.clone(), group["appearance_lr"] / group["lr"]))
        loss = super().step(closure)
        for tail, previous, ratio in tails:
            tail.copy_(previous + (tail - previous) * ratio)
        return loss


def build_full_player_optimizer(model, *, base_lr, appearance_lr, weight_decay=0.01):
    if not all(math.isfinite(lr) and lr > 0 for lr in (base_lr, appearance_lr)):
        raise ValueError("fine-tuning learning rates must be finite and positive")
    if not model.cfg.simple_conditioning:
        raise ValueError("separate appearance learning rate requires M3-Simple")
    base, appearance, patch = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            raise ValueError("full player fine-tuning requires all M3 parameters trainable")
        if name == "core.x_embedder.proj.weight":
            patch.append(parameter)
        elif name.startswith(("reference_encoder.", "roi_appearance_projector.")):
            appearance.append(parameter)
        else:
            base.append(parameter)
    if not base or not appearance or len(patch) != 1:
        raise ValueError("missing expected M3-Simple parameter groups")
    channels = model.core.roi_appearance_dim
    if not 0 < channels < patch[0].shape[1]:
        raise ValueError("invalid appended appearance channel count")
    return AppearanceAdamW([
        {"params": base, "name": "backbone_and_conditions", "lr": base_lr},
        {"params": appearance, "name": "appearance", "lr": appearance_lr},
        {"params": patch, "name": "joint_patch", "lr": base_lr,
         "appearance_lr": appearance_lr, "appearance_input_channels": channels},
    ], weight_decay=weight_decay)
