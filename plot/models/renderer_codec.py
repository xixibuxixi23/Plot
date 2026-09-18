"""Frozen 2DAction Pixel VAE interface with explicit RGB/latent layouts."""
from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy

import torch
from torch import nn

from .renderer_backbone.vae_pixel import ViTVae, ViTVaeArgs
from .latent_normalization import normalization_from_run_config, validate_latent_normalization


class RendererCodec(nn.Module):
    def __init__(self, weights, cfg: ViTVaeArgs | None = None, *, latent_normalization=None):
        super().__init__()
        self.cfg = cfg or ViTVaeArgs()
        self.latent_normalization = validate_latent_normalization(
            latent_normalization, self.cfg.latent_dim
        )
        for name, default in (("mean", 0.0), ("std", 1.0)):
            values = self.latent_normalization.get(name, [default] * self.cfg.latent_dim)
            self.register_buffer(
                f"latent_{name}", torch.tensor(values, dtype=torch.float32).view(1, 1, -1, 1, 1)
            )
        self.vae = ViTVae(**asdict(self.cfg)).eval().requires_grad_(False)
        self.vae.load_state_dict({k.removeprefix("vae."): v for k,v in weights.items()}, strict=True)

    @classmethod
    def from_run_config(cls, weights, run_config, cfg: ViTVaeArgs | None = None):
        return cls(weights, cfg, latent_normalization=normalization_from_run_config(run_config))

    def get_config(self):
        return {"latent_normalization": deepcopy(self.latent_normalization)}

    def normalize_latent(self, latent):
        if self.latent_normalization["mode"] == "none":
            return latent
        # Compute the affine transform in FP32, preserving the VAE output dtype
        # so BF16 rollout does not silently allocate FP32 KV caches.
        return ((latent.float() - self.latent_mean.float()) / self.latent_std.float()).to(latent.dtype)

    def denormalize_latent(self, latent):
        if self.latent_normalization["mode"] == "none":
            return latent
        return (latent.float() * self.latent_std.float() + self.latent_mean.float()).to(latent.dtype)

    def train(self, mode=True):
        # This module is always frozen, even when a parent model enters train().
        return super().train(False)

    @torch.no_grad()
    def encode(self, rgb, chunk_size=4):
        """RGB [B,T,3,H,W] in [0,1] -> latent [B,T,C,H/10,W/10]."""
        if rgb.ndim != 5 or rgb.shape[2:] != (3,self.cfg.input_height,self.cfg.input_width):
            raise ValueError("RGB shape does not match the Pixel VAE configuration")
        b,t = rgb.shape[:2]
        flat = rgb.flatten(0,1)*2-1
        latent = torch.cat([self.vae.encode(x).mode() for x in flat.split(chunk_size)])
        h,w = self.cfg.input_height//self.cfg.patch_size, self.cfg.input_width//self.cfg.patch_size
        latent = latent.reshape(b,t,h,w,self.cfg.latent_dim).permute(0,1,4,2,3).contiguous()
        return self.normalize_latent(latent)

    def _decode(self, latent, chunk_size):
        h,w = self.cfg.input_height//self.cfg.patch_size, self.cfg.input_width//self.cfg.patch_size
        if latent.ndim != 5 or latent.shape[2:] != (self.cfg.latent_dim,h,w):
            raise ValueError("latent shape does not match the Pixel VAE configuration")
        b,t = latent.shape[:2]
        latent = self.denormalize_latent(latent)
        tokens = latent.permute(0,1,3,4,2).reshape(b*t,h*w,self.cfg.latent_dim)
        rgb = torch.cat([self.vae.decode(x) for x in tokens.split(chunk_size)])
        return ((rgb+1)/2).clamp(0,1).reshape(b,t,3,self.cfg.input_height,self.cfg.input_width)

    @torch.no_grad()
    def decode(self, latent, chunk_size=4):
        """Completed latent frames -> RGB [B,T,3,H,W] in [0,1]."""
        return self._decode(latent, chunk_size)

    def decode_for_loss(self, latent, chunk_size=1):
        """Decode selected frames while retaining gradients to their latents.

        VAE parameters remain frozen; this graph is used only to send a
        full-resolution pixel loss back into the renderer prediction.
        """
        return self._decode(latent, chunk_size)
