"""Static PERSIST adaptation: shared image/ray tokens condition one voxel latent.

Spatial/cross-attention weights transfer from PERSIST-S. Temporal and action
modules are omitted; player order has no learned embedding or causal meaning.
"""

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import Mlp

from plot.models.renderer_backbone.attention import MultiHeadRMSNorm
from plot.models.renderer_backbone.embeddings import (
    AbsolutePositionEmbedder,
    PixelPatchEmbedder,
    TimestepEmbedder,
    VoxelPatchEmbedder,
)


class Attention(nn.Module):
    def __init__(self, width, heads, cross=False):
        super().__init__()
        self.heads, self.cross = heads, cross
        if cross:
            self.to_q = nn.Linear(width, width, bias=False)
            self.to_kv = nn.Linear(width, width * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(width, width * 3, bias=False)
        self.to_out = nn.Linear(width, width)
        self.q_rms_norm = MultiHeadRMSNorm(width // heads, heads)
        self.k_rms_norm = MultiHeadRMSNorm(width // heads, heads)

    def forward(self, x, context=None, mask=None):
        q, k, v = (
            (self.to_q(x), *self.to_kv(context).chunk(2, -1))
            if self.cross
            else self.to_qkv(x).chunk(3, -1)
        )

        def split(t):
            return t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)

        q, k, v = map(split, (q, k, v))
        h = F.scaled_dot_product_attention(
            self.q_rms_norm(q),
            self.k_rms_norm(k),
            v,
            attn_mask=None if mask is None else mask[:, None, None],
        )
        return self.to_out(h.transpose(1, 2).flatten(2))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        for prefix in ("s", "cross"):
            setattr(
                self, prefix + "_norm1", nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            )
            setattr(
                self, prefix + "_norm2", nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
            )
            setattr(self, prefix + "_attn", Attention(width, heads, cross=prefix == "cross"))
            setattr(
                self,
                prefix + "_mlp",
                Mlp(width, width * 4, act_layer=lambda: nn.GELU(approximate="tanh")),
            )
            setattr(
                self,
                prefix + "_adaLN_modulation",
                nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6)),
            )

    def forward(self, x, c, context, mask):
        for prefix in ("s", "cross"):
            shift, scale, gate, shift2, scale2, gate2 = getattr(self, prefix + "_adaLN_modulation")(
                c
            ).chunk(6, -1)
            h = getattr(self, prefix + "_norm1")(x) * (1 + scale[:, None]) + shift[:, None]
            h = (
                getattr(self, prefix + "_attn")(h, context, mask)
                if prefix == "cross"
                else self.s_attn(h)
            )
            x = x + gate[:, None] * h
            h = getattr(self, prefix + "_norm2")(x) * (1 + scale2[:, None]) + shift2[:, None]
            x = x + gate2[:, None] * getattr(self, prefix + "_mlp")(h)
        return x


class FinalLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.norm_final = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(width, 8 * 48)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, -1)
        return self.linear(self.norm_final(x) * (1 + scale[:, None]) + shift[:, None])


class MultiViewVoxelDiT(nn.Module):
    def __init__(self, width=1024, depth=12, heads=16):
        super().__init__()
        self.config = dict(width=width, depth=depth, heads=heads)
        self.x_embedder = VoxelPatchEmbedder(12, 12, 12, 2, 48, width)
        self.cross_cond_embedder = PixelPatchEmbedder(36, 64, 2, 22, width)
        self.t_embedder = TimestepEmbedder(width)
        coords = torch.stack(torch.meshgrid(*[torch.arange(6)] * 3, indexing="ij"), -1).reshape(
            -1, 3
        )
        self.register_buffer(
            "position",
            AbsolutePositionEmbedder(width, in_channels=3, pos_range=2)(coords),
            persistent=False,
        )
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(depth)])
        self.final_layer = FinalLayer(width)

    def load_persist(self, state):
        own = self.state_dict()
        selected = {k: state[k] for k in own if k in state and state[k].shape == own[k].shape}
        missing = sorted(set(own) - set(selected))
        if missing:
            raise ValueError(f"Incomplete static backbone transfer: {missing}")
        self.load_state_dict(selected, strict=True)
        return dict(
            tensors=len(selected),
            parameters=sum(v.numel() for v in selected.values()),
            omitted_source_tensors=len(state) - len(selected),
        )

    def forward(self, x, t, image_rays, view_valid, spatial_condition=None):
        # image_rays [B,V,22,36,64]. Invalid views cannot contribute to attention.
        b, views = image_rays.shape[:2]
        if not view_valid.any(-1).all():
            raise ValueError("Each scene needs at least one valid input view")
        clean = torch.where(view_valid[:, :, None, None, None], image_rays, 0)
        context = self.cross_cond_embedder(clean.flatten(0, 1)).reshape(b, views * 576, -1)
        mask = view_valid[:, :, None].expand(-1, -1, 576).reshape(b, -1)
        h = self.x_embedder(x) + self.position[None]
        if spatial_condition is not None:
            h = h + spatial_condition
        c = self.t_embedder(t * 1000)
        for block in self.blocks:
            h = block(h, c, context, mask)
        h = self.final_layer(h, c).reshape(b, 6, 6, 6, 2, 2, 2, 48)
        return h.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(b, 48, 12, 12, 12)
