"""
References:
    - DiT: https://github.com/facebookresearch/DiT/blob/main/models.py
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing/blob/main/algorithms/diffusion_forcing/models/unet3d.py
    - Latte: https://github.com/Vchitect/Latte/blob/main/models/latte.py
"""


from dataclasses import dataclass, field
from math import prod
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from timm.models.vision_transformer import Mlp
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import MultiHeadRMSNorm
from .embeddings import TimestepEmbedder, RotaryEmbedding, apply_rotary_emb, \
    PixelPatchEmbedder, DepthPatchEmbedder
from .camera_util import camera_params_to_matrices
from .kv_cache import commit_current_kv, valid_history


def modulate(x, shift, scale):
    fixed_dims = [1] * len(shift.shape[1:])
    shift = shift.repeat(x.shape[0] // shift.shape[0], *fixed_dims)
    scale = scale.repeat(x.shape[0] // scale.shape[0], *fixed_dims)
    while shift.dim() < x.dim():
        shift = shift.unsqueeze(-2)
        scale = scale.unsqueeze(-2)
    return x * (1 + scale) + shift

def gate(x, g):
    fixed_dims = [1] * len(g.shape[1:])
    g = g.repeat(x.shape[0] // g.shape[0], *fixed_dims)
    while g.dim() < x.dim():
        g = g.unsqueeze(-2)
    return g * x


@torch.compiler.disable
def _kv_cache_rope_offset(rotary_emb, axial_freqs, global_start_idx, spatial_sequence_shape=None):
    """Compute RoPE position offset for KV cache mode. Disabled from torch.compile."""
    idx = global_start_idx.item() if isinstance(global_start_idx, torch.Tensor) else global_start_idx
    if spatial_sequence_shape is not None:
        encoded_offset = rotary_emb.get_axial_freqs(1, *spatial_sequence_shape, offset=idx).reshape(1, prod(spatial_sequence_shape), -1)
    else:
        encoded_offset = rotary_emb.get_axial_freqs(1, offset=idx)
    return axial_freqs + encoded_offset


@torch.compiler.disable
def _kv_cache_update(kv_cache, k, v, global_start_idx, T_new):
    """Update KV cache with new entries. Disabled from torch.compile to avoid recompilation."""
    idx = global_start_idx.item() if isinstance(global_start_idx, torch.Tensor) else global_start_idx
    global_end_idx = idx + T_new
    g_end = kv_cache["global_end_index"].item() if isinstance(kv_cache["global_end_index"], torch.Tensor) else kv_cache["global_end_index"]
    window_moved = global_end_idx > g_end

    cache_size = kv_cache["k"].shape[2]
    l_end = kv_cache["local_end_index"].item() if isinstance(kv_cache["local_end_index"], torch.Tensor) else kv_cache["local_end_index"]
    local_end_idx = l_end

    if not window_moved:
        local_end_idx = max(0, local_end_idx - T_new)

    if window_moved and local_end_idx + T_new > cache_size:
        n_remove = local_end_idx + T_new - cache_size
        n_keep = local_end_idx - n_remove
        kv_cache["k"][:, :, :n_keep] = kv_cache["k"][:, :, n_remove:local_end_idx].clone()
        kv_cache["v"][:, :, :n_keep] = kv_cache["v"][:, :, n_remove:local_end_idx].clone()
        local_end_idx -= n_remove

    kv_cache["k"][:, :, local_end_idx:local_end_idx+T_new] = k
    kv_cache["v"][:, :, local_end_idx:local_end_idx+T_new] = v

    if isinstance(kv_cache["global_end_index"], torch.Tensor):
        kv_cache["global_end_index"].fill_(global_end_idx)
        kv_cache["local_end_index"].fill_(local_end_idx + T_new)
    else:
        kv_cache["global_end_index"] = global_end_idx
        kv_cache["local_end_index"] = local_end_idx + T_new

    return kv_cache["k"][:, :, :local_end_idx+T_new], kv_cache["v"][:, :, :local_end_idx+T_new]


class PixelTemporalAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        is_causal: bool = True,
        qk_rms_norm: bool = False,
        flash_attn: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.rotary_emb = rotary_emb
        self.is_causal = is_causal
        self.qk_rms_norm = qk_rms_norm
        self.flash_attn = flash_attn

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache=None,
        global_start_idx=0,
        prefill=False,
        cache_write=True,
        return_kv_candidate=False,
    ):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B H W) h T d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B H W) h T d", h=self.heads)

        axial_freqs = self.rotary_emb.axial_freqs[:T]

        # When using KV cache, offset RoPE to encode global position
        if kv_cache is not None:
            axial_freqs = _kv_cache_rope_offset(self.rotary_emb, axial_freqs, global_start_idx, self.rotary_emb.spatial_sequence_shape)

        # Correct order: QK RMSNorm first, then RoPE
        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        q = apply_rotary_emb(axial_freqs, q)
        k = apply_rotary_emb(axial_freqs, k)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        current_k, current_v = k, v

        # Legacy prefill/decode writes remain available. Final-denoise rollout
        # uses cache_write=False and commits the returned candidate explicitly.
        if kv_cache is not None:
            if cache_write:
                k, v = _kv_cache_update(kv_cache, k, v, global_start_idx, T)
            else:
                history_k, history_v = valid_history(kv_cache)
                k = torch.cat((history_k, current_k), dim=2)
                v = torch.cat((history_v, current_v), dim=2)

        if self.flash_attn:
            print("Flash Attention is currently disabled.")
        else:
            # During prefill (T>1 with KV cache), causal mask is needed.
            # During single-frame decode, cache only contains past frames so no mask needed.
            cached_chunk = (
                self.is_causal and kv_cache is not None
                and not cache_write and T > 1
            )
            if cached_chunk:
                # SDPA's built-in non-square causal mask is upper-left aligned.
                # Cached chunks instead need every query to see all committed
                # history plus earlier/current positions in this chunk.
                history_length = k.shape[2] - T
                query_index = torch.arange(T, device=q.device)[:, None]
                key_index = torch.arange(k.shape[2], device=q.device)[None, :]
                attention_mask = key_index <= history_length + query_index
                x = F.scaled_dot_product_attention(
                    query=q, key=k, value=v,
                    attn_mask=attention_mask, is_causal=False,
                )
            else:
                use_causal = self.is_causal and (kv_cache is None or prefill)
                x = F.scaled_dot_product_attention(
                    query=q, key=k, value=v, is_causal=use_causal
                )

        x = rearrange(x, "(B H W) h T d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        if return_kv_candidate:
            return x, (current_k, current_v)
        return x

class PixelSpatialAxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        rotary_emb: RotaryEmbedding,
        qk_rms_norm: bool = False,
        flash_attn: bool = False,
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.rotary_emb = rotary_emb
        self.qk_rms_norm = qk_rms_norm
        self.flash_attn = flash_attn

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)

    def forward(self, x: torch.Tensor):
        B, T, H, W, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        k = rearrange(k, "B T H W (h d) -> (B T) h H W d", h=self.heads)
        v = rearrange(v, "B T H W (h d) -> (B T) h H W d", h=self.heads)

        # Correct order: QK RMSNorm first, then RoPE
        if self.qk_rms_norm:
            q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
            k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            q = rearrange(q, "(B T) h (H W) d -> (B T) h H W d", B=B, T=T, H=H, W=W)
            k = rearrange(k, "(B T) h (H W) d -> (B T) h H W d", B=B, T=T, H=H, W=W)

        freqs = rearrange(self.rotary_emb.axial_freqs, "(H W) f -> H W f", H=H, W=W)
        q = apply_rotary_emb(freqs, q)
        k = apply_rotary_emb(freqs, k)

        # prepare for attn
        q = rearrange(q, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        k = rearrange(k, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)
        v = rearrange(v, "(B T) h H W d -> (B T) h (H W) d", B=B, T=T, h=self.heads)

        if self.flash_attn:
            print("Flash Attn is disabled")
            #x = flash_attn.flash_attn_func(q, k, v, causal=False)
        else:
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)

        x = rearrange(x, "(B T) h (H W) d -> B T H W (h d)", B=B, H=H, W=W)
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x

class SpatioTemporalPixelDiTBlock(nn.Module):
    def __init__(
            self,
            hidden_size,
            num_heads,
            mlp_ratio=4.0,
            is_causal=True,
            qk_rms_norm=False,
            flash_attn=False,
            pixel_spatial_emb: Optional[RotaryEmbedding] = None,
            temporal_rotary_emb: Optional[RotaryEmbedding] = None,
    ):
        super().__init__()
        self.is_causal = is_causal
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.flash_attn = flash_attn

        self.s_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Pixel Spatial Self-Attn
        self.s_attn = PixelSpatialAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            rotary_emb=pixel_spatial_emb,
            qk_rms_norm=qk_rms_norm,
            flash_attn=self.flash_attn,
        )
        self.s_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.s_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.s_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        self.t_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Pixel Temporal Self-Attn
        self.t_attn = PixelTemporalAxialAttention(
            hidden_size,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            is_causal=is_causal,
            rotary_emb=temporal_rotary_emb,
            qk_rms_norm=qk_rms_norm,
            flash_attn=self.flash_attn
        )
        self.t_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.t_mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.t_adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))


    def forward(
        self,
        x,
        c,
        kv_cache=None,
        global_start_idx=0,
        prefill=False,
        cache_write=True,
        return_kv_candidate=False,
    ):
        B, T, H, W, D = x.shape

        # spatial block
        s_shift_msa, s_scale_msa, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.s_adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate(self.s_attn(modulate(self.s_norm1(x), s_shift_msa, s_scale_msa)), s_gate_msa)
        x = x + gate(self.s_mlp(modulate(self.s_norm2(x), s_shift_mlp, s_scale_mlp)), s_gate_mlp)

        # temporal block
        t_shift_msa, t_scale_msa, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.t_adaLN_modulation(c).chunk(6, dim=-1)
        temporal_result = self.t_attn(
            modulate(self.t_norm1(x), t_shift_msa, t_scale_msa),
            kv_cache=kv_cache,
            global_start_idx=global_start_idx,
            prefill=prefill,
            cache_write=cache_write,
            return_kv_candidate=return_kv_candidate,
        )
        if return_kv_candidate:
            temporal_out, kv_candidate = temporal_result
        else:
            temporal_out = temporal_result
        x = x + gate(temporal_out, t_gate_msa)
        x = x + gate(self.t_mlp(modulate(self.t_norm2(x), t_shift_mlp, t_scale_mlp)), t_gate_mlp)

        if return_kv_candidate:
            return x, kv_candidate
        return x

class PixelFinalLayer(nn.Module):
    """
    The final layer of pixel DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


@dataclass
class RasterizedVoxelDepthTransformerArgs:
    feature_dim: int = 32
    heads: int = 4
    mlp_ratio: float = 4.0
    output_dim: Optional[int] = None
    frequency_embedding_size: int = 32
    max_period: int = 10000
    num_layers: int = 192
    vol_render_mode: bool = False
    qk_rms_norm: bool = True
    is_causal_depth: bool = True #fixme inconsistent!!
    flash_attn: bool = False
    old: bool = False

    @property
    def name(self) -> str:
        return "RasterizedVoxelDepthTransformer"


class RasterizedVoxelDepthAttention(nn.Module):
    """
    Multi-head attention across depth layers for rasterized voxel latents.
    Each pixel (H, W) location computes attention across its depth layers independently.
    Uses causal masking so tokens only attend to tokens before them in depth (depth-sorted).

    Supports volume rendering configuration with single head and identity values.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        vol_render_mode: bool = False,
        qk_rms_norm: bool = True,
        is_causal: bool = True,
        flash_attn: bool = False
    ):
        super().__init__()
        self.vol_render_mode = vol_render_mode
        self.qk_rms_norm = qk_rms_norm
        self.is_causal = is_causal
        self.flash_attn = flash_attn

        if vol_render_mode:
            self.heads = 1
            self.dim_head = dim
        else:
            assert dim % heads == 0, f"dim={dim} must be divisible by heads={heads}"
            self.heads = heads
            self.dim_head = dim // heads

        self.inner_dim = self.heads * self.dim_head

        # Q, K projections (V is identity in vol_render mode)
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)

        if not vol_render_mode:
            self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
            self.to_out = nn.Linear(self.inner_dim, dim)

        if qk_rms_norm:
            self.q_rms_norm = MultiHeadRMSNorm(self.dim_head, self.heads)
            self.k_rms_norm = MultiHeadRMSNorm(self.dim_head, self.heads)

    def forward(self, voxel_latents: torch.Tensor, depth_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_latents: (B, H, W, num_layers, feature_dim) - rasterized voxel latents (depth-sorted)
            depth_embeddings: (B, H, W, num_layers, enc_dim) - depth positional embeddings

        Returns:
            output: (B, H, W, num_layers, feature_dim) - attended features
        """
        B, H, W, L, D = voxel_latents.shape

        # Combine voxel latents with depth embeddings for Q/K computation
        voxel_latents = rearrange(voxel_latents, 'b h w l c -> (b h w) l c', b=B, h=H, w=W)
        qk_input = voxel_latents + depth_embeddings

        # Generate depth-aware queries and keys
        q = self.to_q(qk_input)
        k = self.to_k(qk_input)

        # Reshape for multi-head attention
        q = rearrange(q, 'bhw l (h d) -> bhw h l d', h=self.heads)
        k = rearrange(k, 'bhw l (h d) -> bhw h l d', h=self.heads)

        if self.qk_rms_norm:
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)

        if self.vol_render_mode:
            v = rearrange(voxel_latents, 'bhw l (h d) -> bhw h l d', h=self.heads)
        else:
            #TODO change naming of qk_input # Standard learned values from voxel_latents + depth_embeddings
            v = self.to_v(qk_input)
            v = rearrange(v, 'bhw l (h d) -> bhw h l d', h=self.heads)

        # Causal attention: depth-aware Q/K compute weights for V
        if self.flash_attn:
            print("Flash attention is currently disabled.")
            #attn_output = flash_attn.flash_attn_func(q, k, v, causal=self.is_causal)
        else:
            attn_output = F.scaled_dot_product_attention(
                query=q, key=k, value=v, is_causal=self.is_causal
            )
        attn_output = rearrange(attn_output, 'bhw h l d -> bhw l (h d)')

        if self.vol_render_mode:
            # Single head output, keep original dimension
            output = attn_output
        else:
            # Multi-head output with projection
            output = self.to_out(attn_output)

        output = rearrange(output, '(b h w) l d -> b h w l d', b=B, h=H, w=W)

        return output


class RasterizedVoxelDepthTransformer(nn.Module):
    """
    Standalone transformer block for aggregating voxel latents across depth layers.
    Processes each pixel's depth layers independently with causal self-attention.

    Uses TimestepEmbedder from .embeddings for depth encoding.
    """

    def __init__(
        self,
        feature_dim: int = 8,
        heads: int = 4,
        mlp_ratio: float = 4.0,
        num_layers : int = 64,
        output_dim: Optional[int] = None,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        vol_render_mode: bool = False,
        qk_rms_norm: bool = True,
        flash_attn: bool = False,
        is_causal_depth: bool = True,
        old: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.vol_render_mode = vol_render_mode
        self.num_layers = num_layers
        self.is_causal_depth = is_causal_depth
        self.flash_attn = flash_attn
        self.old = old

        if self.vol_render_mode:
            self.output_dim = feature_dim
        else:
            self.output_dim = output_dim or feature_dim

        mlp_hidden_dim = int(feature_dim * mlp_ratio)

        # Depth positional encoding using TimestepEmbedder
        # Assume depth values are already normalized [0,1]
        self.depth_embedder = TimestepEmbedder(
            hidden_size=feature_dim,
            frequency_embedding_size=frequency_embedding_size,
            max_period=max_period,
        )

        # Attention block with causal masking and vol_render support
        self.norm1 = nn.LayerNorm(feature_dim)
        self.attn = RasterizedVoxelDepthAttention(
            dim=feature_dim,
            heads=heads,
            vol_render_mode=vol_render_mode,
            qk_rms_norm=qk_rms_norm,
            flash_attn=flash_attn,
            is_causal=is_causal_depth
        )

        # Aggregation MLP (used in both configurations)
        self.norm_2 = nn.LayerNorm(feature_dim)
        if not self.vol_render_mode and not self.old:
            self.aggregation_mlp_0 = Mlp(
                in_features=feature_dim*num_layers,
                hidden_features=mlp_hidden_dim*2,
                out_features=feature_dim,
                drop=0.0
            )
        self.aggregation_mlp = Mlp(
            in_features=feature_dim,
            hidden_features=mlp_hidden_dim,
            out_features=self.output_dim,
            drop=0.0
        )

    def forward(
        self,
        voxel_latents: torch.Tensor,
        depth_values: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            voxel_latents: (B, H, W, num_layers, feature_dim) - rasterized voxel latents (depth-sorted)
            depth_values: (B, H, W, num_layers) - normalized depth values [0,1] (depth-sorted)

        Returns:
            aggregated_features: (B, H, W, output_dim) - single feature map per pixel
        """
        B, H, W, L, D = voxel_latents.shape

        # Flatten for TimestepEmbedder: (B*H*W*L,)
        depth_flat = rearrange(depth_values, 'b h w l -> (b h w l)')

        # Get depth embeddings using TimestepEmbedder
        depth_embeddings_flat = self.depth_embedder(depth_flat*1000)  # (B*H*W*L, feature_dim)

        # Reshape back: (B, H, W, L, feature_dim) -> (B, H, W, feature_dim, L)
        depth_embeddings = rearrange(
            depth_embeddings_flat,
            '(b h w l) d -> (b h w) l d',
            b=B, h=H, w=W, l=L
        )

        # Causal self-attention with depth-aware Q/K and pure/mixed V
        if not self.vol_render_mode:
            voxel_latents = self.norm1(voxel_latents)

        x = self.attn(voxel_latents, depth_embeddings) #B H W L D

        if self.vol_render_mode:
            # Output should be an actual vol rendering of the voxel latents using attn
            x = torch.sum(x, dim=-2)  # (B, H, W, D)
        else:
            x += voxel_latents #residual
            if self.old:
                x = torch.sum(x, dim=-2)
            else:
                x = rearrange(x, 'b h w l d -> b h w (l d)')
                x = self.aggregation_mlp_0(x)
            x = self.norm_2(x)
            x = rearrange(x, 'b h w d -> (b h w) d')  # flatten depth for MLP
            x = self.aggregation_mlp(x)  # (B, H, W, output_dim)
            x = rearrange(x, '(b h w) d -> b h w d', b=B, h=H, w=W)

        return x


@dataclass(kw_only=True)
class FrameDepthStackPixelDitDenoiserArgs:
    """Configuration for DiT Denoiser."""

    input_h : int = 36
    input_w : int = 64
    patch_size : int = 2
    in_channels : int = 16
    hidden_size : int = 1024
    depth : int = 12
    num_heads : int = 16
    mlp_ratio : int = 4.0
    raster_cond_shape : Tuple[int, ...] = (32, 36, 64)
    raster_cond_patch_size : int = 2
    action_cond_dim : int = 23
    extra_condition_dim : int = 0
    actor_condition_dim : int = 0
    deep_condition_reinjection : bool = False
    hud_condition_dim : int = 0
    context_window_size : int = 16
    cache_window_size : int = 32
    is_causal : bool = True
    use_condition_mask : bool = False
    rotary_max_freq : int = 256
    voxel_dim : int = 48
    num_freqs : int = 8
    qk_rms_norm : bool = True
    flash_attn : bool = False
    detach_raster_grad : bool = False
    learned_padding : bool = False
    dropout_p_pixel : float = 0.0
    gradient_checkpointing : bool = False
    depth_patch_size : int = 6
    depth_stride_size : int = 4
    depth_output_dim : int = 16
    aggregation_config : RasterizedVoxelDepthTransformerArgs = field(
        default_factory=RasterizedVoxelDepthTransformerArgs
    )

    @property
    def name(self) -> str:
        return "FrameDepthStackPixelDiT"


class FrameDepthStackPixelDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone and stacking voxel latent raster + pixel latent.
    """

    def __init__(
            self,
            input_h=18,
            input_w=32,
            patch_size=2,
            in_channels=16,
            hidden_size=1024,
            depth=12,
            num_heads=16,
            mlp_ratio=4.0,
            raster_cond_shape=(8, 36, 64),
            raster_cond_patch_size=2,
            action_cond_dim=23,
            extra_condition_dim=0,
            actor_condition_dim=0,
            deep_condition_reinjection=False,
            hud_condition_dim=0,
            context_window_size=32,
            cache_window_size=32,
            is_causal=True,
            use_condition_mask=False,
            rotary_max_freq=256,
            voxel_dim=48,
            num_freqs=8,
            qk_rms_norm=False,
            detach_raster_grad=False,
            flash_attn=False,
            learned_padding=False,
            dropout_p_pixel=0.0,
            gradient_checkpointing=False,
            depth_patch_size=6,
            depth_stride_size=4,
            depth_output_dim=16,
            aggregation_config=None,
            use_kv_cache=False,
            batch_size=None,
    ):
        super().__init__()
        logger.info("FrameDepthStack DiT")
        self.input_h = input_h
        self.input_w = input_w
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.depth_stride_size = depth_stride_size  # non-overlapping
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.depth = depth
        self.context_window_size = context_window_size
        self.cache_window_size = cache_window_size
        if self.cache_window_size <= 0:
            raise ValueError("cache_window_size must be positive")
        self.is_causal = is_causal
        self.use_condition_mask = use_condition_mask
        self.detach_raster_grad = detach_raster_grad
        self.flash_attn = flash_attn
        self.learned_padding = learned_padding
        self.voxel_dim = voxel_dim
        self.num_freqs = num_freqs
        self.use_kv_cache = use_kv_cache
        self.batch_size = batch_size
        self.kv_caches = None
        self.raster_cache = None
        self.qk_rms_norm = qk_rms_norm
        self.gradient_checkpointing = gradient_checkpointing
        self.deep_condition_reinjection = bool(deep_condition_reinjection)

        # Calculate depth-compressed raster dimensions
        raster_num_layers = self.voxel_dim * 4
        raster_layers_out = (raster_num_layers - depth_patch_size) // depth_stride_size + 1
        raster_channels_total = raster_layers_out * depth_output_dim
        total_in_channels = in_channels + raster_channels_total
        #embedders. Takes depth sorted features + depths as input. Concats depth features and depth positional encodings then convolves with 1d filter.
        self.depth_embedder = DepthPatchEmbedder(
            num_layers=raster_num_layers,
            patch_size=depth_patch_size,
            stride=depth_stride_size,
            in_chans=raster_cond_shape[0],
            out_chans=depth_output_dim,
            use_depth_pos_enc=True,
            num_freqs=num_freqs,
        )
        #self.pixel_dropout = nn.Dropout3d(dropout_p_pixel)
        self.x_embedder = PixelPatchEmbedder(input_h, input_w, patch_size, total_in_channels, hidden_size, flatten=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.action_embedder = nn.Linear(action_cond_dim, hidden_size) if action_cond_dim > 0 else nn.Identity()
        self.window_prefix_embedding = nn.Parameter(torch.zeros(hidden_size))
        self.extra_condition_embedder = (
            nn.Linear(extra_condition_dim, hidden_size) if extra_condition_dim > 0 else None
        )
        self.actor_condition_embedder = (
            nn.Conv2d(actor_condition_dim, hidden_size, kernel_size=patch_size, stride=patch_size)
            if actor_condition_dim > 0
            else None
        )
        self.condition_mask_embedder = nn.Linear(1, hidden_size) if use_condition_mask else None
        if self.deep_condition_reinjection:
            if hud_condition_dim < 1:
                raise ValueError("deep condition reinjection requires hud_condition_dim")
            self.hud_condition_embedder = nn.Conv2d(
                hud_condition_dim, hidden_size, kernel_size=patch_size, stride=patch_size
            )
            self.condition_reinjectors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.LayerNorm(hidden_size),
                        nn.SiLU(),
                        nn.Linear(hidden_size, hidden_size),
                    )
                    for _ in range(depth)
                ]
            )
        else:
            self.hud_condition_embedder = None
            self.condition_reinjectors = None

        #spatial encodings
        self.pixel_spatial_emb = RotaryEmbedding(
            dim=hidden_size // num_heads // 2,
            freqs_for="pixel",
            max_freq=rotary_max_freq,
            cache_if_possible=True,
            spatial_sequence_shape=self.x_embedder.grid_size,
        )
        self.temporal_rotary_emb = RotaryEmbedding(
            dim=hidden_size // num_heads,
            cache_if_possible=True,
            temporal_context_length=context_window_size,
        )

        ## aggregation block
        if aggregation_config is not None:
            # Depth stacking never uses the legacy cross-attention RoPE.
            self.cross_raster_emb = None
            #initialize rasterizer

            if self.learned_padding == False:
                background_feature = torch.zeros(raster_cond_shape[0])  # (1, 1, C)
                self.register_buffer("background_feature", background_feature, persistent=False)
            else:
                logger.info("Learned Padding Mode")
                self.background_feature = nn.Parameter(torch.ones(raster_cond_shape[0]), requires_grad=True)
                self.detach_raster_grad = False

            from .voxel_rasterizer import VoxelMeshRasterizer
            self.rasterizer = VoxelMeshRasterizer(
                dim=self.voxel_dim,
                height=raster_cond_shape[1],
                width=raster_cond_shape[2],
                max_layers=self.voxel_dim*4,
            )
        else:
            self.cross_raster_emb = None
            self.rasterizer = None

        self.blocks = nn.ModuleList(
            [
                SpatioTemporalPixelDiTBlock(
                    hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    is_causal=is_causal,
                    qk_rms_norm=qk_rms_norm,
                    flash_attn=self.flash_attn,
                    pixel_spatial_emb=self.pixel_spatial_emb,
                    temporal_rotary_emb=self.temporal_rotary_emb,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = PixelFinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Preserve the pretrained denoiser exactly until the new condition-mask
        # path starts learning.  This also keeps old checkpoints loadable with
        # strict=False without changing x_embedder's input-channel shape.
        if self.condition_mask_embedder is not None:
            nn.init.constant_(self.condition_mask_embedder.weight, 0)
            nn.init.constant_(self.condition_mask_embedder.bias, 0)
        if self.extra_condition_embedder is not None:
            nn.init.constant_(self.extra_condition_embedder.weight, 0)
            nn.init.constant_(self.extra_condition_embedder.bias, 0)
        if self.actor_condition_embedder is not None:
            nn.init.constant_(self.actor_condition_embedder.weight, 0)
            nn.init.constant_(self.actor_condition_embedder.bias, 0)
        if self.condition_reinjectors is not None:
            # This makes an old checkpoint an exact functional warm start.
            for adapter in self.condition_reinjectors:
                nn.init.constant_(adapter[-1].weight, 0)
                nn.init.constant_(adapter[-1].bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.s_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.s_adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.t_adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def _initialize_kv_caches(self, batch_size=None, dtype=None):
        patched_pix_shape = self.x_embedder.grid_size
        H, W = patched_pix_shape
        head_dim = self.hidden_size // self.num_heads
        device = self.x_embedder.proj.weight.device
        dtype = dtype or self.x_embedder.proj.weight.dtype

        if batch_size is not None:
            self.batch_size = batch_size
        if self.batch_size is None:
            raise ValueError("batch_size is required to initialize KV caches")

        self.kv_caches = []
        for _ in range(self.depth):
            cache_shape = (self.batch_size * H * W, self.num_heads, self.cache_window_size, head_dim)
            kv_cache = {
                "k": torch.zeros(cache_shape, device=device, dtype=dtype),
                "v": torch.zeros(cache_shape, device=device, dtype=dtype),
                "global_end_index": torch.tensor(0, dtype=torch.long, device=device),
                "local_end_index": torch.tensor(0, dtype=torch.long, device=device),
            }
            self.kv_caches.append(kv_cache)
        self.raster_cache = {"raster_cond": None, "global_start_idx": -1}

    def init_kv_cache(self, batch_size, dtype=None):
        """Allocate caches in the active inference dtype."""
        self._initialize_kv_caches(batch_size=batch_size, dtype=dtype)
        return self.kv_caches

    def commit_kv_candidates(self, candidates, frame_index):
        """Atomically commit one final-denoise candidate per temporal block."""
        if self.kv_caches is None:
            raise RuntimeError("init_kv_cache must be called before commit")
        if len(candidates) != len(self.kv_caches):
            raise ValueError("candidate count must match temporal block count")
        expected_indices = {
            int(cache["global_end_index"].item()) for cache in self.kv_caches
        }
        if expected_indices != {frame_index}:
            raise ValueError(
                f"All layers must expect frame_index={frame_index}, got {expected_indices}"
            )
        for cache, (current_k, current_v) in zip(self.kv_caches, candidates):
            commit_current_kv(cache, current_k, current_v, frame_index)

    def _reset_kv_cache_indices(self):
        """Reset KV cache indices to zero without reallocating buffers.

        Called before a prefill pass so the entire current window is
        written fresh into the cache.
        """
        if self.kv_caches:
            for cache in self.kv_caches:
                cache["global_end_index"].fill_(0)
                cache["local_end_index"].fill_(0)

    def unpatchify(self, x):
        """
        x: (N, H, W, patch_size**2 * C)
        vox: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        H, W = x.shape[1:3]

        x = x.reshape(shape=(x.shape[0], H, W, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        vox = x.reshape(shape=(x.shape[0], c, H * p, W * p))
        return vox

    def project_raster(self, external_cond, dtype):
        # A projection belongs to this call's state, never merely to a frame index.
        if "raster_features" in external_cond:
            raster_cond = external_cond["raster_features"]
            raster_depth = external_cond["raster_depth"]
            raster_cond = rearrange(raster_cond, "b t h w l c -> (b t) h w l c")
            raster_depth = rearrange(raster_depth, "b t h w l -> (b t) h w l")
        else:
            if self.rasterizer is None:
                raise ValueError("provide raster_features/raster_depth or enable GPU rasterization")
            voxel_latents, camera = external_cond['voxel_latents'], external_cond['camera']
            view, proj = camera_params_to_matrices(camera, image_width=self.input_w, image_height=self.input_h)
            voxel_latents = rearrange(voxel_latents, "b t c x y z -> (b t) (x y z) c")
            view = rearrange(view, "b t p q -> (b t) p q")
            proj = rearrange(proj, "b t p q -> (b t) p q")
            raster_cond, raster_depth = self.rasterizer.rasterize(
                voxel_latents, view, proj, self.background_feature)
            raster_cond = rearrange(raster_cond, "l bt h w c -> bt h w l c")
            raster_depth = rearrange(raster_depth, "l bt h w -> bt h w l")
        raster_cond = self.depth_embedder(raster_cond.to(dtype), raster_depth.to(dtype))
        raster_cond = rearrange(raster_cond, 'bt h w l c -> bt (l c) h w')

        return raster_cond

    def forward(
        self,
        x,
        t,
        external_cond=None,
        global_start_idx=None,
        prefill=False,
        return_features=False,
        cache_write=True,
        return_kv_candidates=False,
        block_callback=None,
    ):
        """
        Forward pass of DiT.
        x: (B, T, C, H, W) tensor of spatial inputs (pix or latent pix)
        t: (B, T,) tensor of diffusion timesteps
        prefill: if True, enables causal masking with KV cache for multi-frame prefill passes
        """
        if global_start_idx is None:
            global_start_idx = 0

        B, T, C, H, W = x.shape

        raster_cond = external_cond.get("raster_embedding")
        if raster_cond is None:
            raster_cond = self.project_raster(external_cond, x.dtype)

        condition_tokens = None
        if self.deep_condition_reinjection:
            # Reuse the trained raster slice of the patch embedder so a warm
            # start already has meaningful aligned 3D condition tokens.
            raster_tokens = F.conv2d(
                raster_cond,
                self.x_embedder.proj.weight[:, self.in_channels:],
                bias=None,
                stride=self.x_embedder.proj.stride,
                padding=self.x_embedder.proj.padding,
                dilation=self.x_embedder.proj.dilation,
                groups=self.x_embedder.proj.groups,
            )
            condition_tokens = rearrange(
                raster_tokens, "(b t) d h w -> b t h w d", b=B, t=T
            )

        # Stack pixel embedding and raster embedding
        x = rearrange(x, "b t c h w -> (b t) c h w")
        x = torch.cat([x, raster_cond], dim=1)  # (bt, 16+64, h, w)

        # embed x
        x = self.x_embedder(x)  # (B*T, C, H, W) -> (B*T, H/2, W/2, D) , C = 16, D = d_model

        actor_tokens = None
        if self.actor_condition_embedder is not None:
            actor_condition = external_cond.get("actor_spatial_condition")
            if actor_condition is None:
                raise ValueError("actor_spatial_condition is required when actor_condition_dim > 0")
            if actor_condition.shape[:2] != (B, T) or actor_condition.shape[-2:] != (H, W):
                raise ValueError(
                    "actor_spatial_condition must be [B,T,C,H,W] with "
                    f"B,T,H,W={(B, T, H, W)}, got {tuple(actor_condition.shape)}"
                )
            actor_tokens = self.actor_condition_embedder(
                rearrange(actor_condition, "b t c h w -> (b t) c h w").to(x.dtype)
            )
            x = x + rearrange(actor_tokens, "bt d h w -> bt h w d")
            if condition_tokens is not None:
                condition_tokens = condition_tokens + rearrange(
                    actor_tokens, "(b t) d h w -> b t h w d", b=B, t=T
                )

        x = rearrange(x, "(b t) h w d -> b t h w d", b=B, t=T)
        # embed noise steps
        t = rearrange(t, "b t -> (b t)")
        c = self.t_embedder(t)  # (N, D)
        c = rearrange(c, "(b t) d -> b t d", b=B, t=T)
        state_condition = torch.zeros_like(c)

        if self.condition_mask_embedder is not None:
            condition_mask = external_cond.get("condition_mask")
            if condition_mask is None:
                condition_mask = torch.zeros((B, T), device=x.device, dtype=x.dtype)
            if condition_mask.shape != (B, T):
                raise ValueError(
                    f"condition_mask must have shape {(B, T)}, got {tuple(condition_mask.shape)}"
                )
            mask_embedding = self.condition_mask_embedder(condition_mask.to(x.dtype).unsqueeze(-1))
            x = x + mask_embedding[:, :, None, None, :]

        if external_cond:
            action_cond = external_cond['action']
            action_embedding = self.action_embedder(action_cond)
            c += action_embedding
            state_condition += action_embedding
            action_prefix_mask = external_cond.get("action_prefix_mask")
            if action_prefix_mask is not None:
                if action_prefix_mask.shape != (B, T):
                    raise ValueError(
                        f"action_prefix_mask must have shape {(B, T)}, "
                        f"got {tuple(action_prefix_mask.shape)}"
                    )
                prefix_embedding = (
                    action_prefix_mask.to(c.dtype).unsqueeze(-1)
                    * self.window_prefix_embedding.to(c.dtype)
                )
                c += prefix_embedding
                state_condition += prefix_embedding
            if self.extra_condition_embedder is not None:
                extra_condition = external_cond.get("extra_condition")
                if extra_condition is not None:
                    if extra_condition.shape[:2] != (B, T):
                        raise ValueError(
                            "extra_condition must start with shape "
                            f"{(B, T)}, got {tuple(extra_condition.shape)}"
                        )
                    extra_embedding = self.extra_condition_embedder(extra_condition.to(c.dtype))
                    c += extra_embedding
                    state_condition += extra_embedding
        if condition_tokens is not None:
            hud_condition = external_cond.get("hud_condition")
            expected_hud = (B, T, self.hud_condition_embedder.in_channels, H, W)
            if hud_condition is None or tuple(hud_condition.shape) != expected_hud:
                raise ValueError(f"hud_condition must have shape {expected_hud}")
            hud_tokens = self.hud_condition_embedder(
                rearrange(hud_condition, "b t c h w -> (b t) c h w").to(x.dtype)
            )
            condition_tokens = condition_tokens + rearrange(
                hud_tokens, "(b t) d h w -> b t h w d", b=B, t=T
            )
            condition_tokens = condition_tokens + state_condition[:, :, None, None]
        # Convert to tensor for torch.compile compatibility (avoids dynamo guards on int values)
        global_start_idx_t = torch.tensor(global_start_idx, device=x.device)
        kv_candidates = []
        for i, block in enumerate(self.blocks):
            if self.condition_reinjectors is not None:
                x = x + self.condition_reinjectors[i](condition_tokens)
            cache = self.kv_caches[i] if self.kv_caches else None
            if self.gradient_checkpointing and self.training and cache is None:
                x = checkpoint(
                    lambda x_, c_, block_=block: block_(
                        x_,
                        c_,
                        kv_cache=None,
                        global_start_idx=global_start_idx_t,
                        prefill=prefill,
                    ),
                    x,
                    c,
                    use_reentrant=False,
                )
            else:
                block_result = block(
                    x,
                    c,
                    kv_cache=cache,
                    global_start_idx=global_start_idx_t,
                    prefill=prefill,
                    cache_write=cache_write,
                    return_kv_candidate=return_kv_candidates,
                )
                if return_kv_candidates:
                    x, candidate = block_result
                    kv_candidates.append(candidate)
                else:
                    x = block_result
            if block_callback is not None:
                block_callback(i, x)

        features = x
        x = self.final_layer(x, c)  # (N, T, H, W, patch_size ** 2 * out_channels)
        # unpatchify
        x = rearrange(x, "b t h w d -> (b t) h w d")
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=B, t=T)

        if return_features and return_kv_candidates:
            return x, features, kv_candidates
        if return_features:
            return x, features
        if return_kv_candidates:
            return x, kv_candidates
        return x



        return x
