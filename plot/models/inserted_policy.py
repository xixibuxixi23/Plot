"""Chunked read-only M4 tokens inserted into the last frozen M3 DiT blocks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from plot.policy_schema import FAMILIES, FAMILY_PROFILES, FAMILY_TO_ID, PROFILES, PROFILE_TO_ID
from .structured_action import StructuredActionHead, constrain_peaceful_logits


@dataclass(frozen=True)
class InsertedPolicyArgs:
    num_policy_blocks: int = 4
    horizons: int = 8
    history_frames: int = 8
    text_hidden_size: int = 768
    mlp_ratio: float = 4.


class ReadOnlyPolicyBlock(nn.Module):
    """One policy-token block; detached video patches are K/V and never queries."""

    def __init__(self, hidden, heads, text_hidden, mlp_ratio=4.):
        super().__init__()
        self.heads, self.head_dim = heads, hidden // heads
        self.policy_norm, self.video_norm = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.policy_q = nn.Linear(hidden, hidden)
        self.policy_kv = nn.Linear(hidden, 2 * hidden)
        self.video_kv = nn.Linear(hidden, 2 * hidden)
        self.spatial_out = nn.Linear(hidden, hidden)
        self.temporal_norm = nn.LayerNorm(hidden)
        self.temporal_qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.temporal_out = nn.Linear(hidden, hidden)
        self.text_projection = nn.Linear(text_hidden, hidden)
        self.text_norm = nn.LayerNorm(hidden)
        self.text_attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.current_text_gate = nn.Parameter(torch.tensor(1.))
        self.mlp_norm = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(nn.Linear(hidden, int(hidden * mlp_ratio)),
                                 nn.GELU(approximate="tanh"),
                                 nn.Linear(int(hidden * mlp_ratio), hidden))

    def _heads(self, value):
        return value.view(*value.shape[:-1], self.heads, self.head_dim).transpose(-3, -2)

    def forward(self, token, video, text, text_mask, valid):
        b, t, d = token.shape
        patches = video.detach().flatten(2, 3)
        q = self._heads(self.policy_q(self.policy_norm(token)).unsqueeze(2))
        video_k, video_v = self.video_kv(self.video_norm(patches)).chunk(2, -1)
        own_k, own_v = self.policy_kv(self.policy_norm(token)).chunk(2, -1)
        k = self._heads(torch.cat((video_k, own_k.unsqueeze(2)), 2))
        v = self._heads(torch.cat((video_v, own_v.unsqueeze(2)), 2))
        spatial = ((q * self.head_dim ** -.5) @ k.transpose(-1, -2)).softmax(-1) @ v
        token = token + self.spatial_out(spatial.transpose(-3, -2).reshape(b, t, d))

        q, k, v = self.temporal_qkv(self.temporal_norm(token)).chunk(3, -1)
        q, k, v = self._heads(q), self._heads(k), self._heads(v)
        temporal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        token = token + self.temporal_out(temporal.transpose(1, 2).reshape(b, t, d))

        projected = self.text_projection(text)
        text_q = self.text_norm(token).reshape(b * t, 1, d)
        if projected.ndim == 3:
            projected = projected[:, None].expand(-1, t, -1, -1)
            text_mask = text_mask[:, None].expand(-1, t, -1)
        if projected.ndim != 4 or projected.shape[:2] != (b, t):
            raise ValueError("current text must be [B,L,D] or [B,8,L,D]")
        text_kv = projected.flatten(0, 1)
        padding = ~text_mask.flatten(0, 1).bool()
        all_padding = padding.all(-1)
        if all_padding.any():
            padding, text_kv = padding.clone(), text_kv.clone()
            padding[all_padding, 0] = False; text_kv[all_padding, 0] = 0
        attended, _ = self.text_attention(text_q, text_kv, text_kv,
                                           key_padding_mask=padding, need_weights=False)
        token = token + self.current_text_gate * attended.reshape(b, t, d)
        token = token + self.mlp(self.mlp_norm(token))
        return token * valid[..., None].to(token.dtype)


class PolicyFamilyBranch(nn.Module):
    def __init__(self, hidden, heads, observable_dim, cfg):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(hidden))
        self.temporal = nn.Parameter(torch.randn(cfg.history_frames, hidden) * .02)
        self.observable = nn.Linear(observable_dim, hidden)
        self.action = nn.Linear(23, hidden)
        self.profile = nn.Embedding(len(PROFILES), hidden)
        self.shared_text = nn.Linear(cfg.text_hidden_size, hidden)
        self.shared_gate = nn.Parameter(torch.tensor(.2))
        self.blocks = nn.ModuleList([
            ReadOnlyPolicyBlock(hidden, heads, cfg.text_hidden_size, cfg.mlp_ratio)
            for _ in range(cfg.num_policy_blocks)
        ])
        self.head = StructuredActionHead(hidden, cfg.horizons)

    def initialize(self, observable, incoming, profile, shared_text, shared_mask):
        weight = shared_mask.to(shared_text.dtype)[..., None]
        pooled = (shared_text * weight).sum(1) / weight.sum(1).clamp_min(1)
        return (self.base + self.temporal[None] + self.observable(observable)
                + self.action(incoming) + self.profile(profile)[:, None]
                + self.shared_gate * self.shared_text(pooled)[:, None])


class InsertedInhabitantPolicy(nn.Module):
    """Frozen M3 plus family-routed, spatially inserted 8-frame M4 branches."""

    def __init__(self, renderer, cfg=InsertedPolicyArgs()):
        super().__init__()
        if cfg.history_frames != 8 or cfg.horizons != 8:
            raise ValueError("PLOT M4 is fixed to eight completed frames -> eight actions")
        if not 1 <= cfg.num_policy_blocks <= renderer.core.depth:
            raise ValueError("num_policy_blocks must fit inside M3 depth")
        self.renderer, self.cfg = renderer, cfg
        for parameter in renderer.parameters(): parameter.requires_grad_(False)
        renderer.core.gradient_checkpointing = False
        hidden, heads = renderer.core.hidden_size, renderer.core.num_heads
        observable_dim = renderer.resident_encoder.output_dim
        self.families = nn.ModuleDict({
            name: PolicyFamilyBranch(hidden, heads, observable_dim, cfg) for name in FAMILIES
        })

    def train(self, mode=True):
        super().train(mode); self.renderer.eval(); return self

    @staticmethod
    def _gather_time(value, indices):
        shape = (*indices.shape, *value.shape[2:])
        gather = indices.view(*indices.shape, *([1] * (value.ndim - 2))).expand(shape)
        return value.gather(1, gather)

    def forward(self, latents, conditions, family_id, profile_id, *,
                shared_text, shared_text_mask, current_text, current_text_mask,
                policy_indices=None, video_layers=None):
        b, t = conditions["player_position"].shape[:2]
        if policy_indices is None:
            if t != self.cfg.history_frames:
                raise ValueError("provide policy_indices when M3 context is longer than eight frames")
            policy_indices = torch.arange(t, device=family_id.device)[None].expand(b, -1)
        policy_indices = policy_indices.long()
        if policy_indices.shape != (b, self.cfg.history_frames):
            raise ValueError("policy_indices must select exactly eight completed frames per sample")
        if (policy_indices < 0).any() or (policy_indices >= t).any():
            raise ValueError("policy_indices fall outside the M3 context")
        if not (policy_indices[:, 1:] == policy_indices[:, :-1] + 1).all():
            raise ValueError("M4 policy frames must be consecutive")
        for family, profiles in FAMILY_PROFILES.items():
            selected = family_id == family
            if selected.any() and not torch.tensor(
                    sorted(profiles), device=profile_id.device).eq(
                    profile_id[selected, None]).any(-1).all():
                raise ValueError("profile is incompatible with its parameter family")
        # Empty text still has an EOS token in T5. Mask non-language families
        # explicitly so text cannot become an accidental behavior condition.
        language = family_id == FAMILY_TO_ID["language_builder"]
        shared_text_mask = shared_text_mask.bool() & language[:, None]
        language_text = language.view(b, *([1] * (current_text_mask.ndim - 1)))
        current_text_mask = current_text_mask.bool() & language_text
        target = conditions["target_agent"].long()
        with torch.no_grad():
            observable, _ = self.renderer.resident_encoder(conditions)
            observable = self._gather_time(
                self.renderer.select(observable, target), policy_indices)
            encoded = (self.renderer.encode_conditions(conditions)
                       if video_layers is None else None)
        incoming = self._gather_time(
            self.renderer.select(conditions["action"], target), policy_indices)
        valid = self._gather_time(
            self.renderer.select(conditions["player_valid"], target), policy_indices).bool()
        states, selections = {}, {}
        for name, index in FAMILY_TO_ID.items():
            selected = (family_id == index).nonzero().flatten()
            selections[name] = selected
            if selected.numel():
                branch = self.families[name]
                states[name] = branch.initialize(
                    observable[selected], incoming[selected], profile_id[selected],
                    shared_text[selected], shared_text_mask[selected])
        first = self.renderer.core.depth - self.cfg.num_policy_blocks

        def consume(block_index, video_hidden):
            if block_index < first: return
            local = block_index - first
            selected_video = self._gather_time(video_hidden, policy_indices)
            for name, selected in selections.items():
                if selected.numel():
                    branch = self.families[name]
                    states[name] = branch.blocks[local](
                        states[name], selected_video[selected], current_text[selected],
                        current_text_mask[selected], valid[selected])

        # The frozen video forward creates no parameter gradients. Policy blocks
        # receive detached patches through the callback and remain trainable.
        if video_layers is None:
            if latents is None or latents.shape[:2] != (b, t):
                raise ValueError("latents must cover the complete M3 context")
            video = self.renderer.core(
                latents, torch.zeros((b, t), device=latents.device), encoded,
                cache_write=False, block_callback=consume)
        else:
            if len(video_layers) != self.cfg.num_policy_blocks:
                raise ValueError("video_layers must contain the final M3 policy insertion layers")
            video = None
            for local, hidden in enumerate(video_layers):
                # Cached rollout already returns only the latest completed block.
                if hidden.shape[1] == self.cfg.history_frames and t == self.cfg.history_frames:
                    selected_hidden = hidden
                else:
                    selected_hidden = self._gather_time(hidden, policy_indices)
                for name, selected in selections.items():
                    if selected.numel():
                        branch = self.families[name]
                        states[name] = branch.blocks[local](
                            states[name], selected_hidden[selected], current_text[selected],
                            current_text_mask[selected], valid[selected])
        family_logits = {
            name: self.families[name].head(states[name][:, -1])
            for name, selected in selections.items() if selected.numel()
        }
        template = next(iter(family_logits.values()))
        outputs = {key: value.new_zeros((b, *value.shape[1:]))
                   for key, value in template.items()}
        for name, selected in selections.items():
            if selected.numel():
                logits = family_logits[name]
                for key in outputs: outputs[key][selected] = logits[key]
        peaceful = profile_id == PROFILE_TO_ID["villager_peaceful"]
        return video, constrain_peaceful_logits(outputs, peaceful)

    def decode(self, logits):
        # Every family uses the same output schema/bins.
        return next(iter(self.families.values())).head.decode(logits)

    def loss(self, logits, actions, valid_mask=None, horizon_weights=None,
             sample_weight=None):
        return next(iter(self.families.values())).head.loss(
            logits, actions, valid_mask, horizon_weights, sample_weight)
