"""Regularized zombie M4 with split attack and gated soft mouse heads."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .state_policy_large import M3StateGeometry
from .structured_action import StructuredActionHead


@dataclass(frozen=True)
class ZombieStatePolicyV2Args:
    num_block_classes: int
    item_vocab_size: int
    profile: str = "zombie_melee"
    hidden: int = 640
    heads: int = 10
    depth: int = 8
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    image_h: int = 36
    image_w: int = 64
    voxel_channels: int = 32


@dataclass(frozen=True)
class ZombieStatePolicyV3Args(ZombieStatePolicyV2Args):
    """Zombie policy with an explicit nearest-player combat representation."""

    attack_range: float = 3.25


class V2Attention(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden size must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.dropout = dropout
        self.q = nn.Linear(hidden, hidden)
        self.kv = nn.Linear(hidden, hidden * 2)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, query, memory, valid):
        def split(value):
            return value.reshape(value.shape[0], value.shape[1], self.heads, self.head_dim).transpose(1, 2)

        q = split(self.q(query))
        k, v = [split(value) for value in self.kv(memory).chunk(2, -1)]
        q = F.rms_norm(q, (self.head_dim,))
        k = F.rms_norm(k, (self.head_dim,))
        value = F.scaled_dot_product_attention(
            q, k, v, attn_mask=valid[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0)
        return self.out(value.transpose(1, 2).flatten(2))


class V2PolicyBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden
        self.dropout = cfg.dropout
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.actor_norm = nn.LayerNorm(hidden)
        self.actor_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(4 * hidden, hidden))

    def forward(self, tokens, valid, actors, actor_valid):
        value = self.self_norm(tokens)
        tokens = tokens + F.dropout(
            self.self_attn(value, value, valid), self.dropout, self.training)
        tokens = tokens + F.dropout(
            self.actor_attn(self.actor_norm(tokens), actors, actor_valid),
            self.dropout, self.training)
        return tokens + F.dropout(self.ff(self.ff_norm(tokens)), self.dropout, self.training)


class ZombieStatePolicyV2(nn.Module):
    """Eight-frame policy with an isolated attack head and mouse move gates."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.profile not in ("zombie_melee", "skeleton_swordsman"):
            raise ValueError("v2 currently supports combat profiles only")
        self.cfg = cfg
        hidden = cfg.hidden
        self.geometry = M3StateGeometry(cfg)
        self.item = nn.Embedding(cfg.item_vocab_size, 64)
        self.kind = nn.Embedding(4, 16)
        self.resident = nn.Sequential(
            nn.Linear(18 + 64 + 16 + 23, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden))
        self.history_position = nn.Parameter(torch.randn(1, 8, 1, hidden) * 0.02)
        self.self_role = nn.Parameter(torch.randn(hidden) * 0.02)
        self.actor_null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.actor_memory_norm = nn.LayerNorm(hidden)
        self.query = nn.Parameter(torch.randn(1, 8, hidden) * 0.02)
        self.blocks = nn.ModuleList([V2PolicyBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(hidden)
        self.output_dropout = nn.Dropout(cfg.dropout)

        # The common head predicts movement keys, hotbar and conditional mouse direction.
        # Structured key slot 7 is replaced by the dedicated attack head below.
        self.common_head = StructuredActionHead(hidden, 1)
        branch_hidden = max(128, hidden // 2)
        self.attack_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, branch_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(branch_hidden, 1))
        self.mouse_move_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, branch_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(branch_hidden, 2))

        # Buffer-only target/decode helper.
        self.loss_head = StructuredActionHead(1, 8)
        for name in ("keys", "hotbar", "mouse_x", "mouse_y"):
            delattr(self.loss_head, name)
        self.register_buffer("move_key_indices", torch.tensor([0, 1, 2, 3, 4, 5, 6, 8, 9]))

    def encode(self, inputs):
        """Encode one committed state into the eight action-query features.

        Keeping this boundary explicit lets an outcome critic reuse exactly the
        actor representation without re-running the M3-compatible geometry
        encoder.  No future state or action is consumed here.
        """
        batch, history, actors_count, _ = inputs["resident_state"].shape
        if history != 8 or not inputs["history_valid"][:, -1].all():
            raise ValueError("invalid history")
        target = inputs["target_agent"].long()
        device = target.device
        valid = inputs["resident_valid"].bool() & inputs["history_valid"].bool()[:, :, None]
        if not valid[torch.arange(batch, device=device), -1, target].all():
            raise ValueError("missing target")

        residents = self.resident(torch.cat((
            inputs["resident_state"], self.item(inputs["held_item"].long()),
            self.kind(inputs["resident_type"].long()), inputs["resident_actions"]), -1))
        residents = residents + self.history_position
        own = residents[torch.arange(batch, device=device), :, target] + self.self_role
        others = valid & ~F.one_hot(target, actors_count).bool()[:, None, :]
        actors = self.actor_memory_norm(torch.cat((
            self.actor_null.expand(batch, -1, -1), residents.flatten(1, 2)), 1))
        actor_valid = torch.cat((
            torch.ones(batch, 1, device=device, dtype=torch.bool), others.flatten(1, 2)), 1)

        geometry = self.geometry(inputs)
        queries = self.query.expand(batch, -1, -1) + own[:, -1:]
        tokens = torch.cat((geometry, own, queries), 1)
        token_valid = torch.cat((
            torch.ones(batch, geometry.shape[1], device=device, dtype=torch.bool),
            inputs["history_valid"].bool(),
            torch.ones(batch, 8, device=device, dtype=torch.bool)), 1)
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, token_valid, actors, actor_valid,
                                    use_reentrant=False)
            else:
                tokens = block(tokens, token_valid, actors, actor_valid)

        return self.output_dropout(self.norm(tokens[:, -8:]))

    def action_logits(self, hidden):
        logits = {name: value.squeeze(-2)
                  for name, value in self.common_head(hidden).items()}
        attack = self.attack_head(hidden).squeeze(-1)
        mouse_move = self.mouse_move_head(hidden)
        logits["keys"] = torch.cat((
            logits["keys"][..., :7], attack[..., None], logits["keys"][..., 8:]), -1)
        logits["attack"] = attack
        logits["mouse_x_move"] = mouse_move[..., 0]
        logits["mouse_y_move"] = mouse_move[..., 1]
        return logits

    def forward(self, inputs):
        return self.action_logits(self.encode(inputs))

    @staticmethod
    def _soft_bin_loss(logits, target, smoothing):
        classes = logits.shape[-1]
        distribution = F.one_hot(target, classes).to(logits.dtype) * (1.0 - smoothing)
        left = (target - 1).clamp_min(0)
        right = (target + 1).clamp_max(classes - 1)
        distribution.scatter_add_(-1, left[..., None],
                                  torch.full_like(left[..., None], smoothing / 2, dtype=logits.dtype))
        distribution.scatter_add_(-1, right[..., None],
                                  torch.full_like(right[..., None], smoothing / 2, dtype=logits.dtype))
        distribution = distribution / distribution.sum(-1, keepdim=True)
        return -(distribution * F.log_softmax(logits, -1)).sum(-1)

    def loss(self, logits, actions, valid_mask=None, positive_weight=1.0,
             move_key_weight=1.0, attack_weight=1.0, mouse_smoothing=0.2,
             mouse_move_positive_weight=1.0, horizon_weights=None):
        targets = self.loss_head.targets(actions)
        key_losses = F.binary_cross_entropy_with_logits(
            logits["keys"], targets["keys"], reduction="none")
        move_keys = key_losses.index_select(-1, self.move_key_indices).mean(-1)
        attack = F.binary_cross_entropy_with_logits(
            logits["attack"], targets["keys"][..., 7], reduction="none",
            pos_weight=logits["attack"].new_tensor(positive_weight))
        hotbar = F.cross_entropy(
            logits["hotbar"].flatten(0, -2), targets["hotbar"].flatten(),
            reduction="none").view_as(attack)

        mouse_parts = {}
        zero_indices = {
            "mouse_x": int((self.loss_head.mouse_bins_x == 0).nonzero()[0]),
            "mouse_y": int((self.loss_head.mouse_bins_y == 0).nonzero()[0]),
        }
        for name in ("mouse_x", "mouse_y"):
            move_target = (targets[name] != zero_indices[name]).to(logits[name].dtype)
            gate = F.binary_cross_entropy_with_logits(
                logits[name + "_move"], move_target, reduction="none",
                pos_weight=logits[name].new_tensor(mouse_move_positive_weight))
            direction = self._soft_bin_loss(logits[name], targets[name], mouse_smoothing)
            mouse_parts[name + "_gate"] = gate
            mouse_parts[name] = gate + move_target * direction

        combined = (move_key_weight * move_keys + attack_weight * attack + hotbar
                    + mouse_parts["mouse_x"] + mouse_parts["mouse_y"])
        weights = (torch.ones_like(combined) if valid_mask is None
                   else valid_mask.to(combined))
        if horizon_weights is not None:
            weights = weights * horizon_weights.to(combined)[None]
        per_sample = (combined * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        loss = per_sample.mean()
        parts = dict(move_keys=move_keys, attack=attack, hotbar=hotbar,
                     mouse_x=mouse_parts["mouse_x"], mouse_y=mouse_parts["mouse_y"],
                     mouse_x_gate=mouse_parts["mouse_x_gate"],
                     mouse_y_gate=mouse_parts["mouse_y_gate"])
        return loss, parts

    @torch.no_grad()
    def decode(self, logits, key_threshold=0.5, mouse_move_threshold=0.5):
        output = self.loss_head.decode(logits, key_threshold)
        for axis, action_index, bins in (
                ("mouse_x", 21, self.loss_head.mouse_bins_x),
                ("mouse_y", 22, self.loss_head.mouse_bins_y)):
            direction_logits = logits[axis].clone()
            zero = int((bins == 0).nonzero()[0])
            direction_logits[..., zero] = -torch.inf
            direction = bins[direction_logits.argmax(-1)]
            moving = logits[axis + "_move"].sigmoid() >= mouse_move_threshold
            output[..., action_index] = torch.where(moving, direction, direction.new_zeros(()))
        return output


class ZombieStatePolicyV3(ZombieStatePolicyV2):
    """Behavior-cloned zombie with explicit pursuit and melee-range supervision.

    The policy remains fully offline: combat targets are derived from the current
    resident state, never from future observations, rewards, or engine state.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        hidden = cfg.hidden
        # relative position, unit direction, distance, relative velocity,
        # self yaw/pitch sin/cos, both HP values, and target-present bit.
        combat_features = 17
        self.combat_encoder = nn.Sequential(
            nn.Linear(combat_features, hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(hidden, hidden))
        self.combat_fusion = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
        branch_hidden = max(128, hidden // 2)
        self.range_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, branch_hidden), nn.GELU(),
            nn.Linear(branch_hidden, 1))
        self.bearing_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, branch_hidden), nn.GELU(),
            nn.Linear(branch_hidden, 3))

    @staticmethod
    def combat_targets(inputs):
        state = inputs["resident_state"]
        valid = inputs["resident_valid"].bool()
        kinds = inputs["resident_type"].long()
        target = inputs["target_agent"].long()
        batch, _, actors, _ = state.shape
        rows = torch.arange(batch, device=state.device)
        own = state[rows, -1, target]
        own_position = own[:, :3]
        position = state[:, -1, :, :3]
        relative = position - own_position[:, None]
        distance_sq = relative.square().sum(-1)
        candidates = valid[:, -1] & kinds[:, -1].eq(0)
        candidates &= ~F.one_hot(target, actors).bool()
        selected = distance_sq.masked_fill(~candidates, torch.inf).argmin(-1)
        present = candidates.any(-1)
        selected_state = state[rows, -1, selected]
        selected_position = selected_state[:, :3]
        relative = selected_position - own_position
        distance_normalized = relative.norm(dim=-1)
        unit = relative / distance_normalized[:, None].clamp_min(1e-6)
        previous_relative = (state[rows, -2, selected, :3]
                             - state[rows, -2, target, :3])
        velocity = relative - previous_relative
        zeros = torch.zeros_like(relative)
        relative = torch.where(present[:, None], relative, zeros)
        unit = torch.where(present[:, None], unit, zeros)
        velocity = torch.where(present[:, None], velocity, zeros)
        distance_blocks = torch.where(
            present, distance_normalized * 24, distance_normalized.new_zeros(()))
        features = torch.cat((
            relative, unit, distance_normalized[:, None], velocity,
            own[:, 3:7], own[:, 7:8], selected_state[:, 7:8],
            present[:, None].to(state.dtype)), -1)
        return dict(features=features, valid=present, unit=unit,
                    distance=distance_blocks)

    def forward(self, inputs):
        combat = self.combat_targets(inputs)
        hidden = super().encode(inputs)
        context = self.combat_encoder(combat["features"])
        hidden = hidden + self.combat_fusion(hidden + context[:, None])
        logits = self.action_logits(hidden)
        logits.update(
            combat_range=self.range_head(hidden).squeeze(-1),
            combat_bearing=self.bearing_head(hidden),
            combat_valid=combat["valid"],
            combat_unit=combat["unit"],
            combat_distance=combat["distance"])
        return logits

    @staticmethod
    def _masked_horizon_mean(values, valid_mask, horizon_weights, sample_valid):
        weights = valid_mask.to(values)
        if horizon_weights is not None:
            weights = weights * horizon_weights.to(values)[None]
        weights = weights * sample_valid.to(values)[:, None]
        return (values * weights).sum() / weights.sum().clamp_min(1)

    def loss(self, logits, actions, valid_mask=None, positive_weight=1.0,
             move_key_weight=1.0, attack_weight=1.0, mouse_smoothing=0.2,
             mouse_move_positive_weight=1.0, horizon_weights=None,
             locomotion_weight=1.0, combat_range_weight=0.5,
             combat_bearing_weight=0.25):
        base, parts = super().loss(
            logits, actions, valid_mask, positive_weight, move_key_weight,
            attack_weight, mouse_smoothing, mouse_move_positive_weight,
            horizon_weights)
        targets = self.loss_head.targets(actions)
        valid = (torch.ones(actions.shape[:-1], dtype=torch.bool, device=actions.device)
                 if valid_mask is None else valid_mask.bool())
        combat_valid = logits["combat_valid"].bool()
        locomotion = F.binary_cross_entropy_with_logits(
            logits["keys"][..., :4], targets["keys"][..., :4], reduction="none").mean(-1)
        in_range = (logits["combat_distance"] <= self.cfg.attack_range).to(logits["combat_range"])
        range_loss = F.binary_cross_entropy_with_logits(
            logits["combat_range"], in_range[:, None].expand_as(logits["combat_range"]),
            reduction="none")
        bearing = F.smooth_l1_loss(
            logits["combat_bearing"],
            logits["combat_unit"][:, None].expand_as(logits["combat_bearing"]),
            reduction="none").mean(-1)
        locomotion_mean = self._masked_horizon_mean(
            locomotion, valid, horizon_weights, combat_valid)
        range_mean = self._masked_horizon_mean(
            range_loss, valid, horizon_weights, combat_valid)
        bearing_mean = self._masked_horizon_mean(
            bearing, valid, horizon_weights, combat_valid)
        loss = (base + locomotion_weight * locomotion_mean
                + combat_range_weight * range_mean
                + combat_bearing_weight * bearing_mean)
        parts.update(locomotion=locomotion, combat_range=range_loss,
                     combat_bearing=bearing)
        return loss, parts

    @torch.no_grad()
    def decode(self, logits, key_threshold=0.5, mouse_move_threshold=0.5,
               attack_threshold=0.3):
        output = super().decode(logits, key_threshold, mouse_move_threshold)
        output[..., 8] = (logits["attack"].sigmoid() >= attack_threshold).to(output)
        return output
