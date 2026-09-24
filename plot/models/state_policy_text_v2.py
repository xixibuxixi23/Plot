"""Regularized text-conditioned M4 aligned with the combat v2 backbone."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .state_policy_large import M3StateGeometry
from .state_policy_large_v2 import V2Attention
from .structured_action import StructuredActionHead


@dataclass(frozen=True)
class TextStatePolicyV2Args:
    num_block_classes: int
    item_vocab_size: int
    profile: str = "language_builder"
    hidden: int = 640
    heads: int = 10
    depth: int = 8
    text_hidden_size: int = 768
    dropout: float = 0.1
    gradient_checkpointing: bool = True
    image_h: int = 36
    image_w: int = 64
    voxel_channels: int = 32
    target_pointer: bool = False


@dataclass(frozen=True)
class UnifiedStatePolicyV4Args(TextStatePolicyV2Args):
    profile: str = "unified"
    num_policy_roles: int = 4


class TextV2PolicyBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden
        self.dropout = cfg.dropout
        self.self_norm = nn.LayerNorm(hidden)
        self.self_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.ego_norm = nn.LayerNorm(hidden)
        self.ego_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.actor_norm = nn.LayerNorm(hidden)
        self.actor_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.target_norm = nn.LayerNorm(hidden)
        self.target_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.text_norm = nn.LayerNorm(hidden)
        self.text_attn = V2Attention(hidden, cfg.heads, cfg.dropout)
        self.ff_norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(4 * hidden, hidden))

    def forward(self, tokens, valid, ego, ego_valid, actors, actor_valid,
                target_memory, target_valid, text, text_valid):
        value = self.self_norm(tokens)
        tokens = tokens + F.dropout(
            self.self_attn(value, value, valid), self.dropout, self.training)
        tokens = tokens + F.dropout(
            self.ego_attn(self.ego_norm(tokens), ego, ego_valid),
            self.dropout, self.training)
        tokens = tokens + F.dropout(
            self.actor_attn(self.actor_norm(tokens), actors, actor_valid),
            self.dropout, self.training)
        tokens = tokens + F.dropout(
            self.target_attn(self.target_norm(tokens), target_memory, target_valid),
            self.dropout, self.training)
        tokens = tokens + F.dropout(
            self.text_attn(self.text_norm(tokens), text, text_valid),
            self.dropout, self.training)
        return tokens + F.dropout(self.ff(self.ff_norm(tokens)), self.dropout, self.training)


class TextStatePolicyV2(nn.Module):
    """Eight-frame state policy with optional text and split action heads."""

    def __init__(self, cfg):
        super().__init__()
        if cfg.profile not in ("language_builder", "zombie_melee"):
            raise ValueError("state policy requires a supported independent profile")
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
        self.ego_memory_norm = nn.LayerNorm(hidden)
        self.query = nn.Parameter(torch.randn(1, 8, hidden) * 0.02)

        # Optional task-target bottleneck.  It selects one exact voxel from the
        # 13^3 neighborhood, one resident, or null.  The selected soft feature
        # is fused into all action queries, so inference remains end-to-end.
        if cfg.target_pointer:
            self.target_voxel = nn.Embedding(cfg.num_block_classes + 1, hidden)
            self.target_coordinate = nn.Sequential(
                nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.target_entity_coordinate = nn.Sequential(
                nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.target_entity_query = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            self.target_entity_key = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            self.target_entity_value = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            self.target_query = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            self.target_key = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            self.target_value = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden))
            self.target_null = nn.Parameter(torch.randn(1, 1, hidden) * 0.02)
            self.target_memory_role = nn.Parameter(torch.randn(1, 2, hidden) * 0.02)
            self.target_fusion = nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.Tanh())
            nn.init.zeros_(self.target_fusion[1].weight)
            nn.init.zeros_(self.target_fusion[1].bias)
            axis = torch.arange(-6, 7, dtype=torch.float32)
            coordinates = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
            self.register_buffer("target_local_coordinates", coordinates.reshape(1, 2197, 3))

        self.text_projection = nn.Linear(cfg.text_hidden_size, hidden)
        self.text_role = nn.Parameter(torch.randn(2, 1, hidden) * 0.02)
        self.text_null = nn.Parameter(torch.zeros(1, 1, hidden))
        self.text_memory_norm = nn.LayerNorm(hidden)

        self.blocks = nn.ModuleList([TextV2PolicyBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(hidden)
        self.output_dropout = nn.Dropout(cfg.dropout)
        self.common_head = StructuredActionHead(hidden, 1)
        branch_hidden = max(128, hidden // 2)
        def binary_head():
            return nn.Sequential(
                nn.LayerNorm(hidden), nn.Linear(hidden, branch_hidden), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(branch_hidden, 1))
        self.attack_head = binary_head()
        self.place_head = binary_head()
        self.mouse_move_head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, branch_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(branch_hidden, 2))

        self.loss_head = StructuredActionHead(1, 8)
        for name in ("keys", "hotbar", "mouse_x", "mouse_y"):
            delattr(self.loss_head, name)
        self.register_buffer("move_key_indices", torch.tensor([0, 1, 2, 3, 4, 5, 6, 9]))

    def encode(self, inputs, return_aux=False):
        """Encode committed state/text into the shared eight query features."""
        inputs = dict(inputs)
        batch, history, actors_count, _ = inputs["resident_state"].shape
        if history != 8 or not inputs["history_valid"][:, -1].all():
            raise ValueError("invalid history")
        target = inputs["target_agent"].long()
        device = target.device
        for name in ("shared", "current"):
            text_key, mask_key = name + "_text", name + "_text_mask"
            if text_key not in inputs:
                inputs[text_key] = torch.zeros(
                    batch, 1, self.cfg.text_hidden_size, device=device)
                inputs[mask_key] = torch.zeros(
                    batch, 1, dtype=torch.bool, device=device)
        valid = inputs["resident_valid"].bool() & inputs["history_valid"].bool()[:, :, None]
        if not valid[torch.arange(batch, device=device), -1, target].all():
            raise ValueError("missing target")
        residents = self.resident(torch.cat((
            inputs["resident_state"], self.item(inputs["held_item"].long()),
            self.kind(inputs["resident_type"].long()), inputs["resident_actions"]), -1))
        residents = residents + self.history_position
        own = residents[torch.arange(batch, device=device), :, target] + self.self_role
        ego = self.ego_memory_norm(own)
        ego_valid = inputs["history_valid"].bool()
        others = valid & ~F.one_hot(target, actors_count).bool()[:, None, :]
        actors = self.actor_memory_norm(torch.cat((
            self.actor_null.expand(batch, -1, -1), residents.flatten(1, 2)), 1))
        actor_valid = torch.cat((
            torch.ones(batch, 1, device=device, dtype=torch.bool), others.flatten(1, 2)), 1)

        text_parts = [self.text_null.expand(batch, -1, -1)]
        text_masks = [torch.ones(batch, 1, device=device, dtype=torch.bool)]
        for role, name in enumerate(("shared", "current")):
            text_parts.append(self.text_projection(inputs[name + "_text"]) + self.text_role[role])
            text_masks.append(inputs[name + "_text_mask"].bool())
        text = self.text_memory_norm(torch.cat(text_parts, 1))
        text_valid = torch.cat(text_masks, 1)

        geometry = self.geometry(inputs)
        queries = self.query.expand(batch, -1, -1) + own[:, -1:]
        target_logits = None
        entity_logits = None
        target_memory = self.actor_null.expand(batch, -1, -1)
        target_valid = torch.ones(batch, 1, dtype=torch.bool, device=device)
        if self.cfg.target_pointer:
            classes = inputs["voxel_classes"][:, 18:31, 18:31, 18:31]
            known = inputs["voxel_known"][:, 18:31, 18:31, 18:31].bool()
            ids = torch.where(known, classes.long(), self.cfg.num_block_classes)
            relative_blocks = (self.target_local_coordinates.to(queries)
                               + inputs["grid_offset"][:, None].to(queries) + 0.5)
            current_residents = residents[:, -1]
            entity_candidates = torch.cat((
                current_residents, self.target_null.expand(batch, -1, -1)), 1)
            entity_valid = torch.cat((
                valid[:, -1], torch.ones(batch, 1, dtype=torch.bool, device=device)), 1)
            text_weight = text_valid.to(text.dtype)
            text_summary = (text * text_weight[..., None]).sum(1) / text_weight.sum(
                1, keepdim=True).clamp_min(1)
            context = own[:, -1] + geometry.mean(1) + text_summary
            entity_logits = torch.einsum(
                "bd,bnd->bn", self.target_entity_query(context),
                self.target_entity_key(entity_candidates)) / self.cfg.hidden ** 0.5
            entity_logits = entity_logits.masked_fill(~entity_valid, -torch.inf)
            entity_probability = entity_logits.float().softmax(-1).to(entity_candidates.dtype)
            entity_feature = torch.einsum(
                "bn,bnd->bd", entity_probability,
                self.target_entity_value(entity_candidates))
            entity_positions = torch.cat((
                inputs["resident_state"][:, -1, :, :3].to(queries) * 24.0,
                torch.zeros(batch, 1, 3, device=device, dtype=queries.dtype)), 1)
            entity_position = torch.einsum(
                "bn,bnd->bd", entity_probability, entity_positions)
            relative_self = relative_blocks / 6.0
            relative_entity = (relative_blocks - entity_position[:, None]) / 6.0
            voxels = (self.target_voxel(ids.flatten(1))
                      + self.target_coordinate(relative_self)
                      + self.target_entity_coordinate(relative_entity))
            candidates = torch.cat((
                voxels, current_residents, self.target_null.expand(batch, -1, -1)), 1)
            candidate_valid = torch.cat((
                known.flatten(1), valid[:, -1],
                torch.ones(batch, 1, dtype=torch.bool, device=device)), 1)
            target_logits = torch.einsum(
                "bd,bnd->bn", self.target_query(context + entity_feature),
                self.target_key(candidates)
            ) / self.cfg.hidden ** 0.5
            target_logits = target_logits.masked_fill(~candidate_valid, -torch.inf)
            target_probability = target_logits.float().softmax(-1).to(candidates.dtype)
            target_feature = torch.einsum(
                "bn,bnd->bd", target_probability, self.target_value(candidates))
            queries = queries + self.target_fusion(target_feature)[:, None]
            target_memory = (torch.stack((entity_feature, target_feature), 1)
                             + self.target_memory_role)
            target_valid = torch.ones(batch, 2, dtype=torch.bool, device=device)
        tokens = torch.cat((geometry, own, queries), 1)
        token_valid = torch.cat((
            torch.ones(batch, geometry.shape[1], device=device, dtype=torch.bool),
            inputs["history_valid"].bool(),
            torch.ones(batch, 8, device=device, dtype=torch.bool)), 1)
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                tokens = checkpoint(block, tokens, token_valid, ego, ego_valid,
                                    actors, actor_valid, target_memory, target_valid,
                                    text, text_valid, use_reentrant=False)
            else:
                tokens = block(tokens, token_valid, ego, ego_valid, actors, actor_valid,
                               target_memory, target_valid, text, text_valid)

        hidden = self.output_dropout(self.norm(tokens[:, -8:]))
        return (hidden, target_logits, entity_logits) if return_aux else hidden

    def action_logits(self, hidden):
        logits = {name: value.squeeze(-2)
                  for name, value in self.common_head(hidden).items()}
        attack = self.attack_head(hidden).squeeze(-1)
        place = self.place_head(hidden).squeeze(-1)
        mouse_move = self.mouse_move_head(hidden)
        logits["keys"] = torch.cat((
            logits["keys"][..., :7], attack[..., None], place[..., None],
            logits["keys"][..., 9:]), -1)
        logits.update(attack=attack, place=place,
                      mouse_x_move=mouse_move[..., 0], mouse_y_move=mouse_move[..., 1])
        return logits

    def forward(self, inputs):
        hidden, target_logits, entity_logits = self.encode(inputs, return_aux=True)
        logits = self.action_logits(hidden)
        if target_logits is not None:
            logits["target"] = target_logits
            logits["target_entity"] = entity_logits
        return logits

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
        return -(distribution / distribution.sum(-1, keepdim=True)
                 * F.log_softmax(logits, -1)).sum(-1)

    def loss(self, logits, actions, valid_mask=None, positive_weight=4.0,
             place_positive_weight=4.0, move_key_weight=1.0, attack_weight=2.0,
             place_weight=2.0, mouse_smoothing=0.2,
             mouse_move_positive_weight=2.0, horizon_weights=None,
             target_address=None, target_weight=0.5):
        targets = self.loss_head.targets(actions)
        key_losses = F.binary_cross_entropy_with_logits(
            logits["keys"], targets["keys"], reduction="none")
        move_keys = key_losses.index_select(-1, self.move_key_indices).mean(-1)
        attack = F.binary_cross_entropy_with_logits(
            logits["attack"], targets["keys"][..., 7], reduction="none",
            pos_weight=logits["attack"].new_tensor(positive_weight))
        place = F.binary_cross_entropy_with_logits(
            logits["place"], targets["keys"][..., 8], reduction="none",
            pos_weight=logits["place"].new_tensor(place_positive_weight))
        hotbar = F.cross_entropy(
            logits["hotbar"].flatten(0, -2), targets["hotbar"].flatten(),
            reduction="none").view_as(attack)
        mouse_parts = {}
        for name, bins in (("mouse_x", self.loss_head.mouse_bins_x),
                           ("mouse_y", self.loss_head.mouse_bins_y)):
            zero = int((bins == 0).nonzero()[0])
            move_target = (targets[name] != zero).to(logits[name].dtype)
            gate = F.binary_cross_entropy_with_logits(
                logits[name + "_move"], move_target, reduction="none",
                pos_weight=logits[name].new_tensor(mouse_move_positive_weight))
            direction = self._soft_bin_loss(logits[name], targets[name], mouse_smoothing)
            mouse_parts[name + "_gate"] = gate
            mouse_parts[name] = gate + move_target * direction
        combined = (move_key_weight * move_keys + attack_weight * attack
                    + place_weight * place + hotbar
                    + mouse_parts["mouse_x"] + mouse_parts["mouse_y"])
        weights = torch.ones_like(combined) if valid_mask is None else valid_mask.to(combined)
        if horizon_weights is not None:
            weights = weights * horizon_weights.to(combined)[None]
        per_sample = (combined * weights).sum(-1) / weights.sum(-1).clamp_min(1)
        loss = per_sample.mean()
        parts = dict(move_keys=move_keys, attack=attack, place=place, hotbar=hotbar,
                     mouse_x=mouse_parts["mouse_x"], mouse_y=mouse_parts["mouse_y"],
                     mouse_x_gate=mouse_parts["mouse_x_gate"],
                     mouse_y_gate=mouse_parts["mouse_y_gate"])
        if target_address is not None:
            if "target" not in logits:
                raise ValueError("target labels require target_pointer=True")
            labels = target_address.long()
            # Sidecars use a fixed 2197 sentinel for null.  Resident candidates
            # are inserted before the model's final null candidate.
            labels = torch.where(labels == 2197, labels.new_full((), logits["target"].shape[-1] - 1), labels)
            target_loss = F.cross_entropy(logits["target"].float(), labels, ignore_index=-1)
            loss = loss + target_weight * target_loss
            parts["target"] = target_loss
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


class IndependentStatePolicyV4(TextStatePolicyV2):
    """Shared architecture template instantiated separately for each task.

    Checkpoints, optimizers and gradients are never shared between profiles.
    Zombie policies use masked text while retaining exactly the same modules and
    parameter shapes as language-builder policies.
    """


class UnifiedStatePolicyV4(TextStatePolicyV2):
    """One actor for combat and language-conditioned building policies.

    All roles share geometry, entity/text attention, action queries and action
    heads.  A small role embedding is the only role-specific model input;
    non-text roles receive the same learned null-text memory as masked prompts.
    """

    def __init__(self, cfg):
        if cfg.profile != "unified":
            raise ValueError("v4 requires the unified profile")
        base = TextStatePolicyV2Args(
            num_block_classes=cfg.num_block_classes,
            item_vocab_size=cfg.item_vocab_size,
            profile="language_builder",
            hidden=cfg.hidden,
            heads=cfg.heads,
            depth=cfg.depth,
            text_hidden_size=cfg.text_hidden_size,
            dropout=cfg.dropout,
            gradient_checkpointing=cfg.gradient_checkpointing,
            image_h=cfg.image_h,
            image_w=cfg.image_w,
            voxel_channels=cfg.voxel_channels,
            target_pointer=cfg.target_pointer)
        super().__init__(base)
        self.cfg = cfg
        self.policy_role = nn.Embedding(cfg.num_policy_roles, cfg.hidden)
        self.role_fusion = nn.Sequential(
            nn.LayerNorm(cfg.hidden), nn.Linear(cfg.hidden, cfg.hidden))
        nn.init.normal_(self.role_fusion[-1].weight, std=1e-3)
        nn.init.zeros_(self.role_fusion[-1].bias)

    def encode(self, inputs, return_aux=False):
        inputs = dict(inputs)
        batch = len(inputs["target_agent"])
        device = inputs["target_agent"].device
        for name in ("shared", "current"):
            text_key, mask_key = name + "_text", name + "_text_mask"
            if text_key not in inputs:
                inputs[text_key] = torch.zeros(
                    batch, 1, self.cfg.text_hidden_size, device=device)
                inputs[mask_key] = torch.zeros(
                    batch, 1, dtype=torch.bool, device=device)
        if "policy_role" not in inputs:
            target = inputs["target_agent"].long()
            target_kind = inputs["resident_type"][
                torch.arange(batch, device=device), -1, target]
            inputs["policy_role"] = (target_kind == 2).long()
        hidden, target_logits, entity_logits = super().encode(inputs, return_aux=True)
        role = self.policy_role(inputs["policy_role"].long())[:, None]
        hidden = hidden + self.role_fusion(hidden + role)
        return (hidden, target_logits, entity_logits) if return_aux else hidden
