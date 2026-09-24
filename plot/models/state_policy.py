"""Independent state-conditioned NPC policy; no renderer or RGB dependency."""
from dataclasses import dataclass

import torch
from torch import nn

from .structured_action import StructuredActionHead, constrain_peaceful_logits

NPC_PROFILES = ('villager_peaceful', 'zombie_melee', 'skeleton_swordsman',
                'villager_defender')
STATE_PROFILES = NPC_PROFILES + ('language_builder',)


@dataclass(frozen=True)
class StatePolicyArgs:
    num_block_classes: int
    item_vocab_size: int
    profile: str
    hidden: int = 256
    heads: int = 8
    resident_layers: int = 2
    temporal_layers: int = 4
    decoder_layers: int = 3
    dropout: float = 0.1
    # xyz, angle sin/cos, hp, camera-relative xyz, camera direction, 4 event cues.
    state_dim: int = 18
    text_hidden_size: int = 0


class GeometryEncoder(nn.Module):
    def __init__(self, classes, hidden):
        super().__init__()
        self.unknown = classes
        self.embedding = nn.Embedding(classes + 1, 16)
        def tower(first_stride):
            return nn.Sequential(
                nn.Conv3d(16, 32, 3, stride=first_stride, padding=1),
                nn.GroupNorm(8, 32), nn.SiLU(),
                nn.Conv3d(32, 64, 3, stride=2, padding=1),
                nn.GroupNorm(8, 64), nn.SiLU(),
                nn.AdaptiveAvgPool3d(4), nn.Conv3d(64, hidden, 1))
        self.near = tower(2)
        self.far = tower(3)
        # Relative token-center coordinates in voxel units, scaled by 24.
        centers = torch.arange(4, dtype=torch.float32) + .5
        grids = [torch.stack(torch.meshgrid(*([centers * (size / 4) - size / 2] * 3),
                                            indexing='ij'), -1).reshape(64, 3) / 24
                 for size in (16, 48)]
        self.register_buffer('coordinates', torch.cat(grids))
        self.position = nn.Linear(3, hidden)
        self.scale = nn.Parameter(torch.randn(2, 1, hidden) * .02)

    def forward(self, blocks, known, grid_offset):
        if blocks.shape[1:] != (48, 48, 48) or known.shape != blocks.shape:
            raise ValueError('geometry must be [B,48,48,48] with known mask')
        ids = torch.where(known.bool(), blocks.long(), self.unknown)
        x = self.embedding(ids).permute(0, 4, 1, 2, 3)
        near = self.near(x[:, :, 16:32, 16:32, 16:32]).flatten(2).transpose(1, 2)
        far = self.far(x).flatten(2).transpose(1, 2)
        tokens = torch.cat((near + self.scale[0], far + self.scale[1]), 1)
        return tokens + self.position(self.coordinates[None] + grid_offset[:, None] / 24)


class StateInhabitantPolicy(nn.Module):
    """One profile per checkpoint, eight past states -> eight future actions.

    Resident axis is an unordered set. Target is selected via target_agent;
    positions and grid_offset are relative to its current world position.
    history_valid allows a padded prefix but the last state must be valid.
    """
    def __init__(self, cfg: StatePolicyArgs):
        super().__init__()
        if cfg.profile not in STATE_PROFILES:
            raise ValueError(f'unsupported NPC profile: {cfg.profile}')
        self.cfg = cfg
        d = cfg.hidden
        if cfg.profile == 'language_builder':
            if cfg.text_hidden_size <= 0: raise ValueError('language policy requires text features')
            self.text_projection = nn.Linear(cfg.text_hidden_size, d)
            self.text_role = nn.Parameter(torch.randn(2, 1, d) * .02)
        self.geometry = GeometryEncoder(cfg.num_block_classes, d)
        self.item = nn.Embedding(cfg.item_vocab_size, 32)
        self.kind = nn.Embedding(4, 16)
        self.state = nn.Sequential(nn.Linear(cfg.state_dim + 48, d), nn.SiLU(),
                                   nn.Linear(d, d), nn.LayerNorm(d))
        self.self_marker = nn.Parameter(torch.randn(d) * .02)
        def encoder(n):
            layer = nn.TransformerEncoderLayer(d, cfg.heads, 4*d, cfg.dropout,
                                               activation='gelu', batch_first=True,
                                               norm_first=True)
            return nn.TransformerEncoder(layer, n, norm=nn.LayerNorm(d),
                                         enable_nested_tensor=False)
        self.residents = encoder(cfg.resident_layers)
        self.history = encoder(cfg.temporal_layers)
        self.action = nn.Linear(23, d)
        self.time = nn.Parameter(torch.randn(1, 8, d) * .02)
        # Optional explicitly supplied task anchor; zero mask means unknown.
        self.goal = nn.Linear(5, d)
        self.query = nn.Parameter(torch.randn(1, 8, d) * .02)
        layer = nn.TransformerDecoderLayer(d, cfg.heads, 4*d, cfg.dropout,
                                           activation='gelu', batch_first=True,
                                           norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, cfg.decoder_layers, nn.LayerNorm(d))
        self.head = StructuredActionHead(d, horizons=1)
        self.loss_head = StructuredActionHead(d, horizons=8)
        # Loss/decode only need the head's buffers; avoid unused trainable heads.
        for name in ('keys', 'hotbar', 'mouse_x', 'mouse_y'):
            delattr(self.loss_head, name)

    def forward(self, inputs):
        state = inputs['resident_state']
        b, t, a, _ = state.shape
        if t != 8 or not inputs['history_valid'][:, -1].all():
            raise ValueError('eight padded/completed states ending in a valid current state required')
        target = inputs['target_agent'].long()
        valid = inputs['resident_valid'].bool()
        if not valid[torch.arange(b, device=state.device), -1, target].all():
            raise ValueError('target must exist at the decision boundary')
        x = self.state(torch.cat((state, self.item(inputs['held_item'].long()),
                                 self.kind(inputs['resident_type'].long())), -1))
        marker = torch.nn.functional.one_hot(target, a).to(x.dtype)
        x = x + marker[:, None, :, None] * self.self_marker
        padding = ~valid.flatten(0, 1)
        # Padded history frames need a nonempty attention row; they are masked later.
        padding = padding.clone()
        padding[padding.all(-1), 0] = False
        x = self.residents(x.flatten(0, 1), src_key_padding_mask=padding).reshape(b, t, a, -1)
        own = x[torch.arange(b, device=x.device), :, target]
        hmask = ~inputs['history_valid'].bool()
        own = own + self.action(inputs['incoming_actions']) + self.time
        history = self.history(own, src_key_padding_mask=hmask)
        geometry = self.geometry(inputs['voxel_classes'], inputs['voxel_known'],
                                 inputs['grid_offset'])
        goal = inputs['goal']
        goal_token = self.goal(goal)[:, None]
        memory = torch.cat((geometry, history, goal_token), 1)
        mask = torch.cat((torch.zeros(b, 128, dtype=torch.bool, device=x.device),
                          hmask, ~goal[:, -1:].bool()), 1)
        if self.cfg.profile == 'language_builder':
            texts = []; text_masks = []
            for role, name in enumerate(('shared', 'current')):
                features = inputs[name+'_text'].to(x.dtype)
                text_valid = inputs[name+'_text_mask'].bool()
                texts.append(self.text_projection(features) + self.text_role[role])
                text_masks.append(~text_valid)
            memory = torch.cat((memory, *texts), 1)
            mask = torch.cat((mask, *text_masks), 1)
        query = self.query.expand(b, -1, -1) + history[:, -1:]
        decoded = self.decoder(query, memory, memory_key_padding_mask=mask)
        logits = {key: value.squeeze(-2) for key, value in self.head(decoded).items()}
        peaceful = torch.full((b,), self.cfg.profile == 'villager_peaceful',
                              device=x.device, dtype=torch.bool)
        return constrain_peaceful_logits(logits, peaceful)

    def loss(self, logits, actions, valid_mask=None, *, attack_positive_weight=1.,
             horizon_weights=None, sample_weight=None):
        if attack_positive_weight == 1.:
            return self.loss_head.loss(logits, actions, valid_mask,
                                       horizon_weights=horizon_weights, sample_weight=sample_weight)
        if attack_positive_weight < 1.:
            raise ValueError('attack_positive_weight must be >= 1')
        _, parts = self.loss_head.loss(logits, actions, valid_mask)
        targets = self.loss_head.targets(actions)
        positive_weights = logits['keys'].new_ones(10)
        positive_weights[7] = attack_positive_weight  # action index 8 = attack/dig
        parts['keys'] = torch.nn.functional.binary_cross_entropy_with_logits(
            logits['keys'], targets['keys'], pos_weight=positive_weights, reduction='none').mean(-1)
        combined = sum(parts.values())
        valid = torch.ones_like(combined) if valid_mask is None else valid_mask.to(combined)
        horizon = torch.ones(8, device=combined.device) if horizon_weights is None else horizon_weights.to(combined)
        weights = valid * horizon[None]
        denominator = weights.sum(-1)
        per_sample = (combined * weights).sum(-1) / denominator.clamp_min(1e-8)
        sample_valid = (denominator > 0).to(combined)
        if sample_weight is not None: sample_valid = sample_valid * sample_weight.to(combined)
        return (per_sample * sample_valid).sum() / sample_valid.sum().clamp_min(1), parts

    def decode(self, logits):
        return self.loss_head.decode(logits)
