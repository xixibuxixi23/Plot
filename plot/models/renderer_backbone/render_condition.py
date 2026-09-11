"""Target-centric structured render conditions for the multi-agent Pixel DiT."""

import torch
from torch import nn


class MultiAgentRenderConditionEncoder(nn.Module):
    """Encode complete per-target render state without mixing noisy views."""

    def __init__(
        self,
        output_dim: int,
        action_dim: int,
        item_vocab_size: int,
        item_embedding_dim: int,
        skin_embedding_dim: int,
        max_agents: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.max_agents = max_agents
        self.item_vocab_size = item_vocab_size
        self.skin_embedding_dim = skin_embedding_dim
        self.position_scale = 48.0
        self.num_appearance_views = 4
        self.item_embedder = nn.Embedding(item_vocab_size, item_embedding_dim)
        self.slot_embedder = nn.Embedding(9, item_embedding_dim)
        appearance_view_dim = max(32, skin_embedding_dim // 2)
        self.skin_view_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, appearance_view_dim),
            nn.SiLU(),
        )
        self.appearance_direction_embedding = nn.Parameter(
            torch.zeros(self.num_appearance_views, appearance_view_dim)
        )
        nn.init.normal_(self.appearance_direction_embedding, std=0.02)
        self.skin_view_fusion = nn.Sequential(
            nn.Linear(self.num_appearance_views * appearance_view_dim, skin_embedding_dim),
            nn.SiLU(),
        )

        # Motion and orientation are intentionally omitted. Camera geometry is
        # consumed only by the core rasterizer and the screen-space projection.
        state_dim = 3
        # Keep all nine hotbar slots in a fixed order. Mean pooling would lose
        # which block belongs to which HUD slot.
        target_input_dim = state_dim + 11 * item_embedding_dim + skin_embedding_dim
        other_input_dim = state_dim + action_dim + item_embedding_dim + skin_embedding_dim
        self.target_mlp = nn.Sequential(
            nn.Linear(target_input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.other_mlp = nn.Sequential(
            nn.Linear(other_input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.other_aggregator = nn.Sequential(
            nn.Linear(2 * output_dim + 1, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )
        self.output_norm = nn.LayerNorm(output_dim)

    def _item_ids(self, value: torch.Tensor, name: str) -> torch.Tensor:
        value = value.long()
        if value.numel() and (value.min() < 0 or value.max() >= self.item_vocab_size):
            raise ValueError(f"{name} contains ids outside [0, {self.item_vocab_size - 1}]")
        return value

    def _encode_appearance(
        self,
        skins: torch.Tensor | None,
        appearance_valid: torch.Tensor | None,
        bsz: int,
        agents: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if skins is None:
            return torch.zeros(
                (bsz, agents, self.skin_embedding_dim), device=device, dtype=dtype
            )

        # New format: [B,A,V,RGBA,H,W].  Accept the old [B,A,RGB,H,W]
        # format as a single front view so old datasets/checkpoints fail softly.
        if skins.ndim == 5:
            skins = skins[:, :, None]
        if skins.ndim != 6 or skins.shape[:2] != (bsz, agents):
            raise ValueError(
                "player_skin must be [B,A,C,H,W] or [B,A,V,C,H,W], "
                f"got {tuple(skins.shape)}"
            )
        views, channels, height, width = skins.shape[2:]
        if views > self.num_appearance_views:
            raise ValueError(
                f"Got {views} appearance views, expected at most {self.num_appearance_views}"
            )
        if channels == 3:
            alpha = torch.ones((*skins.shape[:3], 1, height, width), device=skins.device, dtype=skins.dtype)
            skins = torch.cat((skins, alpha), dim=3)
        elif channels != 4:
            raise ValueError(f"player_skin must have RGB or RGBA channels, got {channels}")

        if appearance_valid is None:
            view_valid = torch.ones((bsz, agents, views), device=device, dtype=torch.bool)
        else:
            view_valid = appearance_valid.to(device=device, dtype=torch.bool)
            if view_valid.ndim == 2:
                view_valid = view_valid[:, :, None].expand(-1, -1, views)
            if view_valid.shape != (bsz, agents, views):
                raise ValueError(
                    f"player_appearance_valid must be {(bsz, agents, views)}, "
                    f"got {tuple(view_valid.shape)}"
                )

        view_embedding = self.skin_view_encoder(
            skins.to(device=device, dtype=dtype).reshape(bsz * agents * views, 4, height, width)
        ).reshape(bsz, agents, views, -1)
        view_embedding = view_embedding + self.appearance_direction_embedding[:views].to(dtype)[None, None]
        view_embedding = view_embedding * view_valid.to(dtype)[..., None]

        if views < self.num_appearance_views:
            padding = torch.zeros(
                (bsz, agents, self.num_appearance_views - views, view_embedding.shape[-1]),
                device=device,
                dtype=dtype,
            )
            view_embedding = torch.cat((view_embedding, padding), dim=2)
        skin_embedding = self.skin_view_fusion(view_embedding.flatten(start_dim=2))
        return skin_embedding * view_valid.any(dim=2).to(dtype)[..., None]

    def forward(self, cond: dict, return_appearance: bool = False):
        action = cond["action"]
        if action.ndim != 4:
            raise ValueError(f"multi-agent action must be [B,T,A,D], got {tuple(action.shape)}")
        bsz, timesteps, agents, _ = action.shape
        if agents > self.max_agents:
            raise ValueError(f"Got {agents} agents, but max_agents={self.max_agents}")
        device = action.device
        dtype = action.dtype

        position = cond.get("player_position")
        if position is None:
            raise ValueError("player_position is required")
        state = position / self.position_scale

        wielded = cond.get("wielded_item_id")
        inventory = cond.get("inventory_item_ids")
        selected = cond.get("selected_slot")
        if wielded is None:
            wielded = torch.zeros((bsz, timesteps, agents), device=device, dtype=torch.long)
        if inventory is None:
            inventory = torch.zeros((bsz, timesteps, agents, 9), device=device, dtype=torch.long)
        if selected is None:
            selected = torch.zeros((bsz, timesteps, agents), device=device, dtype=torch.long)
        wielded_embedding = self.item_embedder(self._item_ids(wielded, "wielded_item_id"))
        inventory_embedding = self.item_embedder(
            self._item_ids(inventory, "inventory_item_ids")
        ).flatten(start_dim=-2)
        if selected.numel() and (selected.min() < 0 or selected.max() >= 9):
            raise ValueError("selected_slot must contain ids in [0, 8]")
        slot_embedding = self.slot_embedder(selected.long())

        skin_embedding = self._encode_appearance(
            cond.get("player_skin"),
            cond.get("player_appearance_valid"),
            bsz,
            agents,
            device,
            dtype,
        )
        skin_per_frame = skin_embedding[:, None].expand(-1, timesteps, -1, -1)

        target_condition = self.target_mlp(
            torch.cat(
                (state, wielded_embedding, inventory_embedding, slot_embedding, skin_per_frame),
                dim=-1,
            )
        )

        relative_state = state[:, :, None] - state[:, :, :, None]
        other_action = action[:, :, None].expand(-1, -1, agents, -1, -1)
        other_wielded = wielded_embedding[:, :, None].expand(-1, -1, agents, -1, -1)
        other_skin = skin_per_frame[:, :, None].expand(-1, -1, agents, -1, -1)
        other_condition = self.other_mlp(
            torch.cat((relative_state, other_action, other_wielded, other_skin), dim=-1)
        )

        player_valid = cond.get("player_valid")
        if player_valid is None:
            player_valid = torch.ones((bsz, timesteps, agents), device=device, dtype=torch.bool)
        player_valid = player_valid.to(device=device, dtype=torch.bool)
        source_valid = player_valid[:, :, None, :].expand(-1, -1, agents, -1)
        target_valid = player_valid[:, :, :, None].expand(-1, -1, -1, agents)
        not_self = ~torch.eye(agents, device=device, dtype=torch.bool)[None, None]
        valid_others = source_valid & target_valid & not_self
        weights = valid_others.to(dtype)[..., None]
        valid_count = weights.sum(dim=3)
        other_mean = (other_condition * weights).sum(dim=3) / valid_count.clamp_min(1.0)
        other_max = other_condition.masked_fill(~valid_others[..., None], -torch.inf).amax(dim=3)
        has_others = valid_count > 0
        other_max = torch.where(has_others, other_max, torch.zeros_like(other_max))
        normalized_count = valid_count / max(self.max_agents - 1, 1)
        other_summary = self.other_aggregator(
            torch.cat((other_mean, other_max, normalized_count), dim=-1)
        )
        other_summary = other_summary * has_others.to(dtype)

        output = self.output_norm(target_condition + other_summary)
        output = output * player_valid.to(dtype)[..., None]
        if return_appearance:
            return output, skin_embedding
        return output
