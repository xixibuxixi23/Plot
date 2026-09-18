"""M3: shared causal Pixel DiT with PLOT resident and memory conditions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .renderer_backbone.dit_pixel import FrameDepthStackPixelDiT
from .renderer_backbone.render_condition import MultiAgentRenderConditionEncoder
from .renderer_backbone.player_spatial_condition import (
    PlayerSpatialCondition,
    ViewAwarePlayerAppearance,
    PlayerReferenceEncoder,
    PlayerReferenceLayout,
    ROIAppearanceProjector,
)


@dataclass(frozen=True)
class RendererArgs:
    num_block_classes: int
    item_vocab_size: int
    input_h: int = 36
    input_w: int = 64
    in_channels: int = 16
    hidden_size: int = 1024
    depth: int = 12
    num_heads: int = 16
    voxel_channels: int = 32
    condition_dim: int = 256
    actor_channels: int = 16
    max_agents: int = 8
    context_frames: int = 65
    cache_frames: int = 64
    block_frames: int = 8
    gradient_checkpointing: bool = True
    gpu_rasterizer: bool = True
    deep_condition_reinjection: bool = False
    view_aware_appearance: bool = False
    detail_preserving_appearance: bool = False
    entity_reference_attention: bool = False
    unified_player_reference: bool = False
    player_reference_grid_size: tuple[int, int] = (8, 4)
    player_reference_position_encoding: bool = False
    geometry_aware_player_reference: bool = False
    unified_reference_reinject_blocks: tuple[int, ...] = ()
    simple_conditioning: bool = False


class SimpleResidentConditionEncoder(nn.Module):
    """Encode target state globally and other-resident state spatially.

    Appearance is intentionally absent here: it has one route through the
    compact reference memory. Other residents retain their individual action
    and state until after screen-space projection.
    """

    def __init__(self, cfg: RendererArgs):
        super().__init__()
        if cfg.actor_channels < 3:
            raise ValueError("M3-Simple actor_channels must be at least 3")
        self.max_agents = cfg.max_agents
        self.actor_feature_dim = cfg.actor_channels - 2
        self.item_embedder = nn.Embedding(cfg.item_vocab_size, 64)
        self.kind_embedder = nn.Embedding(4, 16)
        # hp, sin/cos(yaw,pitch), four event cues, held item, resident kind.
        local_dim = 1 + 4 + 4 + 64 + 16
        self.actor_mlp = nn.Sequential(
            nn.Linear(local_dim + 23, cfg.condition_dim),
            nn.SiLU(),
            nn.Linear(cfg.condition_dim, self.actor_feature_dim),
        )
        self.target_mlp = nn.Sequential(
            nn.Linear(local_dim + 3, cfg.condition_dim),
            nn.SiLU(),
            nn.Linear(cfg.condition_dim, cfg.condition_dim),
            nn.LayerNorm(cfg.condition_dim),
        )

    def forward(self, cond):
        position = cond["player_position"]
        if position.shape[2] > self.max_agents:
            raise ValueError("resident count exceeds max_agents")
        hp = cond["hp"][..., None] / 20.0
        angles = cond["yaw_pitch"]
        local = torch.cat(
            (hp, angles.sin(), angles.cos(), cond["event_cues"],
             self.item_embedder(cond["held_item"].long()),
             self.kind_embedder(cond["resident_type"].long())), dim=-1)
        target = self.target_mlp(torch.cat((local, cond["camera_relative"]), dim=-1))
        actor = self.actor_mlp(torch.cat((local, cond["action"]), dim=-1))
        valid = cond["player_valid"].to(actor.dtype)[..., None]
        return target * valid, actor * valid


class ResidentConditionEncoder(MultiAgentRenderConditionEncoder):
    """Reuse four-view appearance encoding, replace the legacy inventory state.

    Angles are radians in ENU. event_cues = [voxel writes caused, attacks caused,
    damage received, HP delta] for the incoming transition. No text or ID masks.
    """

    def __init__(self, cfg: RendererArgs):
        super().__init__(output_dim=cfg.condition_dim, action_dim=23,
                         item_vocab_size=cfg.item_vocab_size, item_embedding_dim=64,
                         skin_embedding_dim=128, max_agents=cfg.max_agents)
        # These legacy modules would otherwise be unused parameters under DDP.
        del self.target_mlp, self.other_mlp, self.slot_embedder, self.other_aggregator
        self.kind_embedder = nn.Embedding(4, 16)  # human, villager, zombie, skeleton
        self.use_pooled_appearance = not cfg.unified_player_reference
        if not self.use_pooled_appearance:
            # In the unified path, appearance has exactly one route: native
            # reference tokens. Keep the downstream state MLP shape checkpoint
            # compatible, but remove the old four-view pooling parameters.
            del self.skin_view_encoder
            del self.appearance_direction_embedding
            del self.skin_view_fusion
        state_dim = 3 + 1 + 4 + 4 + 64 + 128 + 16
        self.target_mlp = nn.Sequential(nn.Linear(state_dim + 3, cfg.condition_dim),
                                        nn.SiLU(), nn.Linear(cfg.condition_dim, cfg.condition_dim))
        self.other_mlp = nn.Sequential(nn.Linear(state_dim + 23, cfg.condition_dim),
                                       nn.SiLU(), nn.Linear(cfg.condition_dim, cfg.condition_dim))

    def forward(self, cond):
        position = cond["player_position"]
        b, t, a, _ = position.shape
        if a > self.max_agents:
            raise ValueError("resident count exceeds max_agents")
        if self.use_pooled_appearance:
            skins = cond["player_skin"]
            if skins.ndim != 6 or skins.shape[2] != 4:
                raise ValueError("player_skin requires front/back/left/right [B,A,4,C,H,W]")
            appearance = self._encode_appearance(
                skins, cond.get("player_appearance_valid"), b, a,
                position.device, position.dtype)
        else:
            appearance = position.new_zeros((b, a, self.skin_embedding_dim))
        hp = cond["hp"][..., None] / 20.0
        angles = cond["yaw_pitch"]
        shared = torch.cat((hp, angles.sin(), angles.cos(), cond["event_cues"],
                            self.item_embedder(cond["held_item"].long()),
                            appearance[:, None].expand(-1, t, -1, -1),
                            self.kind_embedder(cond["resident_type"].long())), dim=-1)
        # A translation of the complete world cannot alter these neural features.
        target = self.target_mlp(torch.cat((torch.zeros_like(position), shared,
                                            cond["camera_relative"]), dim=-1))
        relative = (position[:, :, None] - position[:, :, :, None]) / 48.0
        other = torch.cat((relative, shared[:, :, None].expand(-1, -1, a, -1, -1),
                           cond["action"][:, :, None].expand(-1, -1, a, -1, -1)), dim=-1)
        other = self.other_mlp(other)
        valid = cond["player_valid"].bool()
        pairs = valid[:, :, :, None] & valid[:, :, None, :]
        pairs &= ~torch.eye(a, dtype=torch.bool, device=position.device)[None, None]
        pooled = (other * pairs[..., None]).sum(3) / pairs.sum(3).clamp_min(1)[..., None]
        return self.output_norm(target + pooled) * valid[..., None], appearance


class Renderer(nn.Module):
    """One target view per batch row, with all residents as state conditions.

    Input latent [B,T,C,H,W]; conditions contain all residents [B,T,A,...].
    target_agent [B] selects the view. Batch rows have independent KV histories.
    crop anchors belong to the geometry adapter and never enter this module.
    """

    def __init__(self, cfg: RendererArgs):
        super().__init__()
        if cfg.block_frames < 1:
            raise ValueError("block_frames must be positive")
        if any(size < 1 for size in cfg.player_reference_grid_size):
            raise ValueError("player reference grid dimensions must be positive")
        if cfg.detail_preserving_appearance and not cfg.view_aware_appearance:
            raise ValueError("detail-preserving appearance requires view-aware appearance")
        if cfg.unified_player_reference and (
            cfg.view_aware_appearance
            or cfg.detail_preserving_appearance
            or cfg.entity_reference_attention
        ):
            raise ValueError(
                "unified player reference replaces all legacy appearance/reference paths"
            )
        if cfg.simple_conditioning and not cfg.unified_player_reference:
            raise ValueError("M3-Simple requires the single unified player reference")
        if cfg.simple_conditioning and (
            cfg.deep_condition_reinjection
            or cfg.geometry_aware_player_reference
            or cfg.unified_reference_reinject_blocks
        ):
            raise ValueError(
                "M3-Simple has one joint spatial input before the DiT; "
                "deep/geometry-aware/repeated adapters are not supported"
            )
        if cfg.geometry_aware_player_reference and not cfg.unified_player_reference:
            raise ValueError(
                "geometry-aware player reference requires unified player reference"
            )
        if cfg.cache_frames < cfg.block_frames or cfg.context_frames < 1 + cfg.block_frames:
            raise ValueError("M3 requires room for one output block and its prefix")
        if (cfg.context_frames - 1) % cfg.block_frames:
            raise ValueError("M3 context after the first frame must contain complete blocks")
        self.cfg = cfg
        self.voxel_embedder = nn.Embedding(cfg.num_block_classes + 1, cfg.voxel_channels)
        self.resident_encoder = (
            SimpleResidentConditionEncoder(cfg)
            if cfg.simple_conditioning
            else ResidentConditionEncoder(cfg)
        )
        spatial_feature_dim = cfg.actor_channels - 2 if cfg.simple_conditioning else 128
        self.spatial_encoder = PlayerSpatialCondition(
            spatial_feature_dim, cfg.actor_channels, cfg.input_h, cfg.input_w
        )
        self.appearance_spatial_encoder = (
            ViewAwarePlayerAppearance(cfg.input_h, cfg.input_w)
            if cfg.view_aware_appearance
            else None
        )
        use_reference_tokens = cfg.entity_reference_attention or cfg.unified_player_reference
        self.reference_encoder = (
            PlayerReferenceEncoder(
                256,
                grid_size=cfg.player_reference_grid_size,
                position_encoding=cfg.player_reference_position_encoding,
            )
            if use_reference_tokens
            else None
        )
        self.reference_layout = (
            PlayerReferenceLayout(cfg.input_h, cfg.input_w)
            if use_reference_tokens else None
        )
        self.roi_appearance_projector = (
            ROIAppearanceProjector(
                reference_dim=256,
                actor_dim=spatial_feature_dim,
                output_channels=32,
                height=cfg.input_h,
                width=cfg.input_w,
            )
            if cfg.simple_conditioning
            else None
        )
        self.core = FrameDepthStackPixelDiT(
            input_h=cfg.input_h, input_w=cfg.input_w, in_channels=cfg.in_channels,
            hidden_size=cfg.hidden_size, depth=cfg.depth, num_heads=cfg.num_heads,
            raster_cond_shape=(cfg.voxel_channels, cfg.input_h, cfg.input_w),
            extra_condition_dim=cfg.condition_dim, actor_condition_dim=cfg.actor_channels,
            roi_appearance_dim=(
                self.roi_appearance_projector.output_channels
                if self.roi_appearance_projector is not None
                else 0
            ),
            context_window_size=cfg.context_frames, cache_window_size=cfg.cache_frames,
            voxel_dim=48, is_causal=True, causal_block_size=cfg.block_frames,
            use_condition_mask=True,
            deep_condition_reinjection=cfg.deep_condition_reinjection,
            hud_condition_dim=3 if cfg.deep_condition_reinjection else 0,
            appearance_condition_dim=(
                self.appearance_spatial_encoder.output_channels
                if self.appearance_spatial_encoder is not None
                else 0
            ),
            detail_preserving_appearance=cfg.detail_preserving_appearance,
            entity_reference_dim=256 if cfg.entity_reference_attention else 0,
            unified_reference_dim=(
                256 if cfg.unified_player_reference and not cfg.simple_conditioning else 0
            ),
            unified_reference_grid_size=cfg.player_reference_grid_size,
            geometry_aware_player_reference=cfg.geometry_aware_player_reference,
            unified_reference_reinject_blocks=cfg.unified_reference_reinject_blocks,
            simple_conditioning=cfg.simple_conditioning,
            gradient_checkpointing=cfg.gradient_checkpointing,
            aggregation_config={} if cfg.gpu_rasterizer else None)

    @staticmethod
    def select(value, target):
        return value[torch.arange(len(target), device=value.device), :, target]

    def encode_conditions(self, cond):
        if {"instance_mask", "entity_mask", "weapon_texture", "crop_anchor"} & cond.keys():
            raise ValueError("supervision masks, weapon textures and crop anchors are not neural inputs")
        target = cond["target_agent"].long()
        extra, actor_features = self.resident_encoder(cond)
        spatial_cond = dict(cond, camera_position=cond["player_position"] + cond["camera_relative"],
                            camera=cond["fov_x"][..., None])
        spatial = self.spatial_encoder(spatial_cond, actor_features)
        result = {
            "action": self.select(cond["action"], target),
            "extra_condition": self.select(extra, target),
            "actor_spatial_condition": self.select(spatial, target),
            "condition_mask": cond["condition_mask"],
            "action_prefix_mask": cond["action_prefix_mask"],
        }
        if self.appearance_spatial_encoder is not None:
            result["appearance_spatial_condition"] = self.appearance_spatial_encoder(
                spatial_cond, target
            )
        if self.reference_encoder is not None:
            reference = cond.get("player_reference")
            if reference is None:
                reference = cond["player_skin"]
            reference_tokens = self.reference_encoder(
                reference, cond.get("player_appearance_valid")
            )
            roi, view_weights, local_coordinates = self.reference_layout(
                spatial_cond, target
            )
            if self.cfg.simple_conditioning:
                valid = cond.get("player_appearance_valid")
                if valid is None:
                    valid = torch.ones(
                        reference_tokens.shape[:3], device=reference_tokens.device,
                        dtype=torch.bool,
                    )
                result["roi_appearance_condition"] = self.roi_appearance_projector(
                    reference_tokens,
                    roi,
                    local_coordinates,
                    actor_features,
                    valid.bool(),
                )
            elif self.cfg.unified_player_reference:
                result["unified_reference_tokens"] = reference_tokens
                result["unified_reference_roi"] = roi
                if self.cfg.geometry_aware_player_reference:
                    result["unified_reference_view_weights"] = view_weights
                    result["unified_reference_local_coordinates"] = local_coordinates
                valid = cond.get("player_appearance_valid")
                if valid is None:
                    valid = torch.ones(
                        reference_tokens.shape[:3], device=reference_tokens.device,
                        dtype=torch.bool,
                    )
                result["unified_reference_valid"] = valid.bool()
            else:
                result["entity_reference_tokens"] = reference_tokens
                result["entity_reference_roi"] = roi
                result["entity_reference_view_weights"] = view_weights
        if self.cfg.deep_condition_reinjection:
            target_hp = self.select(cond["hp"], target) / 20.0
            target_hp_delta = self.select(cond["event_cues"], target)[..., 3] / 20.0
            hud = target_hp.new_zeros(
                (*target_hp.shape, 3, self.cfg.input_h, self.cfg.input_w)
            )
            left = round(190 / 640 * self.cfg.input_w)
            right = round(314 / 640 * self.cfg.input_w)
            top = round(300 / 360 * self.cfg.input_h)
            bottom = round(322 / 360 * self.cfg.input_h)
            hud[..., 0, top:bottom, left:right] = 1
            hud[..., 1, top:bottom, left:right] = target_hp[..., None, None]
            hud[..., 2, top:bottom, left:right] = target_hp_delta[..., None, None]
            result["hud_condition"] = hud
        if "raster_features" in cond:
            result.update(raster_features=cond["raster_features"], raster_depth=cond["raster_depth"])
        else:
            blocks = cond["voxel_classes"].long()
            blocks = torch.where(cond["voxel_known"], blocks, self.cfg.num_block_classes)
            result["voxel_latents"] = self.voxel_embedder(blocks).permute(0, 1, 5, 2, 3, 4)
            result["camera"] = cond["raster_camera"]
        # Project once per state block, reuse across all denoising steps. This
        # is call-local, so changing state at the same index cannot reuse stale pixels.
        result["raster_embedding"] = self.core.project_raster(result, extra.dtype)
        for key in ("voxel_latents", "camera", "raster_features", "raster_depth"):
            result.pop(key, None)
        return result

    def forward(self, x, time, cond, **kwargs):
        return self.core(x, time, self.encode_conditions(cond), **kwargs)

    def policy_features(self, completed_latents, cond):
        """Return pooled read-only M3 features for M4 completed-frame history."""
        if completed_latents.ndim != 5:
            raise ValueError("completed_latents must be [B,T,C,H,W]")
        time = torch.zeros(completed_latents.shape[:2], device=completed_latents.device)
        condition = dict(cond, condition_mask=torch.ones_like(time, dtype=torch.bool))
        _, features = self(completed_latents, time, condition, return_features=True,
                           cache_write=False)
        return self.compress_policy_features(features)

    @staticmethod
    def compress_policy_features(features, output_dim=128):
        """Pool spatial tokens and deterministically compress channels for M4.

        This is parameter-free, so the interface is stable before M4 training
        and cannot create an accidental gradient path back into M3.
        """
        pooled = features.mean(dim=(-3, -2))
        # Adaptive pooling also keeps small contract-test models compatible.
        flattened = pooled.reshape(-1, 1, pooled.shape[-1])
        compressed = F.adaptive_avg_pool1d(flattened, output_dim)
        return compressed.reshape(*pooled.shape[:-1], output_dim).detach()

    def init_kv_cache(self, batch_size, dtype=None):
        return self.core.init_kv_cache(batch_size, dtype=dtype)

    def set_kv_cache_start(self, frame_index):
        """Position an empty cache at an absolute episode frame."""
        if self.core.kv_caches is None:
            raise RuntimeError("initialize the KV cache first")
        if any(int(cache["local_end_index"].item()) != 0 for cache in self.core.kv_caches):
            raise RuntimeError("cache start can only be set before the first commit")
        for cache in self.core.kv_caches:
            cache["global_end_index"].fill_(int(frame_index))

    def clear_cache(self):
        self.core.kv_caches = None
        self.core.raster_cache = None

    def commit_kv_candidates(self, candidates, frame_index):
        self.core.commit_kv_candidates(candidates, frame_index)

    def load_2daction_backbone(self, state: dict):
        """Transfer compatible DiT tensors, explicitly report every rejected key."""
        current = self.core.state_dict()
        accepted, skipped = {}, []
        for key, value in state.items():
            original = key
            for prefix in ("_orig_mod.", "denoiser.", "core."):
                key = key.removeprefix(prefix)
            if key in current and value.shape == current[key].shape:
                accepted[key] = value
            else:
                skipped.append(original)
        if not accepted:
            raise ValueError("checkpoint has no compatible 2DAction DiT tensors")
        missing = self.core.load_state_dict(accepted, strict=False).missing_keys
        return {"loaded": len(accepted), "missing": missing, "skipped": skipped}
