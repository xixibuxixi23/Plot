"""M3: shared causal Pixel DiT with PLOT resident and memory conditions."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .renderer_backbone.dit_pixel import FrameDepthStackPixelDiT
from .renderer_backbone.render_condition import MultiAgentRenderConditionEncoder
from .renderer_backbone.player_spatial_condition import PlayerSpatialCondition


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
    cache_frames: int = 32
    gradient_checkpointing: bool = True
    gpu_rasterizer: bool = True
    deep_condition_reinjection: bool = False


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
        skins = cond["player_skin"]
        if skins.ndim != 6 or skins.shape[2] != 4:
            raise ValueError("player_skin requires front/back/left/right [B,A,4,C,H,W]")
        appearance = self._encode_appearance(
            skins, cond.get("player_appearance_valid"), b, a, position.device, position.dtype)
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
        if cfg.cache_frames < 8 or cfg.context_frames < 9:
            raise ValueError("M3 requires room for an 8-frame output block and its prefix")
        self.cfg = cfg
        self.voxel_embedder = nn.Embedding(cfg.num_block_classes + 1, cfg.voxel_channels)
        self.resident_encoder = ResidentConditionEncoder(cfg)
        self.spatial_encoder = PlayerSpatialCondition(128, cfg.actor_channels, cfg.input_h, cfg.input_w)
        self.core = FrameDepthStackPixelDiT(
            input_h=cfg.input_h, input_w=cfg.input_w, in_channels=cfg.in_channels,
            hidden_size=cfg.hidden_size, depth=cfg.depth, num_heads=cfg.num_heads,
            raster_cond_shape=(cfg.voxel_channels, cfg.input_h, cfg.input_w),
            extra_condition_dim=cfg.condition_dim, actor_condition_dim=cfg.actor_channels,
            context_window_size=cfg.context_frames, cache_window_size=cfg.cache_frames,
            voxel_dim=48, is_causal=True, use_condition_mask=True,
            deep_condition_reinjection=cfg.deep_condition_reinjection,
            hud_condition_dim=3 if cfg.deep_condition_reinjection else 0,
            gradient_checkpointing=cfg.gradient_checkpointing,
            aggregation_config={} if cfg.gpu_rasterizer else None)

    @staticmethod
    def select(value, target):
        return value[torch.arange(len(target), device=value.device), :, target]

    def encode_conditions(self, cond):
        if {"instance_mask", "entity_mask", "weapon_texture", "crop_anchor"} & cond.keys():
            raise ValueError("supervision masks, weapon textures and crop anchors are not neural inputs")
        target = cond["target_agent"].long()
        extra, appearance = self.resident_encoder(cond)
        spatial_cond = dict(cond, camera_position=cond["player_position"] + cond["camera_relative"],
                            camera=cond["fov_x"][..., None])
        spatial = self.spatial_encoder(spatial_cond, appearance)
        result = {
            "action": self.select(cond["action"], target),
            "extra_condition": self.select(extra, target),
            "actor_spatial_condition": self.select(spatial, target),
            "condition_mask": cond["condition_mask"],
            "action_prefix_mask": cond["action_prefix_mask"],
        }
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
