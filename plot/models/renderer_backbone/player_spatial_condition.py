"""Lightweight screen-space condition for other players."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PlayerSpatialCondition(nn.Module):
    """Project every other player to a soft screen-space rectangle.

    The branch only needs state already present in the dataset plus the existing
    four-view identity appearance embedding. It deliberately does not model a
    skeleton or require body-part annotations.
    """

    def __init__(
        self,
        appearance_dim: int,
        output_channels: int = 16,
        height: int = 36,
        width: int = 64,
        player_height: float = 1.8,
        player_aspect: float = 0.45,
        edge_sharpness: float = 2.0,
    ) -> None:
        super().__init__()
        if output_channels < 3:
            raise ValueError("output_channels must be at least 3")
        self.output_channels = output_channels
        self.height = height
        self.width = width
        self.player_height = player_height
        self.player_aspect = player_aspect
        self.edge_sharpness = edge_sharpness
        self.appearance_projection = nn.Linear(appearance_dim, output_channels - 2, bias=False)

    def _project(self, points, camera_position, forward, right, down, tan_half_fov_x):
        relative = points - camera_position
        depth = (relative * forward).sum(dim=-1)
        camera_x = (relative * right).sum(dim=-1)
        camera_y = (relative * down).sum(dim=-1)
        safe_depth = depth.clamp_min(1e-4)
        tan_half_fov_y = tan_half_fov_x * (self.height / self.width)
        u = (camera_x / safe_depth / tan_half_fov_x + 1.0) * (self.width - 1) / 2.0
        v = (camera_y / safe_depth / tan_half_fov_y + 1.0) * (self.height - 1) / 2.0
        return u, v, depth

    def forward(self, cond: dict, appearance_embedding: torch.Tensor) -> torch.Tensor:
        position = cond["player_position"]
        camera_position = cond["camera_position"]
        camera_direction = cond["camera_direction"]
        camera = cond["camera"]
        player_valid = cond.get("player_valid")

        bsz, timesteps, agents, _ = position.shape
        if camera_position.shape != position.shape or camera_direction.shape != position.shape:
            raise ValueError("camera_position and camera_direction must match player_position")
        if appearance_embedding.shape[:2] != (bsz, agents):
            raise ValueError("appearance_embedding must have shape [B,A,D]")
        if player_valid is None:
            player_valid = torch.ones((bsz, timesteps, agents), device=position.device, dtype=torch.bool)
        else:
            player_valid = player_valid.to(device=position.device, dtype=torch.bool)

        dtype = position.dtype
        forward = F.normalize(camera_direction, dim=-1, eps=1e-6)
        world_down = torch.tensor((0.0, 0.0, -1.0), device=position.device, dtype=dtype)
        right = F.normalize(torch.cross(world_down.expand_as(forward), forward, dim=-1), dim=-1, eps=1e-6)
        down = F.normalize(torch.cross(forward, right, dim=-1), dim=-1, eps=1e-6)

        # [B,T,target,source,3]
        camera_position = camera_position[:, :, :, None]
        forward_t = forward[:, :, :, None]
        right_t = right[:, :, :, None]
        down_t = down[:, :, :, None]
        feet = position[:, :, None].expand(-1, -1, agents, -1, -1)
        head = feet.clone()
        head[..., 2] += self.player_height
        tan_half_fov_x = torch.tan(camera[..., -1].clamp(0.05, 3.0) / 2.0)[:, :, :, None]

        foot_u, foot_v, foot_depth = self._project(
            feet, camera_position, forward_t, right_t, down_t, tan_half_fov_x
        )
        head_u, head_v, head_depth = self._project(
            head, camera_position, forward_t, right_t, down_t, tan_half_fov_x
        )
        center_u = (foot_u + head_u) / 2.0
        center_v = (foot_v + head_v) / 2.0
        box_h = (foot_v - head_v).abs().clamp(1.0, float(self.height))
        box_w = (box_h * self.player_aspect).clamp(1.0, float(self.width))
        depth = (foot_depth + head_depth) / 2.0

        grid_y = torch.arange(self.height, device=position.device, dtype=dtype)
        grid_x = torch.arange(self.width, device=position.device, dtype=dtype)
        dy = (grid_y.view(1, 1, 1, 1, self.height, 1) - center_v[..., None, None]).abs()
        dx = (grid_x.view(1, 1, 1, 1, 1, self.width) - center_u[..., None, None]).abs()
        mask = torch.sigmoid((box_h[..., None, None] / 2.0 - dy) * self.edge_sharpness)
        mask = mask * torch.sigmoid((box_w[..., None, None] / 2.0 - dx) * self.edge_sharpness)

        source_valid = player_valid[:, :, None, :]
        target_valid = player_valid[:, :, :, None]
        not_self = ~torch.eye(agents, device=position.device, dtype=torch.bool)[None, None]
        visible = source_valid & target_valid & not_self & (depth > 0.05)
        mask = mask * visible[..., None, None].to(dtype)

        inverse_depth = 1.0 / (1.0 + depth.clamp_min(0.0))

        appearance = torch.tanh(self.appearance_projection(appearance_embedding.to(dtype)))
        appearance = appearance[:, None, None].expand(-1, timesteps, agents, -1, -1)
        source_features = torch.cat(
            (
                torch.ones_like(depth)[..., None],
                inverse_depth[..., None],
                appearance,
            ),
            dim=-1,
        )
        spatial = (mask[..., None, :, :] * source_features[..., :, None, None]).sum(dim=3)
        return spatial.contiguous()


class ViewAwarePlayerAppearance(nn.Module):
    """Warp four-view RGBA references into the target camera at latent resolution.

    Unlike ``PlayerSpatialCondition``, this path never pools a skin into one
    identity vector.  It preserves the reference pixels, chooses a view from
    the source resident's yaw, and supplies local sprite coordinates so the
    DiT can distinguish corresponding texture locations.
    """

    output_channels = 11  # RGB, occupancy, inverse depth, local uv, four view weights

    def __init__(
        self,
        height: int = 36,
        width: int = 64,
        player_height: float = 1.8,
        player_aspect: float = 0.45,
        view_sharpness: float = 8.0,
        depth_sharpness: float = 8.0,
    ) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)
        self.player_height = float(player_height)
        self.player_aspect = float(player_aspect)
        self.view_sharpness = float(view_sharpness)
        self.depth_sharpness = float(depth_sharpness)

    @staticmethod
    def _select_agent(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch = torch.arange(len(target), device=value.device)
        return value[batch, :, target]

    @staticmethod
    def _project(points, camera_position, forward, right, down, tan_half_fov_x, height, width):
        relative = points - camera_position
        depth = (relative * forward).sum(dim=-1)
        safe_depth = depth.clamp_min(1e-4)
        tan_half_fov_y = tan_half_fov_x * (height / width)
        u = ((relative * right).sum(dim=-1) / safe_depth / tan_half_fov_x + 1) * (width - 1) / 2
        v = ((relative * down).sum(dim=-1) / safe_depth / tan_half_fov_y + 1) * (height - 1) / 2
        return u, v, depth

    def forward(self, cond: dict, target: torch.Tensor) -> torch.Tensor:
        skins = cond["player_skin"]
        position = cond["player_position"]
        yaw = cond["yaw_pitch"][..., 0]
        valid = cond["player_valid"].bool()
        bsz, timesteps, agents, _ = position.shape
        if skins.ndim != 6 or skins.shape[:3] != (bsz, agents, 4):
            raise ValueError("player_skin must be [B,A,4,C,H,W]")
        if skins.shape[3] not in (3, 4):
            raise ValueError("player_skin references must be RGB or RGBA")
        view_valid = cond.get("player_appearance_valid")
        if view_valid is None:
            view_valid = torch.ones((bsz, agents, 4), device=position.device, dtype=torch.bool)
        else:
            view_valid = view_valid.to(device=position.device, dtype=torch.bool)
        if view_valid.shape != (bsz, agents, 4):
            raise ValueError("player_appearance_valid must be [B,A,4]")

        dtype = position.dtype
        skins = skins.to(device=position.device, dtype=dtype)
        if skins.shape[3] == 3:
            alpha = torch.ones((*skins.shape[:3], 1, *skins.shape[-2:]), device=skins.device, dtype=dtype)
            skins = torch.cat((skins, alpha), dim=3)

        camera_position = self._select_agent(cond["camera_position"], target)
        camera_direction = self._select_agent(cond["camera_direction"], target)
        fov_x = self._select_agent(cond["camera"], target)[..., -1]
        target_valid = self._select_agent(valid, target)
        camera_forward = F.normalize(camera_direction, dim=-1, eps=1e-6)
        world_down = torch.tensor((0.0, 0.0, -1.0), device=position.device, dtype=dtype)
        camera_right = F.normalize(
            torch.cross(world_down.expand_as(camera_forward), camera_forward, dim=-1),
            dim=-1,
            eps=1e-6,
        )
        camera_down = F.normalize(
            torch.cross(camera_forward, camera_right, dim=-1), dim=-1, eps=1e-6
        )

        feet = position
        head = position.clone()
        head[..., 2] += self.player_height
        tan_half_fov_x = torch.tan(fov_x.clamp(0.05, 3.0) / 2)[..., None]
        foot_u, foot_v, foot_depth = self._project(
            feet,
            camera_position[:, :, None],
            camera_forward[:, :, None],
            camera_right[:, :, None],
            camera_down[:, :, None],
            tan_half_fov_x,
            self.height,
            self.width,
        )
        head_u, head_v, head_depth = self._project(
            head,
            camera_position[:, :, None],
            camera_forward[:, :, None],
            camera_right[:, :, None],
            camera_down[:, :, None],
            tan_half_fov_x,
            self.height,
            self.width,
        )
        center_u, center_v = (foot_u + head_u) / 2, (foot_v + head_v) / 2
        box_h = (foot_v - head_v).abs().clamp(1.0, float(self.height))
        box_w = (box_h * self.player_aspect).clamp(1.0, float(self.width))
        depth = (foot_depth + head_depth) / 2

        # Minetest yaw=0 faces +Y.  The view names are ordered as
        # front/back/left/right in the dataset.
        source_forward = torch.stack((yaw.sin(), yaw.cos()), dim=-1)
        source_right = torch.stack((source_forward[..., 1], -source_forward[..., 0]), dim=-1)
        to_camera = F.normalize(camera_position[:, :, None, :2] - position[..., :2], dim=-1, eps=1e-6)
        front_score = (to_camera * source_forward).sum(dim=-1)
        right_score = (to_camera * source_right).sum(dim=-1)
        view_scores = torch.stack(
            (front_score, -front_score, -right_score, right_score), dim=-1
        ) * self.view_sharpness
        allowed_views = view_valid[:, None].expand(-1, timesteps, -1, -1)
        view_scores = view_scores.masked_fill(~allowed_views, -1e4)
        view_weights = view_scores.softmax(dim=-1)
        selected_skin = torch.einsum("btav,bavchw->btachw", view_weights, skins)

        grid_y = torch.arange(self.height, device=position.device, dtype=dtype)
        grid_x = torch.arange(self.width, device=position.device, dtype=dtype)
        local_y = (
            grid_y.view(1, 1, 1, self.height, 1) - center_v[..., None, None]
        ) / (box_h[..., None, None] / 2)
        local_x = (
            grid_x.view(1, 1, 1, 1, self.width) - center_u[..., None, None]
        ) / (box_w[..., None, None] / 2)
        grid = torch.stack(
            (local_x.expand(-1, -1, -1, self.height, -1),
             local_y.expand(-1, -1, -1, -1, self.width)),
            dim=-1,
        )
        sampled = F.grid_sample(
            selected_skin.flatten(0, 2),
            grid.flatten(0, 2),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ).unflatten(0, (bsz, timesteps, agents))

        source_visible = valid & target_valid[..., None] & (depth > 0.05)
        source_slots = torch.arange(agents, device=position.device)[None, None]
        source_visible &= source_slots != target[:, None, None]
        alpha = sampled[..., 3:4, :, :] * source_visible[..., None, None, None].to(dtype)
        occupancy = alpha.amax(dim=2)
        logits = -self.depth_sharpness * depth[..., None, None] + alpha[:, :, :, 0].clamp_min(1e-6).log()
        logits = logits.masked_fill(alpha[:, :, :, 0] <= 1e-5, -1e4)
        composite_weights = logits.softmax(dim=2) * occupancy
        unpremultiplied_rgb = sampled[..., :3, :, :] / sampled[..., 3:4, :, :].clamp_min(1e-6)
        rgb = (unpremultiplied_rgb * composite_weights[..., None, :, :]).sum(dim=2)
        inverse_depth = (
            composite_weights * (1 / (1 + depth.clamp_min(0)))[..., None, None]
        ).sum(dim=2, keepdim=True)
        uv = grid.permute(0, 1, 2, 5, 3, 4)
        uv = (uv * composite_weights[..., None, :, :]).sum(dim=2)
        selected_view = (
            view_weights[..., :, None, None] * composite_weights[..., None, :, :]
        ).sum(dim=2)
        return torch.cat((rgb, occupancy, inverse_depth, uv, selected_view), dim=2).contiguous()


class PlayerReferenceLayout(nn.Module):
    """Return per-resident coarse ROIs and camera-relative view weights.

    The ROI is deliberately only a routing prior. It is wider than the legacy
    billboard so reference attention can learn limbs and correct projection
    errors instead of receiving a falsely precise target silhouette.
    """

    def __init__(self, height=36, width=64, player_height=1.8, player_aspect=.55,
                 edge_sharpness=1.5, view_sharpness=8.0):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.player_height, self.player_aspect = float(player_height), float(player_aspect)
        self.edge_sharpness, self.view_sharpness = float(edge_sharpness), float(view_sharpness)

    def forward(self, cond: dict, target: torch.Tensor):
        position, yaw = cond["player_position"], cond["yaw_pitch"][..., 0]
        valid = cond["player_valid"].bool()
        bsz, timesteps, agents, _ = position.shape
        camera_position = ViewAwarePlayerAppearance._select_agent(cond["camera_position"], target)
        camera_direction = ViewAwarePlayerAppearance._select_agent(cond["camera_direction"], target)
        fov_x = ViewAwarePlayerAppearance._select_agent(cond["camera"], target)[..., -1]
        target_valid = ViewAwarePlayerAppearance._select_agent(valid, target)
        dtype = position.dtype
        forward = F.normalize(camera_direction, dim=-1, eps=1e-6)
        world_down = torch.tensor((0., 0., -1.), device=position.device, dtype=dtype)
        right = F.normalize(torch.cross(world_down.expand_as(forward), forward, dim=-1), dim=-1, eps=1e-6)
        down = F.normalize(torch.cross(forward, right, dim=-1), dim=-1, eps=1e-6)
        feet, head = position, position.clone()
        head[..., 2] += self.player_height
        tan_x = torch.tan(fov_x.clamp(.05, 3.) / 2)[..., None]
        foot_u, foot_v, foot_depth = ViewAwarePlayerAppearance._project(
            feet, camera_position[:, :, None], forward[:, :, None], right[:, :, None],
            down[:, :, None], tan_x, self.height, self.width)
        head_u, head_v, head_depth = ViewAwarePlayerAppearance._project(
            head, camera_position[:, :, None], forward[:, :, None], right[:, :, None],
            down[:, :, None], tan_x, self.height, self.width)
        center_u, center_v = (foot_u + head_u) / 2, (foot_v + head_v) / 2
        box_h = (foot_v - head_v).abs().clamp(1., float(self.height))
        box_w = (box_h * self.player_aspect).clamp(1., float(self.width))
        grid_y = torch.arange(self.height, device=position.device, dtype=dtype)
        grid_x = torch.arange(self.width, device=position.device, dtype=dtype)
        dy = (grid_y.view(1,1,1,self.height,1) - center_v[...,None,None]).abs()
        dx = (grid_x.view(1,1,1,1,self.width) - center_u[...,None,None]).abs()
        roi = torch.sigmoid((box_h[...,None,None]/2-dy)*self.edge_sharpness)
        roi = roi * torch.sigmoid((box_w[...,None,None]/2-dx)*self.edge_sharpness)
        visible = valid & target_valid[...,None] & (((foot_depth + head_depth) / 2) > .05)
        slots = torch.arange(agents, device=position.device)[None,None]
        visible &= slots != target[:,None,None]
        roi *= visible[...,None,None].to(dtype)

        source_forward = torch.stack((yaw.sin(), yaw.cos()), dim=-1)
        source_right = torch.stack((source_forward[...,1], -source_forward[...,0]), dim=-1)
        to_camera = F.normalize(camera_position[:,:,None,:2]-position[...,:2], dim=-1, eps=1e-6)
        front = (to_camera * source_forward).sum(-1)
        right_score = (to_camera * source_right).sum(-1)
        view_weights = torch.stack((front, -front, -right_score, right_score), -1)
        view_weights = (view_weights * self.view_sharpness).softmax(-1)
        view_weights *= visible[...,None].to(dtype)
        return roi.contiguous(), view_weights.contiguous()


class PlayerReferenceEncoder(nn.Module):
    """Encode native RGBA views into spatial tokens without square resizing."""

    def __init__(self, output_dim=256, grid_size=(8, 4)):
        super().__init__()
        self.output_dim, self.grid_size = int(output_dim), tuple(grid_size)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(128, output_dim, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(self.grid_size),
        )
        self.view_embedding = nn.Parameter(torch.randn(4, output_dim) * .02)

    def forward(self, reference, valid=None):
        if reference.ndim != 6 or reference.shape[2:4] != (4, 4):
            raise ValueError("player_reference must be [B,A,4,4,H,W]")
        bsz, agents, views = reference.shape[:3]
        encoded = self.encoder(reference.flatten(0, 2))
        encoded = encoded.flatten(2).transpose(1, 2).unflatten(0, (bsz, agents, views))
        encoded = encoded + self.view_embedding[None,None,:,None].to(encoded.dtype)
        if valid is not None:
            encoded *= valid.to(encoded.dtype)[...,None,None]
        return encoded
