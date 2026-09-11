from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def camera_rays(
    direction: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Return ENU ray directions as [B,H,W,3], with image row zero at camera up."""
    dtype, device = direction.dtype, direction.device
    forward = F.normalize(direction, dim=-1, eps=1e-6)
    world_up = torch.tensor((0.0, 0.0, 1.0), dtype=dtype, device=device).expand_as(forward)
    right = torch.linalg.cross(forward, world_up, dim=-1)
    alternate_up = torch.tensor((0.0, 1.0, 0.0), dtype=dtype, device=device).expand_as(forward)
    alternate_right = torch.linalg.cross(forward, alternate_up, dim=-1)
    right = torch.where(
        (right.square().sum(-1, keepdim=True) < 1e-8), alternate_right, right
    )
    right = F.normalize(right, dim=-1, eps=1e-6)
    camera_up = F.normalize(torch.linalg.cross(right, forward, dim=-1), dim=-1, eps=1e-6)

    x = (torch.arange(width, dtype=dtype, device=device) + 0.5) * (2.0 / width) - 1.0
    y = 1.0 - (torch.arange(height, dtype=dtype, device=device) + 0.5) * (2.0 / height)
    x = x[None, None, :, None] * torch.tan(fov_x[:, None, None, None] * 0.5)
    y = y[None, :, None, None] * torch.tan(fov_y[:, None, None, None] * 0.5)
    rays = forward[:, None, None] + x * right[:, None, None] + y * camera_up[:, None, None]
    return F.normalize(rays, dim=-1, eps=1e-6)


def _sample_rays(
    volume: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    height: int,
    width: int,
    samples: int,
    max_distance: float,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample [B,X,Y,Z] volumes; positions are cube-centred and normalized by X."""
    if volume.ndim != 4 or len(set(volume.shape[-3:])) != 1:
        raise ValueError("projection volume must have shape [B,S,S,S]")
    size = volume.shape[-1]
    rays = camera_rays(camera_direction, fov_x, fov_y, height, width)
    distances = torch.linspace(
        max_distance / samples,
        max_distance,
        samples,
        dtype=volume.dtype,
        device=volume.device,
    )
    origin = camera_position * float(size) + (size - 1) * 0.5
    points = origin[:, None, None, None] + (
        rays[:, None] * distances[None, :, None, None, None]
    )
    # grid_sample expects input [B,C,Z,Y,X] and grid coordinates [x,y,z].
    grid = 2.0 * (points + 0.5) / float(size) - 1.0
    grid = grid.permute(0, 1, 2, 3, 4)
    source = volume[:, None].permute(0, 1, 4, 3, 2)
    sampled = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )[:, 0]
    return sampled, distances


def render_occupancy_view(
    occupancy: torch.Tensor,
    target_occupancy: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    height: int,
    width: int,
    samples: int,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted, distances = _sample_rays(
        occupancy,
        camera_position,
        camera_direction,
        fov_x,
        fov_y,
        height=height,
        width=width,
        samples=samples,
        max_distance=max_distance,
        mode="bilinear",
    )
    target_silhouette, target_depth, _ = raycast_voxel_targets(
        target_occupancy, camera_position, camera_direction, fov_x, fov_y,
        height=height, width=width, samples=samples, max_distance=max_distance,
    )
    alpha = predicted.clamp(0.0, 1.0)
    transmittance = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-6), dim=1), dim=1
    )[:, :-1]
    weights = alpha * transmittance
    silhouette = weights.sum(1).clamp(0.0, 1.0)
    depth = (weights * distances[None, :, None, None]).sum(1) / silhouette.clamp_min(1e-6)

    return silhouette, depth, target_silhouette, target_depth


def raycast_voxel_targets(
    target_occupancy: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    height: int,
    width: int,
    samples: int,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard first-hit silhouette, ray distance, and world-axis ray directions."""
    target, distances = _sample_rays(
        target_occupancy,
        camera_position,
        camera_direction,
        fov_x,
        fov_y,
        height=height,
        width=width,
        samples=samples,
        max_distance=max_distance,
        mode="nearest",
    )
    target_hits = target > 0.5
    silhouette = target_hits.any(1).to(target_occupancy.dtype)
    first_hit = target_hits.to(torch.int64).argmax(1)
    depth = distances[first_hit] * silhouette
    rays = camera_rays(
        camera_direction, fov_x, fov_y, height, width
    )
    return silhouette, depth, rays


def raycast_voxel_semantic_targets(
    target: torch.Tensor,
    target_valid: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    air_class: int,
    height: int,
    width: int,
    samples: int,
    max_distance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return hard first-hit visibility, depth, ray and discrete block class.

    Class zero is a valid material in the learned vocabulary, so misses are
    represented by ``air_class`` and must always be masked by visibility.
    """
    sampled_class, distances = _sample_rays(
        target.float(), camera_position, camera_direction, fov_x, fov_y,
        height=height, width=width, samples=samples,
        max_distance=max_distance, mode="nearest",
    )
    sampled_valid, _ = _sample_rays(
        target_valid.float(), camera_position, camera_direction, fov_x, fov_y,
        height=height, width=width, samples=samples,
        max_distance=max_distance, mode="nearest",
    )
    hit = (sampled_valid > 0.5) & (sampled_class.round().long() != air_class)
    visibility = hit.any(1)
    first_hit = hit.long().argmax(1)
    depth = distances[first_hit] * visibility.to(distances.dtype)
    semantic_class = sampled_class.gather(1, first_hit[:, None]).squeeze(1).round().long()
    semantic_class = torch.where(
        visibility, semantic_class, torch.full_like(semantic_class, air_class)
    )
    rays = camera_rays(camera_direction, fov_x, fov_y, height, width)
    return visibility.to(torch.float32), depth, rays, semantic_class


def projection_consistency_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    fill_mask: torch.Tensor,
    target_valid: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    camera_valid: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    air_class: int,
    height: int = 24,
    width: int = 40,
    samples: int = 64,
    max_distance: float = 32.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera-space silhouette and first-hit depth losses for a voxel classifier.

    Ground-truth poses are deliberately used on both sides. This makes the loss
    constrain voxel geometry without allowing camera and geometry errors to cancel.
    """
    if logits.ndim != 5 or target.shape != logits.shape[:1] + logits.shape[-3:]:
        raise ValueError("expected logits [B,C,S,S,S] and target [B,S,S,S]")
    if not 0 <= air_class < logits.shape[1]:
        raise ValueError("air_class is outside the classifier vocabulary")
    if height < 1 or width < 1 or samples < 2 or max_distance <= 0:
        raise ValueError("projection dimensions, samples and max_distance must be positive")

    # Avoid materializing a full [B,C,S,S,S] softmax. logsumexp has the same
    # gradient but keeps only the scalar air probability used by ray marching.
    logits_float = logits.float()
    log_air = logits_float[:, air_class] - torch.logsumexp(logits_float, dim=1)
    predicted_occupancy = 1.0 - log_air.exp()
    valid = target_valid.bool()
    target_occupancy = ((target != air_class) & valid).to(predicted_occupancy.dtype)
    generated = fill_mask.bool() & valid
    predicted_occupancy = torch.where(generated, predicted_occupancy, target_occupancy)

    silhouettes, depths, target_silhouettes, target_depths = [], [], [], []
    views = camera_position.shape[1]
    for view in range(views):
        rendered = render_occupancy_view(
            predicted_occupancy,
            target_occupancy,
            camera_position[:, view].float(),
            camera_direction[:, view].float(),
            fov_x[:, view].float(),
            fov_y[:, view].float(),
            height=height,
            width=width,
            samples=samples,
            max_distance=max_distance,
        )
        for values, value in zip(
            (silhouettes, depths, target_silhouettes, target_depths), rendered
        ):
            values.append(value)
    silhouette = torch.stack(silhouettes, dim=1)
    depth = torch.stack(depths, dim=1)
    target_silhouette = torch.stack(target_silhouettes, dim=1)
    target_depth = torch.stack(target_depths, dim=1)
    view_mask = camera_valid.bool()[:, :, None, None].expand_as(target_silhouette)
    # Probability-form BCE is intentionally expanded here: torch rejects
    # F.binary_cross_entropy inside an autocast region even for float32 inputs.
    silhouette = silhouette.float().clamp(1e-5, 1.0 - 1e-5)
    target_silhouette = target_silhouette.float()
    silhouette_loss = -(
        target_silhouette * silhouette.log()
        + (1.0 - target_silhouette) * torch.log1p(-silhouette)
    )
    silhouette_loss = (silhouette_loss * view_mask).sum() / view_mask.sum().clamp_min(1)
    depth_mask = view_mask & target_silhouette.bool()
    depth_loss = F.smooth_l1_loss(
        depth / max_distance, target_depth / max_distance, reduction="none"
    )
    depth_loss = (depth_loss * depth_mask).sum() / depth_mask.sum().clamp_min(1)
    if not math.isfinite(max_distance):
        raise ValueError("max_distance must be finite")
    return silhouette_loss, depth_loss


def first_hit_projection_loss(
    occupancy_logits: torch.Tensor,
    target: torch.Tensor,
    fill_mask: torch.Tensor,
    target_valid: torch.Tensor,
    camera_position: torch.Tensor,
    camera_direction: torch.Tensor,
    camera_valid: torch.Tensor,
    fov_x: torch.Tensor,
    fov_y: torch.Tensor,
    *,
    air_class: int,
    height: int = 24,
    width: int = 40,
    samples: int = 64,
    max_distance: float = 32.0,
    supervised_height_fraction: float = 0.82,
    edge_weight: float = 3.0,
    alpha_reference_step: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Supervise visible geometry as free space followed by its first surface.

    Ground-truth cameras are used on both sides so pose errors cannot reduce the
    geometry loss. Invalid portions of an irregular target volume are ignored.
    """
    if occupancy_logits.ndim != 4 or target.shape != occupancy_logits.shape:
        raise ValueError("expected occupancy logits and target with shape [B,S,S,S]")
    predicted_occupancy = occupancy_logits.float().sigmoid()
    valid_volume = target_valid.bool()
    target_occupancy = ((target != air_class) & valid_volume).to(predicted_occupancy.dtype)
    generated = fill_mask.bool() & valid_volume
    predicted_occupancy = torch.where(generated, predicted_occupancy, target_occupancy)

    totals = {
        "projection_surface_loss": predicted_occupancy.new_zeros(()),
        "projection_free_space_loss": predicted_occupancy.new_zeros(()),
        "projection_silhouette_loss": predicted_occupancy.new_zeros(()),
        "projection_depth_loss": predicted_occupancy.new_zeros(()),
    }
    total_weight = predicted_occupancy.new_zeros(())
    supervised_height = max(1, round(height * supervised_height_fraction))
    sample_indices = torch.arange(samples, device=target.device)[:, None, None]

    for view in range(camera_position.shape[1]):
        pose_args = (
            camera_position[:, view].float(), camera_direction[:, view].float(),
            fov_x[:, view].float(), fov_y[:, view].float(),
        )
        predicted_samples, distances = _sample_rays(
            predicted_occupancy, *pose_args, height=height, width=width,
            samples=samples, max_distance=max_distance, mode="bilinear",
        )
        target_samples, _ = _sample_rays(
            target_occupancy, *pose_args, height=height, width=width,
            samples=samples, max_distance=max_distance, mode="nearest",
        )
        valid_samples, _ = _sample_rays(
            valid_volume.to(predicted_occupancy.dtype), *pose_args,
            height=height, width=width, samples=samples,
            max_distance=max_distance, mode="nearest",
        )
        predicted_samples = predicted_samples[:, :, :supervised_height].clamp(1e-5, 1 - 1e-5)
        target_hits = (
            (target_samples[:, :, :supervised_height] > 0.5)
            & (valid_samples[:, :, :supervised_height] > 0.5)
        )
        valid_samples = valid_samples[:, :, :supervised_height] > 0.5
        has_hit = target_hits.any(1)
        first_index = target_hits.to(torch.int64).argmax(1)
        surface_mask = target_hits & (sample_indices == first_index[:, None])
        free_mask = valid_samples & (
            (~has_hit[:, None]) | (sample_indices < first_index[:, None])
        )

        target_depth = distances[first_index] * has_hit
        edge = torch.zeros_like(target_depth)
        edge[:, 1:] = torch.maximum(
            edge[:, 1:], (target_depth[:, 1:] - target_depth[:, :-1]).abs()
        )
        edge[:, :, 1:] = torch.maximum(
            edge[:, :, 1:], (target_depth[:, :, 1:] - target_depth[:, :, :-1]).abs()
        )
        silhouette_float = has_hit.to(predicted_occupancy.dtype)
        edge = ((edge > max_distance / samples) | (
            F.max_pool2d(silhouette_float[:, None], 3, 1, 1)[:, 0]
            != -F.max_pool2d(-silhouette_float[:, None], 3, 1, 1)[:, 0]
        )).to(predicted_occupancy.dtype)
        ray_weight = 1.0 + edge_weight * edge
        active = camera_valid[:, view].bool()[:, None, None]
        ray_weight = ray_weight * active.to(ray_weight.dtype)

        surface_nll = -torch.log(predicted_samples)
        free_nll = -torch.log1p(-predicted_samples)
        surface_per_ray = (surface_nll * surface_mask).sum(1)
        free_per_ray = (free_nll * free_mask).sum(1) / free_mask.sum(1).clamp_min(1)

        # Experimental optical interpretation, relative to the original 0.5-block
        # segment. Zero preserves the historical per-sample alpha exactly.
        alpha = predicted_samples
        if alpha_reference_step > 0:
            alpha = -torch.expm1(
                torch.log1p(-predicted_samples)
                * (max_distance / samples / alpha_reference_step)
            )
        transmittance = torch.cumprod(
            torch.cat((torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-6), dim=1), dim=1
        )[:, :-1]
        first_hit_probability = alpha * transmittance
        predicted_silhouette = first_hit_probability.sum(1).clamp(1e-5, 1 - 1e-5)
        predicted_depth = (
            first_hit_probability * distances[None, :, None, None]
        ).sum(1) / predicted_silhouette.clamp_min(1e-5)
        silhouette_loss = -(
            silhouette_float * predicted_silhouette.log()
            + (1.0 - silhouette_float) * torch.log1p(-predicted_silhouette)
        )
        depth_loss = F.smooth_l1_loss(
            predicted_depth / max_distance, target_depth / max_distance, reduction="none"
        ) * has_hit

        ray_denominator = ray_weight.sum().clamp_min(1)
        hit_weight = ray_weight * has_hit
        totals["projection_surface_loss"] += (
            surface_per_ray * hit_weight
        ).sum() / hit_weight.sum().clamp_min(1)
        totals["projection_free_space_loss"] += (
            free_per_ray * ray_weight
        ).sum() / ray_denominator
        totals["projection_silhouette_loss"] += (
            silhouette_loss * ray_weight
        ).sum() / ray_denominator
        totals["projection_depth_loss"] += (
            depth_loss * hit_weight
        ).sum() / hit_weight.sum().clamp_min(1)
        total_weight += active.any().to(total_weight.dtype)

    denominator = total_weight.clamp_min(1)
    return {name: value / denominator for name, value in totals.items()}
