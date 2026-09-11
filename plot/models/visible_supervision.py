"""Deduplicated discrete GT voxel supervision from sampled camera rays."""

import torch
from .projection import _sample_rays


@torch.no_grad()
def visible_voxel_masks(
    target,
    valid,
    positions,
    directions,
    camera_valid,
    fov_x,
    fov_y,
    air_class,
    height=60,
    width=106,
    samples=384,
    max_distance=32.0,
    return_ray_weights=False,
):
    surface = torch.zeros_like(valid, dtype=torch.bool)
    free = torch.zeros_like(valid, dtype=torch.bool)
    ray_weights = torch.zeros_like(target, dtype=torch.float32) if return_ray_weights else None
    size = target.shape[-1]
    ids = torch.arange(size**3, device=target.device).float().reshape(1, size, size, size)
    for batch in range(target.shape[0]):
        for view in range(positions.shape[1]):
            if not camera_valid[batch, view]:
                continue
            pose = (
                positions[batch : batch + 1, view].float(),
                directions[batch : batch + 1, view].float(),
                fov_x[batch : batch + 1, view].float(),
                fov_y[batch : batch + 1, view].float(),
            )
            kw = dict(
                height=height,
                width=width,
                samples=samples,
                max_distance=max_distance,
                mode="nearest",
            )
            occ, _ = _sample_rays(
                ((target[batch : batch + 1] != air_class) & valid[batch : batch + 1]).float(),
                *pose,
                **kw,
            )
            validity, _ = _sample_rays(valid[batch : batch + 1].float(), *pose, **kw)
            index, _ = _sample_rays(ids, *pose, **kw)
            # Do not infer free space or a visible surface through unknown/outside cells.
            prefix = (validity > 0.5).long().cumprod(1).bool()
            prefix[:, :, round(height * 0.82) :] = False
            hit = (occ > 0.5) & prefix
            first = hit.long().argmax(1)
            has_hit = hit.any(1)
            order = torch.arange(samples, device=target.device)[None, :, None, None]
            surf = hit & (order == first[:, None])
            empty = prefix & (~has_hit[:, None] | (order < first[:, None])) & (occ < 0.5)
            surface[batch].view(-1)[index[surf].long()] = True
            free[batch].view(-1)[index[empty].long()] = True
            if ray_weights is not None:
                # Each valid free-space ray contributes unit total weight.
                # Repeated samples within a voxel retain their path-length weight.
                weights = empty.float() / empty.sum(1, keepdim=True).clamp_min(1)
                ray_weights[batch].view(-1).scatter_add_(
                    0, index[empty].long(), weights[empty]
                )
    assert not (surface & free).any()
    assert ((target != air_class) | ~surface).all()
    assert ((target == air_class) | ~free).all()
    if return_ray_weights:
        return surface, free, ray_weights
    return surface, free
