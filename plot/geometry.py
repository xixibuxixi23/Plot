from __future__ import annotations

import numpy as np


RAW_OBSERVATION_SIZE = 49
TILE_SIZE = 48
TILE_ORIGIN_INDEX = 24


def tile_bounds(center: np.ndarray | tuple[int, int, int], size: int = TILE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    """Return half-open global bounds for a PERSIST-compatible voxel tile.

    For the production size this is exactly ``[center - 24, center + 24)``.
    Even test sizes use the same lower-heavy half-open convention.
    """
    if size <= 0 or size % 2:
        raise ValueError("tile size must be a positive even integer")
    center_array = np.asarray(center, dtype=np.int64)
    if center_array.shape != (3,):
        raise ValueError("center must have shape (3,)")
    lower = center_array - size // 2
    return lower, lower + size


def tile_global_coordinates(
    center: np.ndarray | tuple[int, int, int], size: int = TILE_SIZE
) -> np.ndarray:
    """Return an ``[size,size,size,3]`` array of integer global coordinates."""
    lower, upper = tile_bounds(center, size)
    axes = [np.arange(lower[i], upper[i], dtype=np.int64) for i in range(3)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def crop_raw_49_to_tile_48(voxels: np.ndarray) -> np.ndarray:
    """Apply PERSIST's right-edge crop while retaining origin index 24."""
    if voxels.shape[-4:-1] != (49, 49, 49):
        raise ValueError(f"expected trailing spatial shape (49,49,49), got {voxels.shape}")
    return voxels[..., :48, :48, :48, :]


def coverage_mask(
    query_center: np.ndarray,
    previous_centers: np.ndarray,
    size: int = TILE_SIZE,
) -> np.ndarray:
    """Mark query voxels covered by any previous resident tile."""
    centers = np.asarray(previous_centers, dtype=np.int64).reshape(-1, 3)
    if centers.size == 0:
        return np.zeros((size, size, size), dtype=bool)
    # Build the union as intersections of axis-aligned boxes. Broadcasting all
    # trajectory centers against 48^3 queries would require gigabytes for a
    # normal episode.
    query_lower, query_upper = tile_bounds(query_center, size)
    result = np.zeros((size, size, size), dtype=bool)
    for center in np.unique(centers, axis=0):
        previous_lower, previous_upper = tile_bounds(center, size)
        lower = np.maximum(query_lower, previous_lower)
        upper = np.minimum(query_upper, previous_upper)
        if np.any(lower >= upper):
            continue
        local_lower = lower - query_lower
        local_upper = upper - query_lower
        result[
            local_lower[0]:local_upper[0],
            local_lower[1]:local_upper[1],
            local_lower[2]:local_upper[2],
        ] = True
    return result
