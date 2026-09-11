from __future__ import annotations

import colorsys
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAMERA_COLORS = ("#ef5350", "#26a69a", "#7e57c2", "#ffa726")


def _surface(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    interior = padded[1:-1, 1:-1, 1:-1].copy()
    for slc in (
        padded[:-2, 1:-1, 1:-1], padded[2:, 1:-1, 1:-1],
        padded[1:-1, :-2, 1:-1], padded[1:-1, 2:, 1:-1],
        padded[1:-1, 1:-1, :-2], padded[1:-1, 1:-1, 2:],
    ):
        interior &= slc
    return mask & ~interior


def _color(token: int) -> tuple[float, float, float, float]:
    digest = hashlib.sha1(str(token).encode()).digest()
    rgb = colorsys.hsv_to_rgb(digest[0] / 255, 0.35 + digest[1] / 900, 0.55 + digest[2] / 700)
    return (*rgb, 0.92)


def _plot_xyz(value: np.ndarray) -> np.ndarray:
    """Collector output is already ENU: east, north, up."""
    return np.asarray(value)


def draw_camera_frustum(ax, position, direction, color, label, depth=7.0, fov_deg=70.0):
    position = np.asarray(position, dtype=float)
    forward = np.asarray(direction, dtype=float)
    forward /= max(np.linalg.norm(forward), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-5:
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
    right /= max(np.linalg.norm(right), 1e-8)
    up = np.cross(right, forward)
    half = np.tan(np.deg2rad(fov_deg) / 2) * depth * 0.55
    center = position + forward * depth
    corners = [center + sx * half * right + sy * half * up for sx, sy in ((-1,-1),(-1,1),(1,1),(1,-1))]
    p = _plot_xyz(position)
    corners_plot = [_plot_xyz(c) for c in corners]
    for corner in corners_plot:
        ax.plot(*zip(p, corner), color=color, linewidth=1.8, alpha=0.9)
    loop = corners_plot + [corners_plot[0]]
    ax.plot([x[0] for x in loop], [x[1] for x in loop], [x[2] for x in loop],
            color=color, linewidth=2.0)
    end = _plot_xyz(position + forward * depth * 1.15)
    ax.plot(*zip(p, end), color=color, linewidth=3.0)
    ax.scatter(*p, s=70, color=color, edgecolor="white", depthshade=False)
    ax.text(*p, label, color=color, fontsize=9, weight="bold")


def render_voxel_cameras(
    voxels: np.ndarray,
    camera_position: np.ndarray,
    camera_direction: np.ndarray,
    output_path: str | Path,
    *,
    air_class: int = 0,
    title: str = "M1 initialization",
    normalized_camera: bool = True,
    elevation: float = 28.0,
    azimuth: float = -52.0,
    render_stride: int = 1,
):
    """Render one 48^3 tile and camera poses as view frustums."""
    voxels = np.asarray(voxels)
    render_stride = int(render_stride)
    if render_stride < 1:
        raise ValueError("render_stride must be positive")
    if render_stride > 1:
        voxels = voxels[::render_stride, ::render_stride, ::render_stride]
    occupied = voxels != air_class
    visible = _surface(occupied)
    colors = np.zeros(voxels.shape + (4,), dtype=np.float32)
    for token in np.unique(voxels[visible]):
        colors[voxels == token] = _color(int(token))
    visible_plot = visible
    colors_plot = colors
    size = np.asarray(voxels.shape, dtype=float)
    positions = np.asarray(camera_position, dtype=float)
    if normalized_camera:
        positions = (positions * 48.0 + 23.5) / render_stride

    fig = plt.figure(figsize=(9, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    if visible_plot.any():
        ax.voxels(visible_plot, facecolors=colors_plot, edgecolors=(0, 0, 0, 0.12), linewidth=0.08)
    for index, (position, direction) in enumerate(zip(positions, camera_direction)):
        draw_camera_frustum(
            ax, position, direction, CAMERA_COLORS[index % len(CAMERA_COLORS)], f"camera {index}",
            depth=7.0 / render_stride,
        )
    ax.set_xlim(0, size[0])
    ax.set_ylim(0, size[1])
    ax.set_zlim(0, size[2])
    ax.set_box_aspect((size[0], size[1], size[2] * 0.75))
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_axis_off()
    ax.set_title(title, pad=4)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout(pad=0.2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output_path


def render_initialization_comparison(target, prediction, batch, output_path):
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    base = Path(output_path)
    gt_path = base.with_name(base.stem + "_gt" + base.suffix)
    pred_path = base.with_name(base.stem + "_pred" + base.suffix)
    render_voxel_cameras(
        target, np.asarray(batch["camera_position"]), np.asarray(batch["camera_direction"]),
        gt_path, title="Ground truth voxel + cameras",
    )
    render_voxel_cameras(
        prediction, np.asarray(batch["pred_camera_position"]),
        np.asarray(batch["pred_camera_direction"]), pred_path,
        title="M1 predicted voxel + cameras",
    )
    return gt_path, pred_path
