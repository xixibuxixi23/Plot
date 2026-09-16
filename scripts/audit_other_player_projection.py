"""Compare M3's pose-based other-player projection with recorded observations.

This diagnostic intentionally evaluates the deterministic geometry before the
renderer learns anything.  It reports visibility, bounding-box placement, and
silhouette overlap using the native four-view RGBA references.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


VIEW_NAMES = ("front", "back", "left", "right")


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def project_player_box(
    player_position: np.ndarray,
    camera_position: np.ndarray,
    camera_direction: np.ndarray,
    fov_x: float,
    *,
    height: int,
    width: int,
    player_height: float = 1.8,
    player_aspect: float = 0.45,
) -> tuple[np.ndarray, float]:
    """Reproduce ``ViewAwarePlayerAppearance`` at image resolution."""
    forward = _normalize(np.asarray(camera_direction, dtype=np.float64))
    world_down = np.asarray((0.0, 0.0, -1.0))
    right = _normalize(np.cross(world_down, forward))
    down = _normalize(np.cross(forward, right))
    feet = np.asarray(player_position, dtype=np.float64)
    head = feet + np.asarray((0.0, 0.0, player_height))
    tan_x = np.tan(np.clip(float(fov_x), 0.05, 3.0) / 2.0)
    tan_y = tan_x * height / width

    def project(point: np.ndarray) -> tuple[float, float, float]:
        relative = point - camera_position
        depth = float(relative @ forward)
        safe_depth = max(depth, 1e-4)
        u = (float(relative @ right) / safe_depth / tan_x + 1.0) * (width - 1) / 2.0
        v = (float(relative @ down) / safe_depth / tan_y + 1.0) * (height - 1) / 2.0
        return u, v, depth

    foot_u, foot_v, foot_depth = project(feet)
    head_u, head_v, head_depth = project(head)
    center_u, center_v = (foot_u + head_u) / 2.0, (foot_v + head_v) / 2.0
    box_height = np.clip(abs(foot_v - head_v), 1.0, float(height))
    box_width = np.clip(box_height * player_aspect, 1.0, float(width))
    box = np.asarray(
        (
            center_u - box_width / 2.0,
            center_v - box_height / 2.0,
            center_u + box_width / 2.0,
            center_v + box_height / 2.0,
        ),
        dtype=np.float64,
    )
    return box, (foot_depth + head_depth) / 2.0


def project_player_prism_box(
    player_position: np.ndarray,
    player_yaw: float,
    camera_position: np.ndarray,
    camera_direction: np.ndarray,
    fov_x: float,
    *,
    height: int,
    width: int,
    player_height: float = 1.8,
    player_half_width: float = 0.3125,
) -> tuple[np.ndarray, float]:
    """Project a fixed yaw-oriented player prism using the same camera model."""
    forward = _normalize(np.asarray(camera_direction, dtype=np.float64))
    world_down = np.asarray((0.0, 0.0, -1.0))
    right = _normalize(np.cross(world_down, forward))
    down = _normalize(np.cross(forward, right))
    source_forward = np.asarray((np.sin(player_yaw), np.cos(player_yaw), 0.0))
    source_right = np.asarray((source_forward[1], -source_forward[0], 0.0))
    base = np.asarray(player_position, dtype=np.float64)
    corners = []
    for local_forward in (-player_half_width, player_half_width):
        for local_right in (-player_half_width, player_half_width):
            for local_up in (0.0, player_height):
                corners.append(
                    base + local_forward * source_forward + local_right * source_right
                    + np.asarray((0.0, 0.0, local_up))
                )
    corners = np.stack(corners)
    relative = corners - camera_position
    depth = relative @ forward
    keep = depth > 0.05
    if not keep.any():
        return np.asarray((np.inf, np.inf, np.inf, np.inf)), float(depth.mean())
    relative, safe_depth = relative[keep], depth[keep]
    tan_x = np.tan(np.clip(float(fov_x), 0.05, 3.0) / 2.0)
    tan_y = tan_x * height / width
    u = ((relative @ right) / safe_depth / tan_x + 1.0) * (width - 1) / 2.0
    v = ((relative @ down) / safe_depth / tan_y + 1.0) * (height - 1) / 2.0
    return np.asarray((u.min(), v.min(), u.max(), v.max())), float(depth.mean())


def view_weights(
    source_yaw: float,
    source_position: np.ndarray,
    camera_position: np.ndarray,
    sharpness: float = 8.0,
) -> np.ndarray:
    """Reproduce M3's front/back/left/right soft view choice."""
    source_forward = np.asarray((np.sin(source_yaw), np.cos(source_yaw)))
    source_right = np.asarray((source_forward[1], -source_forward[0]))
    to_camera = _normalize(camera_position[:2] - source_position[:2])
    front = float(to_camera @ source_forward)
    right = float(to_camera @ source_right)
    logits = sharpness * np.asarray((front, -front, -right, right))
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def _clip_box(box: np.ndarray, height: int, width: int) -> np.ndarray:
    if not np.isfinite(box).all():
        return np.zeros(4, dtype=np.float64)
    clipped = box.copy()
    clipped[[0, 2]] = np.clip(clipped[[0, 2]], 0, width)
    clipped[[1, 3]] = np.clip(clipped[[1, 3]], 0, height)
    return clipped


def _box_mask(box: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if not np.isfinite(box).all():
        return mask
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    x0, x1 = np.clip((x0, x1), 0, width)
    y0, y1 = np.clip((y0, y1), 0, height)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection_size = np.maximum(
        0.0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])
    )
    intersection = float(np.prod(intersection_size))
    first_area = float(np.prod(np.maximum(first[2:] - first[:2], 0.0)))
    second_area = float(np.prod(np.maximum(second[2:] - second[:2], 0.0)))
    return intersection / max(first_area + second_area - intersection, 1e-6)


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.logical_and(first, second).sum() / max(np.logical_or(first, second).sum(), 1))


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Approximate the body when a render ID also covers a detached held item."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def _load_references(episode: Path, source: int) -> np.ndarray:
    views = []
    for name in VIEW_NAMES:
        image = cv2.imread(
            str(episode / "players" / f"agent{source}" / f"{name}.png"),
            cv2.IMREAD_UNCHANGED,
        )
        if image is None or image.shape[-1] != 4:
            raise ValueError(f"missing RGBA player reference for agent{source}/{name}")
        views.append(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA).astype(np.float32) / 255.0)
    return np.stack(views)


def warp_reference(
    references: np.ndarray,
    weights: np.ndarray,
    box: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Softly select an RGBA reference and stretch it into M3's billboard."""
    premultiplied = references.copy()
    premultiplied[..., :3] *= premultiplied[..., 3:4]
    selected = np.einsum("v,vhwc->hwc", weights, premultiplied)
    output = np.zeros((height, width, 4), np.float32)
    x0, y0 = np.floor(box[:2]).astype(int)
    x1, y1 = np.ceil(box[2:]).astype(int)
    raw_width, raw_height = x1 - x0, y1 - y0
    if raw_width <= 0 or raw_height <= 0:
        return output, np.zeros((height, width), bool)
    sprite = cv2.resize(selected, (raw_width, raw_height), interpolation=cv2.INTER_LINEAR)
    dst_x0, dst_x1 = max(x0, 0), min(x1, width)
    dst_y0, dst_y1 = max(y0, 0), min(y1, height)
    if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
        return output, np.zeros((height, width), bool)
    src_x0, src_x1 = dst_x0 - x0, dst_x1 - x0
    src_y0, src_y1 = dst_y0 - y0, dst_y1 - y0
    output[dst_y0:dst_y1, dst_x0:dst_x1] = sprite[src_y0:src_y1, src_x0:src_x1]
    return output, output[..., 3] >= 0.05


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p10": None, "median": None, "p90": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
    }


def _read_frame(video: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot decode frame {frame} from {video}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_example(candidate: dict, output: Path) -> dict[str, float]:
    episode = Path(candidate["episode"])
    frame, target, source = candidate["frame"], candidate["target"], candidate["source"]
    with np.load(episode / "data.npz", allow_pickle=False) as data:
        height, width = data["instance_mask"].shape[-2:]
        render_id = data["entity_render_object_id"][frame, source]
        truth = _largest_component(
            (data["instance_mask"][frame, target] == render_id) & (render_id != 0)
        )
        box, _ = project_player_box(
            data["player_pos"][frame, source], data["cam_pos"][frame, target],
            data["cam_dir"][frame, target], data["fov_x"][frame, target],
            height=height, width=width,
        )
        weights = view_weights(
            np.deg2rad(data["player_yaw"][frame, source]), data["player_pos"][frame, source],
            data["cam_pos"][frame, target],
        )
    references = _load_references(episode, source)
    projected, projected_mask = warp_reference(
        references, weights, box, height=height, width=width
    )
    manifest = json.loads((episode / "manifest.json").read_text())
    videos = manifest.get("agent_video_files") or [f"rgb_agent{i}.mp4" for i in range(target + 1)]
    rgb = _read_frame(episode / videos[target], frame).astype(np.float32) / 255.0
    overlap = truth & projected_mask
    unpremultiplied = projected[..., :3] / np.maximum(projected[..., 3:4], 1e-6)
    rgb_l1 = float(np.abs(unpremultiplied[overlap] - rgb[overlap]).mean()) if overlap.any() else float("nan")

    overlay = rgb.copy()
    alpha = projected[..., 3:4] * 0.65
    overlay = overlay * (1 - alpha) + unpremultiplied * alpha
    contour = truth.astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_bgr = cv2.cvtColor(np.clip(overlay * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 0), 2)
    x0, y0, x1, y1 = np.rint(box).astype(int)
    cv2.rectangle(overlay_bgr, (x0, y0), (x1, y1), (255, 128, 0), 2)
    real_bgr = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # A neutral canvas makes the third panel an unambiguous view of the
    # projected reference alone, without pixels from the recorded observation.
    projection_only = np.full_like(rgb, 0.18)
    projection_only = projection_only * (1 - alpha) + unpremultiplied * alpha
    projection_only_bgr = cv2.cvtColor(
        np.clip(projection_only * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    panel = np.concatenate((real_bgr, overlay_bgr, projection_only_bgr), axis=1)
    label = (
        f"left=actual  center=projected overlay  right=projection only | "
        f"target={target} source={source} "
        f"frame={frame} view={VIEW_NAMES[int(weights.argmax())]}"
    )
    cv2.putText(panel, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    cv2.imwrite(str(output), panel)
    return {"rgb_l1_on_overlap": rgb_l1, "overlap_pixels": int(overlap.sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--min-visible-pixels", type=int, default=32)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root, output = Path(args.data_root), Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    episodes = sorted(path.parent for path in root.glob("*/manifest.json"))
    rng = np.random.default_rng(args.seed)
    if len(episodes) > args.max_episodes:
        episodes = [episodes[index] for index in sorted(rng.choice(len(episodes), args.max_episodes, replace=False))]

    rows: list[dict] = []
    visibility = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    prism_visibility = {"true_positive": 0, "false_positive": 0, "false_negative": 0, "true_negative": 0}
    candidates: list[dict] = []
    for episode_number, episode in enumerate(episodes, 1):
        with np.load(episode / "data.npz", allow_pickle=False) as data:
            manifest = json.loads((episode / "manifest.json").read_text())
            scenario = manifest.get("scenario_id", "unknown")
            masks = data["instance_mask"]
            timesteps, agents, height, width = masks.shape
            entity_ids = list(data["entity_id"].astype(str))
            slots = [entity_ids.index(f"agent{agent}") for agent in range(agents)]
            references_by_source = [_load_references(episode, source) for source in range(agents)]
            for frame in range(0, timesteps, args.frame_stride):
                for target in range(agents):
                    target_mask = masks[frame, target]
                    for source in range(agents):
                        if source == target:
                            continue
                        entity_slot = slots[source]
                        render_id = int(data["entity_render_object_id"][frame, entity_slot])
                        full_truth = (target_mask == render_id) & (render_id != 0)
                        full_true_pixels = int(full_truth.sum())
                        truth = _largest_component(full_truth)
                        true_pixels = int(truth.sum())
                        actual_visible = full_true_pixels >= args.min_visible_pixels
                        box, depth = project_player_box(
                            data["player_pos"][frame, source], data["cam_pos"][frame, target],
                            data["cam_dir"][frame, target], data["fov_x"][frame, target],
                            height=height, width=width,
                        )
                        clipped = _clip_box(box, height, width)
                        box_mask = _box_mask(box, height, width)
                        predicted_visible = depth > 0.05 and bool(box_mask.any())
                        yaw_radians = np.deg2rad(data["player_yaw"][frame, source])
                        prism_box, prism_depth = project_player_prism_box(
                            data["player_pos"][frame, source], yaw_radians,
                            data["cam_pos"][frame, target], data["cam_dir"][frame, target],
                            data["fov_x"][frame, target], height=height, width=width,
                        )
                        prism_clipped = _clip_box(prism_box, height, width)
                        prism_mask = _box_mask(prism_box, height, width)
                        prism_visible = prism_depth > 0.05 and bool(prism_mask.any())
                        key = (
                            "true_positive" if actual_visible and predicted_visible else
                            "false_positive" if predicted_visible else
                            "false_negative" if actual_visible else "true_negative"
                        )
                        visibility[key] += 1
                        prism_key = (
                            "true_positive" if actual_visible and prism_visible else
                            "false_positive" if prism_visible else
                            "false_negative" if actual_visible else "true_negative"
                        )
                        prism_visibility[prism_key] += 1
                        if not actual_visible:
                            continue
                        ys, xs = np.where(truth)
                        true_box = np.asarray((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1), np.float64)
                        true_center = (true_box[:2] + true_box[2:]) / 2.0
                        pred_center = (box[:2] + box[2:]) / 2.0
                        weights = view_weights(
                            yaw_radians, data["player_pos"][frame, source],
                            data["cam_pos"][frame, target],
                        )
                        _, sprite_mask = warp_reference(
                            references_by_source[source], weights, box, height=height, width=width
                        )
                        true_size = true_box[2:] - true_box[:2]
                        pred_size = box[2:] - box[:2]
                        row = {
                            "episode": episode.name, "scenario": scenario,
                            "frame": frame, "target": target, "source": source,
                            "true_pixels": true_pixels, "full_instance_pixels": full_true_pixels,
                            "largest_component_fraction": true_pixels / max(full_true_pixels, 1),
                            "source_alive": bool(data["player_alive"][frame, source]),
                            "target_alive": bool(data["player_alive"][frame, target]),
                            "depth": depth,
                            "predicted_visible": predicted_visible,
                            "prism_predicted_visible": prism_visible,
                            "bbox_iou": _box_iou(clipped, true_box),
                            "prism_bbox_iou": _box_iou(prism_clipped, true_box),
                            "center_error_px": float(np.linalg.norm(pred_center - true_center)),
                            "center_error_over_true_height": float(np.linalg.norm(pred_center - true_center) / max(true_size[1], 1)),
                            "width_ratio": float(pred_size[0] / max(true_size[0], 1)),
                            "height_ratio": float(pred_size[1] / max(true_size[1], 1)),
                            "rectangle_mask_iou": _mask_iou(box_mask, truth),
                            "prism_rectangle_mask_iou": _mask_iou(prism_mask, truth),
                            "sprite_mask_iou": _mask_iou(sprite_mask, truth),
                            "visible_fraction_in_true_bbox": float(true_pixels / max(np.prod(true_size), 1)),
                            "selected_view": VIEW_NAMES[int(weights.argmax())],
                        }
                        rows.append(row)
                        if true_pixels >= 256:
                            candidates.append({
                                "episode": str(episode), "frame": frame,
                                "target": target, "source": source,
                                "bbox_iou": row["bbox_iou"],
                            })
        print(f"[{episode_number}/{len(episodes)}] {episode.name}: visible samples={len(rows)}", flush=True)

    fields = list(rows[0]) if rows else []
    with (output / "projection_samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    tp, fp = visibility["true_positive"], visibility["false_positive"]
    fn = visibility["false_negative"]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    prism_tp, prism_fp = prism_visibility["true_positive"], prism_visibility["false_positive"]
    prism_fn = prism_visibility["false_negative"]
    prism_precision = prism_tp / max(prism_tp + prism_fp, 1)
    prism_recall = prism_tp / max(prism_tp + prism_fn, 1)
    report = {
        "data_root": str(root), "episodes": len(episodes),
        "frame_stride": args.frame_stride, "visible_samples": len(rows),
        "visibility": {**visibility, "precision": precision, "recall": recall,
                       "f1": 2 * precision * recall / max(precision + recall, 1e-12)},
        "prism_visibility": {
            **prism_visibility, "precision": prism_precision, "recall": prism_recall,
            "f1": 2 * prism_precision * prism_recall / max(prism_precision + prism_recall, 1e-12),
        },
        "metrics_on_all_actually_visible_players": {
            key: _summary([float(row[key]) for row in rows])
            for key in (
                "bbox_iou", "prism_bbox_iou", "rectangle_mask_iou",
                "prism_rectangle_mask_iou",
                "sprite_mask_iou", "visible_fraction_in_true_bbox",
            )
        },
        "placement_metrics_when_current_projection_is_on_screen": {
            key: _summary([float(row[key]) for row in rows if row["predicted_visible"]])
            for key in (
                "center_error_px", "center_error_over_true_height",
                "width_ratio", "height_ratio",
            )
        },
        "bbox_iou_by_depth": {
            label: {
                "current": _summary([row["bbox_iou"] for row in rows if low <= row["depth"] < high]),
                "prism": _summary([row["prism_bbox_iou"] for row in rows if low <= row["depth"] < high]),
                "current_visibility_recall": float(np.mean([
                    row["predicted_visible"] for row in rows if low <= row["depth"] < high
                ])) if any(low <= row["depth"] < high for row in rows) else None,
                "prism_visibility_recall": float(np.mean([
                    row["prism_predicted_visible"] for row in rows if low <= row["depth"] < high
                ])) if any(low <= row["depth"] < high for row in rows) else None,
            }
            for label, low, high in (("near_lt_2", -np.inf, 2.0), ("mid_2_to_5", 2.0, 5.0), ("far_ge_5", 5.0, np.inf))
        },
        "bbox_iou_by_source_state": {
            state: {
                "current": _summary([row["bbox_iou"] for row in rows if row["source_alive"] is alive]),
                "prism": _summary([row["prism_bbox_iou"] for row in rows if row["source_alive"] is alive]),
            }
            for state, alive in (("alive", True), ("dead_or_respawning", False))
        },
        "largest_component_fraction": _summary(
            [row["largest_component_fraction"] for row in rows]
        ),
        "notes": [
            "Bounding-box metrics isolate placement and scale better than silhouette IoU.",
            "Visibility false positives include players hidden by terrain because this projector has no voxel occlusion test.",
            "Sprite IoU additionally includes pose/animation and canonical-reference silhouette mismatch.",
        ],
        "examples": [],
    }
    if candidates and args.max_examples:
        candidates.sort(key=lambda row: row["bbox_iou"])
        indices = np.linspace(0, len(candidates) - 1, min(args.max_examples, len(candidates))).astype(int)
        for number, index in enumerate(indices):
            candidate = candidates[index]
            image_path = output / f"example_{number:02d}.png"
            appearance_metrics = _draw_example(candidate, image_path)
            report["examples"].append({**candidate, **appearance_metrics, "image": image_path.name})
    rgb_values = [row["rgb_l1_on_overlap"] for row in report["examples"] if np.isfinite(row["rgb_l1_on_overlap"])]
    report["example_rgb_l1_on_overlap"] = _summary(rgb_values)
    (output / "projection_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
