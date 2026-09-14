import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_other_player_projection.py"
SPEC = importlib.util.spec_from_file_location("projection_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_centered_upright_player_projects_near_image_center():
    box, depth = MODULE.project_player_box(
        # Feet at z=.7 put the 1.8 m body center at the 1.6 m camera height.
        np.array([0.0, 4.0, 0.7]),
        np.array([0.0, 0.0, 1.6]),
        np.array([0.0, 1.0, 0.0]),
        1.823954,
        height=360,
        width=640,
    )
    center = (box[:2] + box[2:]) / 2
    assert depth == 4.0
    assert abs(center[0] - 319.5) < 1e-5
    assert abs(center[1] - 179.5) < 1e-5
    assert box[3] > box[1]


def test_view_selection_matches_minetest_yaw_convention():
    front = MODULE.view_weights(0.0, np.zeros(3), np.array([0.0, 5.0, 1.6]))
    back = MODULE.view_weights(0.0, np.zeros(3), np.array([0.0, -5.0, 1.6]))
    assert MODULE.VIEW_NAMES[int(front.argmax())] == "front"
    assert MODULE.VIEW_NAMES[int(back.argmax())] == "back"


def test_prism_projection_is_finite_and_covers_player_center():
    args = (
        np.array([1.0, 4.0, 0.7]), np.array([0.0, 0.0, 1.6]),
        np.array([0.0, 1.0, 0.0]), 1.823954,
    )
    simple, _ = MODULE.project_player_box(*args, height=360, width=640)
    prism, _ = MODULE.project_player_prism_box(
        args[0], 0.0, *args[1:], height=360, width=640
    )
    assert np.isfinite(prism).all()
    simple_center = (simple[:2] + simple[2:]) / 2
    assert prism[0] < simple_center[0] < prism[2]
    assert prism[1] < simple_center[1] < prism[3]


def test_warped_reference_stays_inside_projected_box():
    references = np.ones((4, 8, 4, 4), np.float32)
    projected, mask = MODULE.warp_reference(
        references, np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([2.0, 3.0, 8.0, 11.0]), height=16, width=16,
    )
    assert mask.sum() == 48
    assert projected[:3].sum() == 0
    assert projected[:, :2].sum() == 0


def test_largest_component_removes_detached_item():
    mask = np.zeros((12, 12), bool)
    mask[2:8, 2:7] = True
    mask[9:11, 9:11] = True
    body = MODULE._largest_component(mask)
    assert body.sum() == 30
    assert not body[9, 9]
