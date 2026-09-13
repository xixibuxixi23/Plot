"""Compare M1 cache to original capture arrays and video frames."""

import json
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "outputs/m1_raw_alignment_v1"
rows = json.loads((out / "audit.json").read_text())
results = []
for row in rows:
    p = Path(row["path"])
    with np.load(p / "m1_initial.npz") as c, np.load(p / "data.npz") as d:
        t = int(c["observation"])
        target = row["target_agent"]
        raw = d["obs_voxel_mt"][t, target, :48, :48, :48]
        checks = {
            "raw_tile_equal": bool(np.array_equal(raw, c["voxel_tiles"][target])),
            "center_equal": bool(np.array_equal(d["obs_voxel_center"][t], c["voxel_center"])),
        }
        for k in ["cam_pos", "cam_dir"]:
            checks[k + "_max_error"] = float(np.abs(d[k][t] - c[k]).max())
        fy = d["fov_y"][t]
        fx = d["fov_x"][t]
        K = d["intrinsics"][t]
        checks["fallback_fovy_max_error"] = float(np.abs(fy - np.deg2rad(72)).max())
        checks["fallback_fovx_max_error"] = float(
            np.abs(fx - 2 * np.arctan(1280 / 720 * np.tan(np.deg2rad(72) / 2))).max()
        )
        checks["K_fov_max_error"] = float(
            max(
                np.abs(K[:, 0, 0] - 1 / (2 * np.tan(fx / 2))).max(),
                np.abs(K[:, 1, 1] - 1 / (2 * np.tan(fy / 2))).max(),
            )
        )
    videos = []
    for a in range(2):
        jpg = cv2.imread(str(p / f"m1_rgb_agent{a}.jpg"))
        cap = cv2.VideoCapture(str(p / f"rgb_agent{a}.mp4"))
        errors = {}
        for frame in range(max(0, t - 2), t + 3):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, img = cap.read()
            if ok:
                errors[frame] = float(
                    np.abs(
                        cv2.resize(
                            img, (jpg.shape[1], jpg.shape[0]), interpolation=cv2.INTER_AREA
                        ).astype(float)
                        - jpg.astype(float)
                    ).mean()
                )
        cap.release()
        videos.append(
            dict(
                agent=a,
                cached_shape=list(jpg.shape),
                frame_mae=errors,
                best_frame=min(errors, key=errors.get),
            )
        )
    results.append(dict(index=row["index"], observation=t, checks=checks, videos=videos))
    print(results[-1], flush=True)
(out / "provenance.json").write_text(json.dumps(results, indent=2))
