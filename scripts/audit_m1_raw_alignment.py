"""Independent world-coordinate voxel DDA over raw M1 inputs; no model projection imports."""

import sys, json, argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.train_multiview_persist_full import Source, Cache


def render(raw, center, pos, direction, fx, fy, h=180, w=320):
    # Nodes occupy world-coordinate cubes centered at integer coordinates.
    f = direction / np.linalg.norm(direction)
    r = np.cross(f, [0, 0, 1])
    r = r / np.linalg.norm(r)
    u = np.cross(r, f)
    xx, yy = np.meshgrid((np.arange(w) + 0.5) * 2 / w - 1, 1 - (np.arange(h) + 0.5) * 2 / h)
    d = f + xx[..., None] * np.tan(fx / 2) * r + yy[..., None] * np.tan(fy / 2) * u
    d = d.reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1)[:, None]
    low = np.asarray(center) - 24.5
    origin = np.asarray(pos) - low
    inv = np.divide(1, d, out=np.full_like(d, np.inf), where=np.abs(d) > 1e-10)
    ta = (0 - origin) * inv
    tb = (48 - origin) * inv
    enter = np.maximum(np.minimum(ta, tb).max(1), 0)
    leave = np.maximum(ta, tb).min(1)
    live = leave > enter
    t = enter + 1e-6
    ids = np.full(len(d), 126)
    depth = np.full(len(d), np.nan)
    cell = np.floor(origin + t[:, None] * d).astype(int)
    step = np.sign(d).astype(int)
    for _ in range(160):
        active = np.flatnonzero(live & (cell >= 0).all(1) & (cell < 48).all(1) & (t <= leave))
        if not len(active):
            break
        vals = raw[tuple(cell[active].T)]
        hit = active[(vals != 126) & (vals != 127)]
        ids[hit] = raw[tuple(cell[hit].T)]
        depth[hit] = t[hit]
        live[hit] = False
        a = active[live[active]]
        boundary = cell[a] + (step[a] > 0)
        nxt = np.where(np.abs(d[a]) > 1e-10, (boundary - origin) * inv[a], np.inf)
        axis = nxt.argmin(1)
        t[a] = nxt[np.arange(len(a)), axis] + 1e-7
        cell[a, axis] += step[a, axis]
    return ids.reshape(h, w), depth.reshape(h, w)


def main():
    out = ROOT / "outputs/m1_raw_alignment_v1"
    out.mkdir(exist_ok=True)
    meta = json.loads((ROOT / "outputs/m1_multiview_persist_s01_cache/metadata.json").read_text())
    source = Source(
        argparse.Namespace(dataset_root=Path(meta["dataset_root"]), persist=ROOT.parent / "PERSIST")
    )
    cache = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    nodes = json.loads(
        (ROOT / "outputs/diagnostics/node_geometry/node_geometry_mapping.json").read_text()
    )["nodes"]
    trunks = [int(n["id"]) for n in nodes if n.get("groups", {}).get("tree")]
    leaves = [int(n["id"]) for n in nodes if n.get("groups", {}).get("leaves")]
    candidates = np.linspace(0, len(source) - 1, 64).round().astype(int)
    selected = [int(i) for i in candidates if np.isin(cache.arrays["raw"][i], trunks).sum() > 10][
        :8
    ]
    rows = []
    for i in selected:
        split, j = source.items[i]
        ds = source.datasets[split]
        ep, target, _ = ds.index[j]
        path, manifest = ds.episodes[ep]
        b = source[i]
        with np.load(path / "m1_initial.npz") as z:
            raw = z["voxel_tiles"][target, ..., 0]
            center = z["voxel_center"][target]
            pos = z["cam_pos"]
            dire = z["cam_dir"]
            keys = list(z.files)
            fy = z["fov_y"] if "fov_y" in z else np.full(len(pos), np.deg2rad(72.0))
            fx = (
                z["fov_x"]
                if "fov_x" in z
                else 2
                * np.arctan(
                    float(manifest.get("width", 1280))
                    / float(manifest.get("height", 720))
                    * np.tan(fy / 2)
                )
            )
        order = [target] + [k for k in range(len(pos)) if k != target]
        fig, axes = plt.subplots(2, 3, figsize=(15, 6))
        errs = {
            k: float(
                np.max(
                    np.abs(
                        b[k].numpy().astype(float) - np.asarray(cache.arrays[k][i]).astype(float)
                    )
                )
            )
            for k in ["raw", "camera_position", "camera_direction", "fov_x", "fov_y"]
        }
        for v, a in enumerate(order[:2]):
            ids, depth = render(raw, center, pos[a], dire[a], fx[a], fy[a])
            rgb = b["images"][v].permute(1, 2, 0).numpy()
            axes[v, 0].imshow(rgb)
            axes[v, 0].set_title(f"Raw RGB window {i} player {a}")
            axes[v, 1].imshow(rgb, extent=(0, 320, 180, 0))
            tree = np.isin(ids, trunks)
            leaf = np.isin(ids, leaves)
            if tree.any():
                axes[v, 1].contour(tree, levels=[0.5], colors=["red"], linewidths=0.8)
            if leaf.any():
                axes[v, 1].contour(leaf, levels=[0.5], colors=["cyan"], linewidths=0.5)
            axes[v, 1].set_title("Independent GT: trunk red / leaves cyan")
            axes[v, 2].imshow(depth, vmin=0, vmax=32)
            axes[v, 2].set_title("Independent GT first cube depth")
            np.savez_compressed(out / f"window_{i}_player_{a}.npz", ids=ids, depth=depth)
        for ax in axes.flat:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"window_{i}.png", dpi=130)
        plt.close(fig)
        rows.append(
            dict(
                index=i,
                path=str(path),
                target_agent=int(target),
                cache_max_abs_errors=errs,
                raw_keys=keys,
                manifest=manifest,
            )
        )
        print(i, errs, flush=True)
    (out / "audit.json").write_text(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
