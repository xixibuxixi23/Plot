"""Plot fitted-scene geometry and curves from the GT-camera PERSIST pilot."""

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plot.models.projection import raycast_voxel_targets


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--all-scenes", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    cache = torch.load(args.output / "cache.pt", weights_only=False)
    data = cache["data"]
    path = sorted(args.output.glob("prediction_*.pt"))[-1]
    pred = torch.load(path, weights_only=False)["pred"]
    selection_path = args.output / "selection.json"
    selection = json.loads(selection_path.read_text()) if selection_path.exists() else None
    mapping = json.loads(
        (ROOT / "outputs/diagnostics/node_geometry/node_geometry_mapping.json").read_text()
    )
    tree_ids = torch.tensor(
        [
            int(n["id"])
            for n in mapping["nodes"]
            if n.get("groups", {}).get("tree") or n.get("groups", {}).get("leaves")
        ]
    )
    tree_rows = []
    for i in range(len(pred)):
        gt = data["raw"][i]
        tree = torch.isin(gt, tree_ids)
        tree_rows.append(
            dict(
                sample=i,
                blocks=int(tree.sum()),
                exact_recall=float((pred[i][tree] == gt[tree]).float().mean())
                if tree.any()
                else None,
            )
        )
        if not args.all_scenes and i > 1 and not tree.any():
            continue
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        label = (
            f"window {selection[i]['index']} · recall {selection[i]['surface_exact_recall']:.1%}"
            if selection
            else f"fitted scene {i}"
        )
        for view in range(2):
            axes[view, 0].imshow(data["images"][i, view].permute(1, 2, 0).numpy())
            axes[view, 0].set_title(f"Player {view + 1} input ({label})")
            for col, vol in enumerate((gt, pred[i]), 1):
                hit, depth, _ = raycast_voxel_targets(
                    (vol[None].cuda() != 126).float(),
                    data["camera_position"][i : i + 1, view].cuda(),
                    data["camera_direction"][i : i + 1, view].cuda(),
                    data["fov_x"][i : i + 1, view].cuda(),
                    data["fov_y"][i : i + 1, view].cuda(),
                    height=90,
                    width=160,
                    samples=512,
                    max_distance=32.0,
                )
                depth = torch.where(hit.bool(), depth, 32)[0].cpu()
                axes[view, col].imshow(depth, vmin=0, vmax=32, cmap="viridis")
                axes[view, col].set_title("GT cube depth" if col == 1 else "Predicted cube depth")
        fig.tight_layout()
        fig.savefig(args.output / f"fitted_scene_{i:02d}.png", dpi=130)
        plt.close(fig)
    (args.output / "tree_metrics.json").write_text(json.dumps(tree_rows, indent=2))
    rows = [json.loads(p.read_text()) for p in sorted(args.output.glob("metrics_*.json"))]
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ("surface_exact_recall", "visible_exact_precision", "gt_camera_half_block_hit"):
        ax.plot(
            [r["step"] for r in rows], [100 * r["mean"][key] for r in rows], marker="o", label=key
        )
    ax.axhline(90, color="gray", linestyle="--")
    ax.set(xlabel="Training updates", ylabel="Fitted-scene macro mean (%)", ylim=(0, 100))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output / "fit_curve.png", dpi=150)


if __name__ == "__main__":
    main()
