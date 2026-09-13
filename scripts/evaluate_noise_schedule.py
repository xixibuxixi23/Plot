"""Matched final pure-noise and mid-noise recovery metrics, including visible trees."""

# ruff: noqa: E402
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from experiments.m1.train_multiview_persist_full import Cache
from experiments.m1.train_multiview_voxel_dit import codec, evaluate
from plot.models.visible_supervision import visible_voxel_masks


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    out = args.output
    state = torch.load(
        out / "checkpoint_latest.pt", map_location="cpu", mmap=True, weights_only=False
    )
    model = MultiViewVoxelDiT().cuda()
    model.load_state_dict(state["model"])
    step = state["step"]
    del state
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    source = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    indices = np.linspace(0, len(source) - 1, 64).round().astype(int).tolist()
    cache = source.evaluation_cache(indices)
    data = cache["data"]
    mapping = json.loads(
        (ROOT / "outputs/diagnostics/node_geometry/node_geometry_mapping.json").read_text()
    )
    tree_ids = {
        k: torch.tensor(
            [int(n["id"]) for n in mapping["nodes"] if n.get("groups", {}).get(group)],
            device="cuda",
        )
        for k, group in [("trunk", "tree"), ("leaves", "leaves")]
    }
    results = {}
    for start_t in [1.0, 0.5]:
        preds = []
        model.eval()
        for offset in range(0, 64, 4):
            ix = indices[offset : offset + 4]
            target = data["latent"][offset : offset + 4].cuda()
            cond = data["image_rays"][offset : offset + 4].cuda()
            valid = (data["agent_mask"] & data["camera_valid"])[offset : offset + 4].cuda()
            noise = torch.cat(
                [
                    torch.randn(
                        1,
                        48,
                        12,
                        12,
                        12,
                        device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(1234 + i),
                    )
                    for i in ix
                ]
            )
            x = (1 - start_t) * target + (1e-5 + (1 - 1e-5) * start_t) * noise
            schedule = np.linspace(1, 0, 21)
            schedule = start_t * 3 * schedule / (1 + 2 * schedule)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for t, tp in zip(schedule[:-1], schedule[1:]):
                    x = x - float(t - tp) * model(
                        x, torch.full((len(ix),), t, device="cuda"), cond, valid
                    )
            preds.append(x.float().cpu())
        data["sampled_latent"] = torch.cat(preds)
        folder = out / f"eval64_t{start_t}"
        folder.mkdir(exist_ok=True)
        result = evaluate(model, decoder, cache, step, folder)
        rawpred = torch.load(folder / f"prediction_{step:06d}.pt", weights_only=False)["pred"]
        tree = {k: [] for k in tree_ids}
        for i in range(64):
            gt = data["raw"][i : i + 1].cuda()
            pred = rawpred[i : i + 1].cuda()
            surface, free = visible_voxel_masks(
                gt,
                gt != 127,
                data["camera_position"][i : i + 1].cuda(),
                data["camera_direction"][i : i + 1].cuda(),
                (data["agent_mask"] & data["camera_valid"])[i : i + 1].cuda(),
                data["fov_x"][i : i + 1].cuda(),
                data["fov_y"][i : i + 1].cuda(),
                126,
            )
            for k, ids in tree_ids.items():
                mask = torch.isin(gt, ids) & surface
                tree[k].append(
                    dict(
                        index=indices[i],
                        gt=int(mask.sum()),
                        pred=int((torch.isin(pred, ids) & (surface | free)).sum()),
                        correct=int(((pred == gt) & mask).sum()),
                    )
                )
        summary = {}
        for k, rows in tree.items():
            validrows = [r for r in rows if r["gt"]]
            summary[k] = dict(
                windows=len(validrows),
                macro_recall=sum(r["correct"] / r["gt"] for r in validrows) / len(validrows),
                micro_precision=sum(r["correct"] for r in rows)
                / max(1, sum(r["pred"] for r in rows)),
            )
        results[str(start_t)] = dict(**result, tree=summary, tree_samples=tree)
        print(start_t, result["mean"], summary, flush=True)
    (out / "final_eval64.json").write_text(
        json.dumps(dict(step=step, indices=indices, results=results), indent=2)
    )


if __name__ == "__main__":
    main()
