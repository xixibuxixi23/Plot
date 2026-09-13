"""Fixed-camera shuffled-image diagnostic on fitted full-run samples."""

# ruff: noqa: E402
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from scripts.evaluate_multiview_persist_full import sample
from experiments.m1.train_multiview_persist_full import Cache
from experiments.m1.train_multiview_voxel_dit import codec, evaluate


def main():
    torch.set_num_threads(4)
    out = ROOT / "outputs/m1_multiview_persist_gtcam_s01_all_v1"
    source = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    indices = np.linspace(0, len(source) - 1, 32).round().astype(int).tolist()
    cache = source.evaluation_cache(indices)
    state = torch.load(out / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    model = MultiViewVoxelDiT().cuda().eval()
    model.load_state_dict(state["model"])
    step = state["step"]
    del state
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    original = cache["data"]["image_rays"].clone()
    results = {}
    for mode in ["original", "shuffled_images"]:
        cond = original.clone()
        if mode == "shuffled_images":
            cond[:, :, :16] = cond[:, :, :16].roll(1, 0)
        valid = cache["data"]["agent_mask"] & cache["data"]["camera_valid"]
        predicted = []
        for i in range(0, len(indices), 8):
            predicted.append(
                sample(
                    model, cond[i : i + 8].cuda(), valid[i : i + 8].cuda(), indices[i : i + 8]
                ).cpu()
            )
        cache["data"]["sampled_latent"] = torch.cat(predicted)
        folder = out / f"condition_probe_{mode}"
        folder.mkdir(exist_ok=True)
        results[mode] = evaluate(model, decoder, cache, step, folder)["mean"]
    (out / "condition_probe.json").write_text(
        json.dumps(dict(indices=indices, results=results), indent=2)
    )


if __name__ == "__main__":
    main()
