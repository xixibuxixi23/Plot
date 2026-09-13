"""Fixed checkpoint and noise, compare Euler step counts on fitted scenes."""

# ruff: noqa: E402
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from experiments.m1.train_multiview_persist_full import Cache
from experiments.m1.train_multiview_voxel_dit import codec, evaluate
from scripts.evaluate_multiview_persist_full import sample


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    out = ROOT / "outputs/m1_noise_schedule_compare_v1"
    state = torch.load(
        out / "source_checkpoint.pt", map_location="cpu", mmap=True, weights_only=False
    )
    model = MultiViewVoxelDiT().cuda()
    model.load_state_dict(state["model"])
    step = state["step"]
    del state
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    source = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    indices = np.linspace(0, len(source) - 1, 64).round().astype(int).tolist()
    cache = source.evaluation_cache(indices)
    d = cache["data"]
    results = {}
    for steps in [20, 50, 100]:
        preds = []
        for i in range(0, 64, 4):
            preds.append(
                sample(
                    model,
                    d["image_rays"][i : i + 4].cuda(),
                    (d["agent_mask"] & d["camera_valid"])[i : i + 4].cuda(),
                    indices[i : i + 4],
                    steps=steps,
                ).cpu()
            )
        d["sampled_latent"] = torch.cat(preds)
        folder = out / f"sampling_{steps}"
        folder.mkdir(exist_ok=True)
        results[str(steps)] = evaluate(model, decoder, cache, step, folder)
        print(steps, results[str(steps)]["mean"], flush=True)
    (out / "sampling_steps.json").write_text(
        json.dumps(dict(step=step, indices=indices, results=results), indent=2)
    )


if __name__ == "__main__":
    main()
