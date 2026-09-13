"""Test whether the fitted GT-camera DiT uses the supplied player observations."""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from experiments.m1.train_multiview_voxel_dit import codec, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--mode",
        choices=["one_view", "shuffled_conditions", "shuffled_images", "new_noise"],
        required=True,
    )
    args = p.parse_args()
    torch.set_num_threads(8)
    cache = torch.load(args.output / "cache.pt", weights_only=False)
    state = torch.load(args.output / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    model = MultiViewVoxelDiT(**state["config"])
    model.load_state_dict(state["model"])
    model.cuda().eval()
    step = state["step"]
    del state
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    if args.mode == "one_view":
        cache["data"]["condition_view_valid"] = cache["data"]["agent_mask"].clone()
        cache["data"]["condition_view_valid"][:, 1:] = False
    elif args.mode == "shuffled_conditions":
        # Move both RGB features and their camera rays together from a different scene.
        cache["data"]["image_rays"] = cache["data"]["image_rays"].roll(1, 0)
    elif args.mode == "shuffled_images":
        # Keep both cameras fixed; replace only visual features with another scene.
        cache["data"]["image_rays"][:, :, :16] = cache["data"]["image_rays"][:, :, :16].roll(1, 0)
    output = args.output / ("ablation_" + args.mode)
    output.mkdir(exist_ok=True)
    evaluate(
        model, decoder, cache, step, output, seed_base=4321 if args.mode == "new_noise" else 1234
    )


if __name__ == "__main__":
    main()
