"""Measure whether M3 output actually changes with its native player reference."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import load_file

from plot.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec
from plot.training.renderer_monitoring import RendererProbe, write_comparison_video
from plot.training.renderer_trainer import RendererRollout, slice_conditions


def _weights(path):
    if Path(path).suffix == ".safetensors":
        return load_file(str(path))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload.get("model", payload)


@torch.no_grad()
def _predict(model, codec, sample, *, seed, precision):
    device = next(model.parameters()).device
    rgb = sample["rgb"][:, :65].to(device)
    conditions = {
        key: value.to(device) for key, value in sample["conditions"].items()
    }
    use_bf16 = precision == "bf16"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        latent = codec.encode(rgb)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(
            latent[:, 1:65].shape,
            device=device,
            dtype=latent.dtype,
            generator=generator,
        )
        rollout = RendererRollout(model, denoising_steps=20)
        try:
            rollout.start(latent[:, :1], slice_conditions(conditions, 0, 1))
            prediction = codec.decode(
                rollout.generate_64(noise, slice_conditions(conditions, 1, 65))
            )[0]
        finally:
            model.clear_cache()
    return rgb[0, 1:65], prediction


def _masked_mean(value, mask):
    return float((value * mask).sum() / (mask.sum().clamp_min(1) * value.shape[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--probes",
        nargs="*",
        default=("construction", "four_player", "three_resident_combat"),
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else checkpoint_path.parent / "config.json"
    config = json.loads(config_path.read_text())
    renderer_keys = {field.name for field in fields(RendererArgs)}
    renderer_config = {
        key: value for key, value in config["renderer"].items() if key in renderer_keys
    }
    device = torch.device(args.device)
    model = Renderer(RendererArgs(**renderer_config)).to(device).eval()
    model.load_state_dict(_weights(checkpoint_path), strict=True)
    codec = RendererCodec(_weights(config["training"]["pixel_vae"])).to(device).eval()
    vocabulary = config["training"]["vocabulary"]
    probe_path = checkpoint_path.parent / "visualizations" / "probes.json"
    probes = [RendererProbe(**row) for row in json.loads(probe_path.read_text())]
    probes = [probe for probe in probes if probe.name in set(args.probes)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(checkpoint_path), "probes": {}}

    for probe_number, probe in enumerate(probes):
        raw = TextAgentRendererDataset.read_window(
            probe.episode,
            vocabulary,
            start=probe.start,
            target=probe.target,
            context_frames=65,
        )
        variants = {}
        for name in ("correct", "shuffled", "zero"):
            sample = collate_renderer([raw])
            if name == "shuffled":
                sample["conditions"]["player_reference"] = sample["conditions"][
                    "player_reference"
                ].roll(1, dims=1)
            elif name == "zero":
                sample["conditions"]["player_reference"].zero_()
            truth, prediction = _predict(
                model,
                codec,
                sample,
                seed=probe_number,
                precision=config["training"].get("precision", "bf16"),
            )
            variants[name] = prediction.float()
            write_comparison_video(
                output / f"{probe.name}_{name}.mp4",
                truth,
                prediction,
                sample["region_weight"][0, 1:65],
            )
        player_mask = raw["player_region_mask"][1:65].to(device).float()
        outside = 1 - player_mask
        correct_error = (variants["correct"] - truth.float()).abs()
        row = {
            "player_pixels": float(player_mask.sum()),
            "correct_player_l1": _masked_mean(correct_error, player_mask),
            "correct_global_l1": float(correct_error.mean()),
        }
        for name in ("shuffled", "zero"):
            delta = (variants[name] - variants["correct"]).abs()
            error = (variants[name] - truth.float()).abs()
            row.update({
                f"{name}_player_l1": _masked_mean(error, player_mask),
                f"{name}_conditioning_delta_player": _masked_mean(delta, player_mask),
                f"{name}_conditioning_delta_outside": _masked_mean(delta, outside),
            })
        report["probes"][probe.name] = row
        (output / "reference_condition_audit.json").write_text(
            json.dumps(report, indent=2)
        )
        print(probe.name, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
