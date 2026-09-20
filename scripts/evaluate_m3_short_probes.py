"""Fixed-seed pure-noise short-block evaluation before/after player fine-tuning."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec
from plot.training.renderer_monitoring import RendererProbe, render_probe
from train_scripts.train_renderer import load_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--denoising-steps", type=int, default=20)
    args = parser.parse_args()
    config = json.loads((args.checkpoint.parent / "config.json").read_text())
    names = {field.name for field in fields(RendererArgs)}
    renderer = {key: value for key, value in config["renderer"].items() if key in names}
    renderer["context_frames"] = args.context_frames
    model = Renderer(RendererArgs(**renderer)).cuda().eval()
    model.load_state_dict(load_weights(args.checkpoint), strict=True)
    codec = RendererCodec.from_run_config(load_weights(config["training"]["pixel_vae"]), config).cuda().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": str(args.checkpoint), "seed": args.seed,
              "context_frames": args.context_frames, "denoising_steps": args.denoising_steps,
              "protocol": "one GT prefix; future initialized from Gaussian noise; GT future only used for scoring",
              "probes": {}}
    probes = [RendererProbe(**row) for row in json.loads(args.probe_manifest.read_text())]
    for i, probe in enumerate(probes):
        sample = TextAgentRendererDataset.read_window(probe.episode, config["training"]["vocabulary"],
                    start=probe.start, target=probe.target, context_frames=args.context_frames)
        metrics = render_probe(model, codec, collate_renderer([sample]), args.output_dir / f"{probe.name}.mp4",
                    seed=args.seed + i, denoising_steps=args.denoising_steps,
                    precision=config["training"].get("precision", "bf16"))
        report["probes"][probe.name] = metrics
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({probe.name: metrics}), flush=True)


if __name__ == "__main__":
    main()
