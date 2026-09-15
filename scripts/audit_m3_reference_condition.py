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
from plot.models.player_identity import PlayerIdentityEncoder, crop_masked_players
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
def _predict(
    model, codec, sample, *, seed, precision, mask_prefix_players=False,
    denoising_steps=20, horizon=64,
):
    device = next(model.parameters()).device
    rgb = sample["rgb"][:, :65].to(device)
    if mask_prefix_players:
        # Diagnostic only: remove the redundant identity evidence from the
        # rollout prefix while keeping geometry/reference conditions intact.
        # Mid-gray avoids introducing an extreme all-black latent patch.
        prefix_mask = sample["player_region_mask"][:, :1].to(device).bool()
        rgb = rgb.clone()
        rgb[:, :1] = torch.where(prefix_mask, rgb.new_tensor(0.5), rgb[:, :1])
    conditions = {
        key: value.to(device) for key, value in sample["conditions"].items()
    }
    use_bf16 = precision == "bf16"
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        latent = codec.encode(rgb)
        generator = torch.Generator(device=device).manual_seed(seed)
        if horizon < model.cfg.block_frames or horizon % model.cfg.block_frames:
            raise ValueError("audit horizon must contain complete renderer blocks")
        noise = torch.randn(
            latent[:, 1:1 + horizon].shape,
            device=device,
            dtype=latent.dtype,
            generator=generator,
        )
        rollout = RendererRollout(model, denoising_steps=denoising_steps)
        try:
            rollout.start(latent[:, :1], slice_conditions(conditions, 0, 1))
            future = slice_conditions(conditions, 1, 1 + horizon)
            chunks = []
            for start in range(0, horizon, model.cfg.block_frames):
                end = start + model.cfg.block_frames
                chunks.append(rollout.generate(
                    noise[:, start:end], slice_conditions(future, start, end)
                ))
            prediction = codec.decode(torch.cat(chunks, dim=1))[0]
        finally:
            model.clear_cache()
    return rgb[0, 1:1 + horizon], prediction


def _masked_mean(value, mask):
    return float((value * mask).sum() / (mask.sum().clamp_min(1) * value.shape[1]))


def _load_identity_encoder(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    encoder = PlayerIdentityEncoder(int(payload.get("embedding_dim", 128))).to(device).eval()
    encoder.crop_size = tuple(payload.get("crop_size", (128, 64)))
    encoder.load_state_dict(payload["model"], strict=True)
    return encoder


@torch.no_grad()
def _identity_metrics(identity_encoder, prediction, raw):
    """Score a future-only rollout against the selected resident reference."""
    if not bool(raw["player_identity_valid"]):
        return None
    # Dataset indices address the original 65-frame clip, while ``prediction``
    # contains only frames 1..64.
    frame = int(raw["player_identity_frame"])
    prediction_frame = frame - 1
    if not 0 <= prediction_frame < len(prediction):
        raise ValueError("player identity frame is outside the predicted future")
    slot = int(raw["player_identity_slot"])
    mask = raw["player_identity_mask"].to(prediction.device)[None].float()
    crop, crop_valid = crop_masked_players(
        prediction[prediction_frame : prediction_frame + 1],
        mask,
        output_size=identity_encoder.crop_size,
    )
    if not bool(crop_valid[0]):
        return None
    references = raw["conditions"]["player_reference"].to(prediction.device)
    crop_embedding = identity_encoder.encode_crop(crop)
    reference_embedding = identity_encoder.encode_reference(references[slot : slot + 1])
    similarity = (crop_embedding * reference_embedding).sum(-1)[0]
    result = {"identity_similarity": float(similarity)}

    appearance_valid = raw["conditions"].get("player_appearance_valid")
    if appearance_valid is not None:
        candidate_valid = appearance_valid.bool().any(-1)
        candidates = torch.where(
            candidate_valid
            & (torch.arange(len(candidate_valid), device=candidate_valid.device) != slot)
        )[0]
        if len(candidates):
            wrong_slot = int(candidates[0])
            wrong_embedding = identity_encoder.encode_reference(
                references[wrong_slot : wrong_slot + 1]
            )
            wrong_similarity = (crop_embedding * wrong_embedding).sum(-1)[0]
            result.update({
                "identity_wrong_similarity": float(wrong_similarity),
                "identity_ranking_correct": bool(similarity > wrong_similarity),
            })
    return result


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
    parser.add_argument(
        "--normal-only", action="store_true",
        help="Run only the ordinary rollout and label it as generated output.",
    )
    parser.add_argument(
        "--probe-manifest",
        help="Optional probe JSON path; defaults to the checkpoint visualization manifest.",
    )
    parser.add_argument(
        "--mask-prefix-players",
        action="store_true",
        help=(
            "diagnostic: replace visible player pixels in the first rollout "
            "frame so reference dependence is not hidden by video history"
        ),
    )
    parser.add_argument(
        "--identity-checkpoint",
        help="frozen player identity encoder (defaults to the training config)",
    )
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
    identity_path = args.identity_checkpoint or config["training"].get(
        "player_identity_checkpoint"
    )
    identity_encoder = (
        _load_identity_encoder(identity_path, device) if identity_path else None
    )
    vocabulary = config["training"]["vocabulary"]
    probe_path = (
        Path(args.probe_manifest) if args.probe_manifest
        else checkpoint_path.parent / "visualizations" / "probes.json"
    )
    probes = [RendererProbe(**row) for row in json.loads(probe_path.read_text())]
    probes = [probe for probe in probes if probe.name in set(args.probes)]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(checkpoint_path),
        "identity_checkpoint": str(identity_path) if identity_path else None,
        "mask_prefix_players": args.mask_prefix_players,
        "normal_only": args.normal_only,
        "probes": {},
    }

    for probe_number, probe in enumerate(probes):
        raw = TextAgentRendererDataset.read_window(
            probe.episode,
            vocabulary,
            start=probe.start,
            target=probe.target,
            context_frames=65,
        )
        variants = {}
        variant_names = ("generated",) if args.normal_only else ("correct", "shuffled", "zero")
        for name in variant_names:
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
                mask_prefix_players=args.mask_prefix_players,
            )
            variants[name] = prediction.float()
            output_name = "gt_vs_generated" if args.normal_only else name
            write_comparison_video(
                output / f"{probe.name}_{output_name}.mp4",
                truth,
                prediction,
                sample["region_weight"][0, 1:65],
            )
        player_mask = raw["player_region_mask"][1:65].to(device).float()
        outside = 1 - player_mask
        primary_name = "generated" if args.normal_only else "correct"
        primary_error = (variants[primary_name] - truth.float()).abs()
        row = {
            "player_pixels": float(player_mask.sum()),
            f"{primary_name}_player_l1": _masked_mean(primary_error, player_mask),
            f"{primary_name}_global_l1": float(primary_error.mean()),
        }
        for name in ("shuffled", "zero"):
            if name not in variants:
                continue
            delta = (variants[name] - variants[primary_name]).abs()
            error = (variants[name] - truth.float()).abs()
            row.update({
                f"{name}_player_l1": _masked_mean(error, player_mask),
                f"{name}_conditioning_delta_player": _masked_mean(delta, player_mask),
                f"{name}_conditioning_delta_outside": _masked_mean(delta, outside),
            })
        if identity_encoder is not None:
            for name, prediction in variants.items():
                metrics = _identity_metrics(identity_encoder, prediction, raw)
                if metrics is not None:
                    row.update({f"{name}_{key}": value for key, value in metrics.items()})
        report["probes"][probe.name] = row
        (output / "reference_condition_audit.json").write_text(
            json.dumps(report, indent=2)
        )
        print(probe.name, json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
