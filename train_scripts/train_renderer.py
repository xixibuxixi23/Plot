"""Train M3 from accepted continuous TextAgent episodes and a frozen Pixel VAE."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import json
from pathlib import Path
import sys
import os
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from plot.data.renderer_dataset import TextAgentRendererDataset, collate_renderer
from plot.data.appearance_counterfactual_dataset import AppearanceCounterfactualRendererDataset
from plot.models.renderer import Renderer, RendererArgs
from plot.models.renderer_codec import RendererCodec
from plot.models.latent_normalization import resolve_training_normalization
from plot.models.player_identity import PlayerIdentityEncoder
from plot.checkpoint_io import record_checkpoint_failure, staged_torch_save
from plot.training.renderer_monitoring import render_probe, save_probe_manifest, select_renderer_probes
from plot.training.renderer_trainer import renderer_training_losses
from plot.training.qk_warm_start import load_qk_warm_start


LOSS_NAMES = (
    "total_loss",
    "flow_loss",
    "auxiliary_loss",
    "entity_pixel_l1",
    "entity_pixel_edge",
    "player_pixel_l1",
    "player_pixel_edge",
    "health_pixel_l1",
    "player_identity_loss",
    "player_identity_similarity",
    "player_identity_ranking_accuracy",
    "counterfactual_player_difference_loss",
)

AUXILIARY_LOSS_WEIGHT_NAMES = (
    "entity_pixel_l1_weight",
    "entity_pixel_edge_weight",
    "player_pixel_l1_weight",
    "player_pixel_edge_weight",
    "health_pixel_l1_weight",
    "player_identity_loss_weight",
    "counterfactual_player_difference_weight",
)


def load_weights(path):
    if Path(path).suffix == ".safetensors":
        return load_file(str(path))
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value.get("model", value)


def build_flow_region_weight(
    base_weight,
    player_mask,
    *,
    player_upweight,
    probability,
    randomize,
):
    """Apply player-focused flow supervision on a random subset of samples."""
    if player_upweight <= 0 or probability <= 0:
        return base_weight
    batch, frames = player_mask.shape[:2]
    latent_mask = torch.nn.functional.interpolate(
        player_mask.float().flatten(0, 1),
        size=base_weight.shape[-2:],
        mode="area",
    ).unflatten(0, (batch, frames))
    if randomize:
        gate = (torch.rand(batch, device=base_weight.device) < probability).float()
    else:
        gate = base_weight.new_full((batch,), probability)
    return base_weight + player_upweight * gate[:, None, None, None, None] * latent_mask


def use_counterfactual_step(step: int, probability: float, seed: int) -> bool:
    """Choose mixed S11 steps reproducibly, including after checkpoint resume."""
    if probability <= 0:
        return False
    if probability >= 1:
        return True
    # SplitMix64 gives every absolute training step a stable pseudo-random draw.
    # Depending on the absolute step (rather than iterator RNG state) makes a
    # resumed run use exactly the same data-source schedule.
    mask = (1 << 64) - 1
    value = (int(step) + int(seed) * 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    value ^= value >> 31
    return value < int(float(probability) * (1 << 64))


def effective_auxiliary_loss_weights(args):
    """Resolve configured auxiliary weights for the selected training mode."""
    if args.loss_mode == "flow":
        return {name: 0.0 for name in AUXILIARY_LOSS_WEIGHT_NAMES}
    return {name: float(getattr(args, name)) for name in AUXILIARY_LOSS_WEIGHT_NAMES}


def load_renderer_resume(
    model,
    optimizer,
    checkpoint,
    item_vocabulary,
    *,
    allow_item_vocabulary_extension=False,
    optimizer_lrs=None,
):
    """Strictly resume M3, optionally appending new item embedding rows.

    Item IDs already present in the checkpoint may never be renamed or moved.
    New IDs retain the model's normal random initialization, while the Adam
    moments for those rows start at zero. Every other model tensor and all
    scalar optimizer state remain an exact resume.
    """
    embedding_key = "resident_encoder.item_embedder.weight"
    saved_model = dict(checkpoint["model"])
    saved_embedding = saved_model[embedding_key]
    current_embedding = model.state_dict()[embedding_key]
    checkpoint_vocabulary = checkpoint.get("config", {}).get("item_vocabulary")

    if checkpoint_vocabulary is not None:
        changed = {
            name: (index, item_vocabulary.get(name))
            for name, index in checkpoint_vocabulary.items()
            if item_vocabulary.get(name) != index
        }
        if changed:
            raise RuntimeError(
                "item vocabulary changed existing checkpoint IDs: "
                f"{changed}"
            )

    if saved_embedding.shape == current_embedding.shape:
        model.load_state_dict(saved_model, strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        saved_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if optimizer_lrs is not None:
            if len(optimizer_lrs) != len(optimizer.param_groups):
                raise RuntimeError(
                    "resume LR override group count differs: "
                    f"checkpoint={len(optimizer.param_groups)} "
                    f"requested={len(optimizer_lrs)}"
                )
            for group, lr in zip(optimizer.param_groups, optimizer_lrs):
                group["lr"] = float(lr)
        return {
            "item_vocabulary_extended": False,
            "new_items": {},
            "optimizer_saved_lrs": saved_lrs,
            "optimizer_active_lrs": [
                float(group["lr"]) for group in optimizer.param_groups
            ],
        }

    compatible_shape = (
        saved_embedding.ndim == current_embedding.ndim
        and saved_embedding.shape[1:] == current_embedding.shape[1:]
        and saved_embedding.shape[0] < current_embedding.shape[0]
    )
    if not allow_item_vocabulary_extension or not compatible_shape:
        raise RuntimeError(
            "resume item embedding shape differs: "
            f"checkpoint={tuple(saved_embedding.shape)} "
            f"current={tuple(current_embedding.shape)}; pass "
            "--allow-item-vocabulary-extension only when new items were appended"
        )
    if checkpoint_vocabulary is None:
        raise RuntimeError(
            "cannot verify an item-vocabulary extension because the checkpoint "
            "does not record config.item_vocabulary"
        )
    new_items = {
        name: index
        for name, index in item_vocabulary.items()
        if name not in checkpoint_vocabulary
    }
    if not new_items or any(
        index < saved_embedding.shape[0] for index in new_items.values()
    ):
        raise RuntimeError(
            "new item vocabulary entries must use IDs appended after all "
            "checkpoint embedding rows"
        )

    expanded_embedding = current_embedding.clone()
    expanded_embedding[: saved_embedding.shape[0]].copy_(saved_embedding)
    saved_model[embedding_key] = expanded_embedding
    model.load_state_dict(saved_model, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    saved_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if optimizer_lrs is not None:
        if len(optimizer_lrs) != len(optimizer.param_groups):
            raise RuntimeError(
                "resume LR override group count differs: "
                f"checkpoint={len(optimizer.param_groups)} "
                f"requested={len(optimizer_lrs)}"
            )
        for group, lr in zip(optimizer.param_groups, optimizer_lrs):
            group["lr"] = float(lr)

    embedding_parameter = dict(model.named_parameters())[embedding_key]
    optimizer_state = optimizer.state.get(embedding_parameter, {})
    for name, value in list(optimizer_state.items()):
        if torch.is_tensor(value) and value.shape == saved_embedding.shape:
            expanded = value.new_zeros(current_embedding.shape)
            expanded[: saved_embedding.shape[0]].copy_(value)
            optimizer_state[name] = expanded
    return {
        "item_vocabulary_extended": True,
        "old_embedding_rows": saved_embedding.shape[0],
        "new_embedding_rows": current_embedding.shape[0],
        "new_items": dict(sorted(new_items.items(), key=lambda pair: pair[1])),
        "optimizer_saved_lrs": saved_lrs,
        "optimizer_active_lrs": [
            float(group["lr"]) for group in optimizer.param_groups
        ],
    }


def _start_profile_range(enabled, device):
    if not enabled:
        return None
    torch.cuda.synchronize(device)
    return perf_counter()


def _finish_profile_range(timings, name, started, device):
    if started is None:
        return
    torch.cuda.synchronize(device)
    timings[name] = timings.get(name, 0.0) + perf_counter() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--train-split", choices=("train", "pilot"), default="train",
        help="pilot is intended only for small controlled diagnostics such as S11",
    )
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--pixel-vae", required=True)
    parser.add_argument(
        "--latent-normalization", choices=("none", "pixel-vae"),
        help="Fresh M3-Simple defaults to fixed Pixel VAE mean/std; resume/warm-start inherits checkpoint scale",
    )
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument(
        "--warm-start",
        help="Load model weights without optimizer state; intended for compatible architecture changes",
    )
    parser.add_argument(
        "--warm-start-enable-qk-rms-norm", action="store_true",
        help="Strictly migrate a no-QK Simple checkpoint; reset optimizer and inherit step",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-index")
    parser.add_argument(
        "--chunk-cache-root",
        help="Optional M3 chunk-cache root; requires a portable chunk-cache window index",
    )
    parser.add_argument(
        "--appearance-counterfactual",
        action="store_true",
        help="group complete S11 skin variants by identical trajectory/start/target",
    )
    parser.add_argument("--counterfactual-variants", type=int, default=4)
    parser.add_argument(
        "--counterfactual-dataset-root",
        help=(
            "Optional S11 root mixed with the ordinary --dataset-root while "
            "preserving complete appearance groups"
        ),
    )
    parser.add_argument("--counterfactual-window-index")
    parser.add_argument(
        "--counterfactual-chunk-cache-root",
        help="Optional independent chunk-cache root for the grouped S11 dataset",
    )
    parser.add_argument(
        "--counterfactual-step-probability",
        type=float,
        default=0.0,
        help="Fraction of optimizer steps drawn from grouped S11 data",
    )
    parser.add_argument(
        "--counterfactual-player-mask-probability",
        type=float,
        default=1.0,
        help="Player-region flow-focus probability on grouped S11 steps",
    )
    parser.add_argument(
        "--counterfactual-random-timesteps",
        action="store_true",
        help=(
            "Train grouped S11 variants at shared ordinary random diffusion times "
            "instead of forcing every future block to the pure-noise endpoint"
        ),
    )
    parser.add_argument("--val-window-index")
    parser.add_argument("--health-focus-index")
    parser.add_argument(
        "--health-focus-oversample",
        type=int,
        default=1,
        help="Total sampling multiplicity for windows containing a non-full-health target",
    )
    parser.add_argument("--resume")
    parser.add_argument(
        "--override-resume-lr",
        action="store_true",
        help=(
            "After restoring AdamW state, replace checkpoint learning rates with "
            "the rates constructed from --lr and --unfrozen-base-lr-scale"
        ),
    )
    parser.add_argument(
        "--train-sampler-seed",
        type=int,
        help=(
            "Independent shuffled-training sampler seed; defaults to --seed so "
            "validation/probe randomness can remain fixed across a data-order restart"
        ),
    )
    parser.add_argument(
        "--allow-item-vocabulary-extension",
        action="store_true",
        help=(
            "Allow --resume when the dataset only appends new item IDs; preserves "
            "old embedding rows and optimizer moments and zero-initializes moments "
            "for new rows"
        ),
    )
    parser.add_argument(
        "--deep-condition-reinjection",
        action="store_true",
        help="Reinject aligned raster, actor, action, resident state, and HUD conditions at every DiT block",
    )
    parser.add_argument(
        "--simple-m3",
        action="store_true",
        help=(
            "Use one fused scene encoder, target-state AdaLN, and one compact "
            "player-reference attention"
        ),
    )
    parser.add_argument(
        "--view-aware-appearance",
        action="store_true",
        help="Warp dense four-view resident references into masked per-block appearance adapters",
    )
    parser.add_argument(
        "--detail-preserving-appearance",
        action="store_true",
        help="Preserve 2x2 sub-patch resident appearance and use nonattenuating occupancy gates",
    )
    parser.add_argument(
        "--entity-reference-attention",
        action="store_true",
        help="Cross-attend native-resolution per-resident RGBA reference tokens inside coarse ROIs",
    )
    parser.add_argument(
        "--unified-player-reference",
        action="store_true",
        help=(
            "Use one input-level all-view player reference adapter; replaces pooled, "
            "view-warped and per-block reference appearance paths"
        ),
    )
    parser.add_argument(
        "--unified-reference-reinject-blocks",
        nargs="*",
        type=int,
        default=(),
        metavar="BLOCK",
        help=(
            "Reuse the single unified reference adapter before selected zero-indexed "
            "DiT blocks; additional residual gates start at zero"
        ),
    )
    parser.add_argument(
        "--player-reference-token-grid",
        nargs=2,
        type=int,
        default=(8, 4),
        metavar=("HEIGHT", "WIDTH"),
        help="Spatial token grid retained from each native 256x128 RGBA player view",
    )
    parser.add_argument(
        "--player-reference-position-encoding",
        action="store_true",
        help="Add fixed 2D source coordinates to each per-view appearance token",
    )
    parser.add_argument(
        "--geometry-aware-player-reference",
        action="store_true",
        help=(
            "Bias unified reference attention by camera-facing view and matching "
            "source/target local player coordinates"
        ),
    )
    parser.add_argument(
        "--freeze-base-for-appearance",
        action="store_true",
        help="Stage-one training: update only the new dense appearance modules",
    )
    parser.add_argument(
        "--freeze-base-for-reference",
        action="store_true",
        help="Stage-one training: update only native-resolution reference encoder and adapters",
    )
    parser.add_argument(
        "--appearance-unfreeze-last-spatial-blocks",
        type=int,
        default=0,
        help=(
            "With --freeze-base-for-appearance, also train the spatial attention, spatial "
            "MLP/AdaLN in the last N DiT blocks and the final output layer"
        ),
    )
    parser.add_argument(
        "--reference-unfreeze-last-spatial-blocks",
        type=int,
        default=0,
        help=(
            "With --freeze-base-for-reference, also train the final N spatial DiT "
            "blocks and output layer"
        ),
    )
    parser.add_argument("--context-frames", type=int, default=65)
    parser.add_argument("--cache-frames", type=int, default=64)
    parser.add_argument("--block-frames", type=int, default=8)
    parser.add_argument(
        "--qk-rms-norm", action="store_true",
        help="Enable QK RMSNorm in the spatial and temporal DiT attention blocks",
    )
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--voxel-channels", type=int, default=32)
    parser.add_argument("--condition-dim", type=int, default=256)
    parser.add_argument("--actor-channels", type=int, default=16)
    parser.add_argument("--target-views-per-window", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader workers per DDP rank (4 means 32 total workers on 8 GPUs)",
    )
    parser.add_argument(
        "--prefetch-factor", type=int, default=2,
        help="Batches prefetched by each DataLoader worker (used only when --workers > 0)",
    )
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--validate-every", type=int, default=1000)
    parser.add_argument("--visualize-every", type=int, default=1000)
    parser.add_argument("--visualization-denoising-steps", type=int, default=20)
    parser.add_argument("--val-batches", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--loss-mode",
        choices=("combined", "flow"),
        default="combined",
        help=(
            "combined uses the configured auxiliary losses; flow forces all "
            "decoded-pixel, identity, and counterfactual auxiliary weights to zero"
        ),
    )
    parser.add_argument(
        "--profile-local-step",
        type=int,
        default=0,
        help=(
            "time the Nth optimizer step after resume, print a synchronized stage "
            "breakdown, then exit without writing a checkpoint; 0 disables profiling"
        ),
    )
    parser.add_argument(
        "--latent-entity-region-upweight",
        type=float,
        default=0.0,
        help="Extra latent flow weight inside entity masks; 0 keeps flow loss uniform",
    )
    parser.add_argument(
        "--latent-player-region-upweight",
        type=float,
        default=0.0,
        help="Extra flow weight inside visible human-player masks",
    )
    parser.add_argument(
        "--player-mask-probability",
        type=float,
        default=0.5,
        help="Probability of applying the player-region flow upweight per sample",
    )
    parser.add_argument(
        "--pixel-loss-frames",
        type=int,
        default=2,
        help="Entity-rich and HP-informative future frames decoded per target view",
    )
    parser.add_argument(
        "--pixel-frame-selection",
        choices=("mixed", "player", "player_unique"),
        default="mixed",
        help=(
            "Select decoded frames using the legacy mixed rule, a visible-player "
            "draw, or distinct visible-player frames with empty-mask fallback"
        ),
    )
    parser.add_argument("--entity-pixel-l1-weight", type=float, default=0.5)
    parser.add_argument("--entity-pixel-edge-weight", type=float, default=0.2)
    parser.add_argument("--player-pixel-l1-weight", type=float, default=0.0)
    parser.add_argument("--player-pixel-edge-weight", type=float, default=0.0)
    parser.add_argument("--health-pixel-l1-weight", type=float, default=1.0)
    parser.add_argument("--damaged-health-upweight", type=float, default=4.0)
    parser.add_argument("--player-identity-checkpoint")
    parser.add_argument("--player-identity-loss-weight", type=float, default=0.0)
    parser.add_argument("--player-identity-margin", type=float, default=0.2)
    parser.add_argument("--player-identity-negative-weight", type=float, default=0.5)
    parser.add_argument("--counterfactual-player-difference-weight", type=float, default=0.0)
    parser.add_argument(
        "--mask-prefix-player-probability", type=float, default=0.0,
        help="replace visible other-player pixels in the observed prefix with mid-gray",
    )
    parser.add_argument(
        "--counterfactual-mask-prefix-player-probability",
        type=float,
        default=1.0,
        help="prefix-player masking probability on grouped S11 steps",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--unfrozen-base-lr-scale",
        type=float,
        default=1.0,
        help="Learning-rate multiplier for selectively unfrozen pretrained DiT parameters",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project", default="plot-m3")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument(
        "--checkpoint-staging-dir",
        help="Local/PFS directory used to serialize before copying to output (also PLOT_CHECKPOINT_STAGING_DIR)",
    )
    parser.add_argument(
        "--checkpoint-errors",
        choices=("warn", "raise"),
        default="warn",
        help="Keep training and record failures, or stop if a checkpoint cannot be copied",
    )
    args = parser.parse_args()
    if args.workers < 0:
        parser.error("--workers must be nonnegative")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor must be positive")
    if sum(bool(path) for path in (args.resume, args.warm_start, args.backbone_checkpoint)) > 1:
        parser.error("--resume, --warm-start, and --backbone-checkpoint are mutually exclusive")
    if args.override_resume_lr and not args.resume:
        parser.error("--override-resume-lr requires --resume")
    if args.train_sampler_seed is not None and args.train_sampler_seed < 0:
        parser.error("--train-sampler-seed must be nonnegative")
    if args.warm_start_enable_qk_rms_norm:
        if not (args.warm_start and args.simple_m3 and args.qk_rms_norm):
            parser.error("QK migration requires --warm-start, --simple-m3 and --qk-rms-norm")
        if args.loss_mode != "flow" or args.freeze_base_for_appearance or args.freeze_base_for_reference:
            parser.error("QK migration requires full-network flow training")
        if args.allow_item_vocabulary_extension:
            parser.error("QK migration cannot also extend the item vocabulary")
    if args.freeze_base_for_appearance and not args.view_aware_appearance:
        parser.error("--freeze-base-for-appearance requires --view-aware-appearance")
    if args.detail_preserving_appearance and not args.view_aware_appearance:
        parser.error("--detail-preserving-appearance requires --view-aware-appearance")
    if args.freeze_base_for_appearance and args.resume:
        parser.error("use --warm-start for staged appearance training")
    if args.freeze_base_for_reference and not (
        args.entity_reference_attention or args.unified_player_reference
    ):
        parser.error(
            "--freeze-base-for-reference requires --entity-reference-attention "
            "or --unified-player-reference"
        )
    if (
        args.freeze_base_for_reference
        and args.resume
        and not args.unified_player_reference
    ):
        parser.error("use --warm-start for staged legacy reference training")
    if args.freeze_base_for_reference and args.freeze_base_for_appearance:
        parser.error("choose only one staged-freezing mode")
    if args.unified_player_reference and (
        args.entity_reference_attention
        or args.view_aware_appearance
        or args.detail_preserving_appearance
    ):
        parser.error(
            "--unified-player-reference replaces all legacy appearance/reference flags"
        )
    if args.simple_m3 and (
        args.deep_condition_reinjection
        or args.view_aware_appearance
        or args.detail_preserving_appearance
        or args.entity_reference_attention
        or args.geometry_aware_player_reference
        or args.unified_reference_reinject_blocks
    ):
        parser.error(
            "--simple-m3 is incompatible with legacy, geometry-aware, or repeated adapters"
        )
    if args.simple_m3 and args.actor_channels < 3:
        parser.error("--simple-m3 requires --actor-channels >= 3")
    if args.unified_reference_reinject_blocks and not args.unified_player_reference:
        parser.error(
            "--unified-reference-reinject-blocks requires --unified-player-reference"
        )
    if args.geometry_aware_player_reference and not args.unified_player_reference:
        parser.error(
            "--geometry-aware-player-reference requires --unified-player-reference"
        )
    if any(size < 1 for size in args.player_reference_token_grid):
        parser.error("--player-reference-token-grid dimensions must be positive")
    if len(set(args.unified_reference_reinject_blocks)) != len(
        args.unified_reference_reinject_blocks
    ) or any(
        index < 0 or index >= args.depth
        for index in args.unified_reference_reinject_blocks
    ):
        parser.error(
            "--unified-reference-reinject-blocks must contain unique valid block indices"
        )
    if args.appearance_unfreeze_last_spatial_blocks and not args.freeze_base_for_appearance:
        parser.error(
            "--appearance-unfreeze-last-spatial-blocks requires "
            "--freeze-base-for-appearance"
        )
    if args.reference_unfreeze_last_spatial_blocks and not args.freeze_base_for_reference:
        parser.error(
            "--reference-unfreeze-last-spatial-blocks requires --freeze-base-for-reference"
        )
    if not 0 <= args.appearance_unfreeze_last_spatial_blocks <= args.depth:
        parser.error("--appearance-unfreeze-last-spatial-blocks must be between 0 and depth")
    if not 0 <= args.reference_unfreeze_last_spatial_blocks <= args.depth:
        parser.error("--reference-unfreeze-last-spatial-blocks must be between 0 and depth")
    if not 0 <= args.player_mask_probability <= 1:
        parser.error("--player-mask-probability must be between 0 and 1")
    if not 0 <= args.mask_prefix_player_probability <= 1:
        parser.error("--mask-prefix-player-probability must be between 0 and 1")
    if not 0 <= args.counterfactual_player_mask_probability <= 1:
        parser.error("--counterfactual-player-mask-probability must be between 0 and 1")
    if not 0 <= args.counterfactual_mask_prefix_player_probability <= 1:
        parser.error(
            "--counterfactual-mask-prefix-player-probability must be between 0 and 1"
        )
    if args.appearance_counterfactual and args.target_views_per_window != 1:
        parser.error("--appearance-counterfactual requires --target-views-per-window 1")
    if not 0 <= args.counterfactual_step_probability <= 1:
        parser.error("--counterfactual-step-probability must be between 0 and 1")
    if bool(args.counterfactual_dataset_root) != bool(args.counterfactual_step_probability):
        parser.error(
            "--counterfactual-dataset-root and a positive "
            "--counterfactual-step-probability must be used together"
        )
    if args.appearance_counterfactual and args.counterfactual_dataset_root:
        parser.error(
            "use either dedicated --appearance-counterfactual training or mixed "
            "--counterfactual-dataset-root training"
        )
    if (
        args.loss_mode == "combined"
        and args.counterfactual_player_difference_weight > 0
        and not (args.appearance_counterfactual or args.counterfactual_dataset_root)
    ):
        parser.error("counterfactual difference loss requires grouped S11 data")
    if args.unfrozen_base_lr_scale <= 0:
        parser.error("--unfrozen-base-lr-scale must be positive")
    if args.profile_local_step < 0:
        parser.error("--profile-local-step must be nonnegative")
    if args.profile_local_step and args.gradient_accumulation != 1:
        parser.error("--profile-local-step currently requires --gradient-accumulation 1")
    if (
        args.loss_mode == "combined"
        and args.player_identity_loss_weight > 0
        and not args.player_identity_checkpoint
    ):
        parser.error("--player-identity-loss-weight requires --player-identity-checkpoint")
    if (
        min(
            args.steps,
            args.save_every,
            args.validate_every,
            args.target_views_per_window,
            args.batch_size,
            args.gradient_accumulation,
        )
        < 1
    ):
        parser.error("steps, save interval, anchors and batch size must be positive")
    if not 1 <= args.pixel_loss_frames < args.context_frames:
        parser.error("pixel-loss-frames must be within the future-frame count")
    if args.health_focus_oversample < 1:
        parser.error("health-focus-oversample must be positive")
    if args.health_focus_oversample > 1 and not args.health_focus_index:
        parser.error("health-focus-oversample above 1 requires --health-focus-index")
    if min(
        args.latent_entity_region_upweight,
        args.latent_player_region_upweight,
        args.entity_pixel_l1_weight,
        args.entity_pixel_edge_weight,
        args.player_pixel_l1_weight,
        args.player_pixel_edge_weight,
        args.health_pixel_l1_weight,
        args.damaged_health_upweight,
        args.player_identity_loss_weight,
        args.player_identity_margin,
        args.player_identity_negative_weight,
        args.counterfactual_player_difference_weight,
    ) < 0:
        parser.error("region and pixel loss weights must be nonnegative")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if world > 1 else args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")
    if device.type != "cuda":
        parser.error(
            "raw voxel rasterization needs CUDA; use CPU contract tests for smoke validation"
        )
    torch.cuda.set_device(device)
    capability = torch.cuda.get_device_capability(device)
    compiled_arch = f"sm_{capability[0]}{capability[1]}"
    if capability >= (10, 0) and compiled_arch not in torch.cuda.get_arch_list():
        parser.error(
            f"PyTorch {torch.__version__} was not compiled for {compiled_arch}; "
            "run scripts/check_m3_environment.py on the B200 node"
        )
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.manual_seed(args.seed + rank)
    dataset_class = (
        AppearanceCounterfactualRendererDataset
        if args.appearance_counterfactual
        else TextAgentRendererDataset
    )
    dataset_kwargs = dict(
        split=args.train_split,
        context_frames=args.context_frames,
        window_index=args.window_index,
        targets_per_window=args.target_views_per_window,
        entity_region_upweight=args.latent_entity_region_upweight,
        health_focus_index=args.health_focus_index,
        health_focus_oversample=args.health_focus_oversample,
        chunk_cache_root=args.chunk_cache_root,
    )
    if args.appearance_counterfactual:
        dataset_kwargs["variants_per_group"] = args.counterfactual_variants
    dataset = dataset_class(
        args.dataset_root,
        args.vocabulary,
        **dataset_kwargs,
    )
    train_sampler_seed = (
        args.seed if args.train_sampler_seed is None else args.train_sampler_seed
    )
    sampler = (
        DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True,
            seed=train_sampler_seed,
        )
        if world > 1
        else None
    )
    worker_kwargs = {"num_workers": args.workers}
    if args.workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        collate_fn=collate_renderer,
        **worker_kwargs,
    )
    counterfactual_dataset = None
    counterfactual_sampler = None
    counterfactual_loader = None
    if args.counterfactual_dataset_root:
        counterfactual_dataset = AppearanceCounterfactualRendererDataset(
            args.counterfactual_dataset_root,
            args.vocabulary,
            split="train",
            context_frames=args.context_frames,
            window_index=args.counterfactual_window_index,
            chunk_cache_root=args.counterfactual_chunk_cache_root,
            targets_per_window=1,
            variants_per_group=args.counterfactual_variants,
            entity_region_upweight=args.latent_entity_region_upweight,
        )
        if counterfactual_dataset.item_vocabulary != dataset.item_vocabulary:
            raise ValueError("ordinary and S11 item vocabularies differ")
        counterfactual_sampler = (
            DistributedSampler(
                counterfactual_dataset,
                num_replicas=world,
                rank=rank,
                shuffle=True,
                seed=args.seed + 1,
            )
            if world > 1
            else None
        )
        counterfactual_loader = DataLoader(
            counterfactual_dataset,
            batch_size=args.batch_size,
            shuffle=counterfactual_sampler is None,
            sampler=counterfactual_sampler,
            collate_fn=collate_renderer,
            **worker_kwargs,
        )
    cfg = RendererArgs(
        dataset.vocabulary.size,
        max(dataset.item_vocabulary.values()) + 1,
        context_frames=args.context_frames,
        cache_frames=args.cache_frames,
        block_frames=args.block_frames,
        qk_rms_norm=args.qk_rms_norm,
        hidden_size=args.hidden_size,
        depth=args.depth,
        num_heads=args.heads,
        voxel_channels=args.voxel_channels,
        condition_dim=args.condition_dim,
        actor_channels=args.actor_channels,
        deep_condition_reinjection=args.deep_condition_reinjection,
        view_aware_appearance=args.view_aware_appearance,
        detail_preserving_appearance=args.detail_preserving_appearance,
        entity_reference_attention=args.entity_reference_attention,
        unified_player_reference=args.unified_player_reference or args.simple_m3,
        player_reference_grid_size=tuple(args.player_reference_token_grid),
        player_reference_position_encoding=args.player_reference_position_encoding,
        geometry_aware_player_reference=args.geometry_aware_player_reference,
        unified_reference_reinject_blocks=tuple(
            args.unified_reference_reinject_blocks
        ),
        simple_conditioning=args.simple_m3,
    )
    with torch.cuda.device(device):
        raw_model = Renderer(cfg).to(device).train()
    if args.backbone_checkpoint:
        report = raw_model.load_2daction_backbone(load_weights(args.backbone_checkpoint))
        if rank == 0:
            print(json.dumps(report))
    checkpoint_path = args.resume or args.warm_start
    checkpoint = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path else None
    )
    latent_normalization = resolve_training_normalization(
        args.latent_normalization, simple_m3=args.simple_m3,
        checkpoint_config=checkpoint.get("config", {}) if checkpoint is not None else None,
    )
    codec = RendererCodec(
        load_weights(args.pixel_vae), latent_normalization=latent_normalization
    ).to(device).eval()
    if rank == 0:
        print(json.dumps({"codec": codec.get_config()}))
    identity_encoder = None
    if args.loss_mode == "combined" and args.player_identity_checkpoint:
        identity_checkpoint = torch.load(
            args.player_identity_checkpoint, map_location="cpu", weights_only=False
        )
        identity_encoder = PlayerIdentityEncoder(
            int(identity_checkpoint.get("embedding_dim", 128))
        ).to(device).eval()
        identity_encoder.crop_size = tuple(identity_checkpoint.get("crop_size", (128, 64)))
        identity_encoder.load_state_dict(identity_checkpoint["model"], strict=True)
        identity_encoder.requires_grad_(False)
    if args.freeze_base_for_reference:
        first_unfrozen_spatial_block = (
            args.depth - args.reference_unfreeze_last_spatial_blocks
        )
        for name, parameter in raw_model.named_parameters():
            trainable = name.startswith((
                "reference_encoder.",
                "core.entity_reference_adapters.",
                "core.unified_reference_adapter.",
            ))
            if args.reference_unfreeze_last_spatial_blocks:
                trainable = trainable or name.startswith("core.final_layer.")
                for block_index in range(first_unfrozen_spatial_block, args.depth):
                    trainable = trainable or name.startswith((
                        f"core.blocks.{block_index}.s_attn.",
                        f"core.blocks.{block_index}.s_mlp.",
                        f"core.blocks.{block_index}.s_adaLN_modulation.",
                    ))
            parameter.requires_grad_(trainable)
    elif args.freeze_base_for_appearance:
        first_unfrozen_spatial_block = (
            args.depth - args.appearance_unfreeze_last_spatial_blocks
        )
        for name, parameter in raw_model.named_parameters():
            trainable = name.startswith(
                (
                    "core.appearance_condition_embedder.",
                    "core.appearance_detail_embedder.",
                    "core.appearance_detail_reinjectors.",
                    "reference_encoder.",
                    "core.entity_reference_adapters.",
                    "core.appearance_reinjectors.",
                )
            )
            if args.appearance_unfreeze_last_spatial_blocks:
                trainable = trainable or name.startswith("core.final_layer.")
                for block_index in range(first_unfrozen_spatial_block, args.depth):
                    trainable = trainable or name.startswith(
                        (
                            f"core.blocks.{block_index}.s_attn.",
                            f"core.blocks.{block_index}.s_mlp.",
                            f"core.blocks.{block_index}.s_adaLN_modulation.",
                        )
                    )
            parameter.requires_grad_(trainable)
    trainable_parameters = [p for p in raw_model.parameters() if p.requires_grad]
    if (
        args.freeze_base_for_reference
        and args.reference_unfreeze_last_spatial_blocks
        and args.unfrozen_base_lr_scale != 1
    ):
        reference_parameters, base_parameters = [], []
        for name, parameter in raw_model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith((
                "reference_encoder.",
                "core.entity_reference_adapters.",
                "core.unified_reference_adapter.",
            )):
                reference_parameters.append(parameter)
            else:
                base_parameters.append(parameter)
        optimizer = torch.optim.AdamW([
            {"params": reference_parameters, "lr": args.lr},
            {"params": base_parameters, "lr": args.lr * args.unfrozen_base_lr_scale},
        ])
    else:
        optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr)
    requested_optimizer_lrs = [
        float(group["lr"]) for group in optimizer.param_groups
    ]
    if rank == 0:
        print(json.dumps({
            "trainable_parameters": sum(p.numel() for p in trainable_parameters),
            "total_parameters": sum(p.numel() for p in raw_model.parameters()),
        }))
    start_step = 0
    qk_migration = None
    if args.resume:
        resume_report = load_renderer_resume(
            raw_model,
            optimizer,
            checkpoint,
            dataset.item_vocabulary,
            allow_item_vocabulary_extension=args.allow_item_vocabulary_extension,
            optimizer_lrs=(
                requested_optimizer_lrs if args.override_resume_lr else None
            ),
        )
        start_step = int(checkpoint["step"])
        if rank == 0:
            print(json.dumps({"resume": str(args.resume), **resume_report}))
    elif args.warm_start_enable_qk_rms_norm:
        qk_migration = load_qk_warm_start(
            raw_model, checkpoint,
            item_vocabulary=dataset.item_vocabulary,
            class_to_raw=dataset.vocabulary.class_to_raw,
        )
        start_step = qk_migration["parent_step"]
        if args.steps <= start_step:
            raise ValueError("--steps is the cumulative target and must exceed the parent step")
        if rank == 0:
            print(json.dumps({"warm_start": args.warm_start, "lr": args.lr, **qk_migration}))
    elif args.warm_start:
        warm_state = dict(checkpoint["model"])
        migrated_parameters = []
        if args.unified_player_reference:
            # Reuse the trained first legacy reference adapter for the single
            # input-level adapter. Shapes are identical; only routing retains
            # the four-view axis instead of pre-averaging it.
            old_prefix = "core.entity_reference_adapters.0."
            new_prefix = "core.unified_reference_adapter."
            for key, value in list(warm_state.items()):
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    warm_state[new_key] = value
                    migrated_parameters.append(f"{key}->{new_key}")
            legacy_prefixes = (
                "resident_encoder.skin_view_encoder.",
                "resident_encoder.appearance_direction_embedding",
                "resident_encoder.skin_view_fusion.",
                "appearance_spatial_encoder.",
                "core.appearance_condition_embedder.",
                "core.appearance_detail_embedder.",
                "core.appearance_detail_reinjectors.",
                "core.appearance_reinjectors.",
                "core.entity_reference_adapters.",
            )
            warm_state = {
                key: value for key, value in warm_state.items()
                if not key.startswith(legacy_prefixes)
            }
        incompatible = raw_model.load_state_dict(warm_state, strict=False)
        allowed_missing = (
            "core.hud_condition_embedder.",
            "core.condition_reinjectors.",
            "core.appearance_condition_embedder.",
            "core.appearance_detail_embedder.",
            "core.appearance_detail_reinjectors.",
            "reference_encoder.",
            "core.entity_reference_adapters.",
            "core.unified_reference_adapter.",
            "core.appearance_reinjectors.",
        )
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith(allowed_missing)
        ]
        if invalid_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "incompatible warm start: "
                f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
            )
        start_step = int(checkpoint.get("step", 0))
        if rank == 0:
            print(json.dumps({
                "warm_start": str(args.warm_start),
                "step": start_step,
                "initialized_parameters": incompatible.missing_keys,
                "migrated_parameters": migrated_parameters,
            }))
    model = (
        DistributedDataParallel(raw_model, device_ids=[local_rank], broadcast_buffers=False)
        if world > 1
        else raw_model
    )
    output = Path(args.output_dir)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()
    run_config = {
        "renderer": asdict(cfg),
        "codec": codec.get_config(),
        "training": vars(args),
        "item_vocabulary": dataset.item_vocabulary,
        "class_to_raw": dataset.vocabulary.class_to_raw,
    }
    if qk_migration is not None:
        run_config["qk_migration"] = qk_migration
    if rank == 0:
        (output / "config.json").write_text(json.dumps(run_config, indent=2))
    run = None
    if rank == 0 and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.wandb_name,
            mode=args.wandb_mode,
            dir=str(output),
            config={
                **run_config,
                "world_size": world,
                "effective_source_window_batch_size": (
                    args.batch_size * world * args.gradient_accumulation
                ),
                "effective_view_batch_size": (
                    args.batch_size
                    * args.target_views_per_window
                    * getattr(dataset, "group_size", 1)
                    * world
                    * args.gradient_accumulation
                ),
            },
        )
        (output / "wandb_run.json").write_text(
            json.dumps(
                {
                    "id": run.id,
                    "url": run.url,
                    "entity": args.wandb_entity,
                    "project": args.wandb_project,
                },
                indent=2,
            )
        )
    val_loader = None
    probes = []
    if args.val_window_index:
        val_dataset = TextAgentRendererDataset(
            args.dataset_root,
            args.vocabulary,
            split="val_id",
            context_frames=args.context_frames,
            window_index=args.val_window_index,
            chunk_cache_root=args.chunk_cache_root,
            entity_region_upweight=args.latent_entity_region_upweight,
        )
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world, rank=rank, shuffle=False)
            if world > 1
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            shuffle=False,
            collate_fn=collate_renderer,
            **worker_kwargs,
        )
        if rank == 0 and args.visualize_every:
            probes = select_renderer_probes(val_dataset)
            save_probe_manifest(probes, output / "visualizations" / "probes.json")
    elif rank == 0 and args.visualize_every:
        print(
            "warning: --visualize-every requires --val-window-index; visualization disabled",
            flush=True,
        )
    iterator = iter(loader)
    counterfactual_iterator = (
        iter(counterfactual_loader) if counterfactual_loader is not None else None
    )
    epoch = 0
    counterfactual_epoch = 0
    auxiliary_weights = effective_auxiliary_loss_weights(args)
    train_total_loss_ema = None
    train_total_loss_ema_decay = 0.99
    loss_kwargs = {
        "frames_per_sample": args.pixel_loss_frames,
        "pixel_frame_selection": args.pixel_frame_selection,
        "entity_pixel_l1_weight": auxiliary_weights["entity_pixel_l1_weight"],
        "entity_pixel_edge_weight": auxiliary_weights["entity_pixel_edge_weight"],
        "player_pixel_l1_weight": auxiliary_weights["player_pixel_l1_weight"],
        "player_pixel_edge_weight": auxiliary_weights["player_pixel_edge_weight"],
        "health_pixel_l1_weight": auxiliary_weights["health_pixel_l1_weight"],
        "damaged_health_upweight": args.damaged_health_upweight,
        "identity_encoder": identity_encoder,
        "player_identity_loss_weight": auxiliary_weights["player_identity_loss_weight"],
        "player_identity_margin": args.player_identity_margin,
        "player_identity_negative_weight": args.player_identity_negative_weight,
        "counterfactual_group_size": getattr(dataset, "group_size", 1),
        "counterfactual_player_difference_weight": (
            auxiliary_weights["counterfactual_player_difference_weight"]
        ),
        "counterfactual_pure_noise": not args.counterfactual_random_timesteps,
    }
    for step in range(start_step + 1, args.steps + 1):
        profile_this_step = args.profile_local_step == step - start_step
        profile_timings = {}
        profile_step_started = _start_profile_range(profile_this_step, device)
        optimizer.zero_grad(set_to_none=True)
        accumulated = {name: 0.0 for name in LOSS_NAMES}
        counterfactual_step = bool(
            counterfactual_loader is not None
            and use_counterfactual_step(
                step, args.counterfactual_step_probability, args.seed
            )
        )
        active_dataset = counterfactual_dataset if counterfactual_step else dataset
        for micro in range(args.gradient_accumulation):
            data_started = perf_counter() if profile_this_step else None
            if counterfactual_step:
                try:
                    batch = next(counterfactual_iterator)
                except StopIteration:
                    counterfactual_epoch += 1
                    if counterfactual_sampler is not None:
                        counterfactual_sampler.set_epoch(counterfactual_epoch)
                    counterfactual_iterator = iter(counterfactual_loader)
                    batch = next(counterfactual_iterator)
            else:
                try:
                    batch = next(iterator)
                except StopIteration:
                    epoch += 1
                    if sampler is not None:
                        sampler.set_epoch(epoch)
                    iterator = iter(loader)
                    batch = next(iterator)
            if data_started is not None:
                profile_timings["data_wait_cpu"] = perf_counter() - data_started
            transfer_started = _start_profile_range(profile_this_step, device)
            rgb = batch["rgb"].to(device)
            prefix_mask_probability = (
                args.counterfactual_mask_prefix_player_probability
                if counterfactual_step or args.appearance_counterfactual
                else args.mask_prefix_player_probability
            )
            if prefix_mask_probability > 0:
                group_size = getattr(active_dataset, "group_size", 1)
                group_count = len(rgb) // group_size
                gate = (torch.rand(group_count, device=device)
                        < prefix_mask_probability)
                gate = gate.repeat_interleave(group_size)
                prefix_player = batch["player_region_mask"][:, :1].to(device).bool()
                rgb = rgb.clone()
                rgb[:, :1] = torch.where(
                    gate[:, None, None, None, None] & prefix_player,
                    rgb.new_tensor(0.5), rgb[:, :1],
                )
            _finish_profile_range(
                profile_timings, "rgb_to_gpu_and_prefix", transfer_started, device
            )
            encode_started = _start_profile_range(profile_this_step, device)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"),
            ):
                latent = codec.encode(rgb)
            _finish_profile_range(profile_timings, "vae_encode", encode_started, device)
            loss_forward_started = _start_profile_range(profile_this_step, device)
            conditions = {k: v.to(device) for k, v in batch["conditions"].items()}
            sync = (
                model.no_sync()
                if world > 1 and micro + 1 < args.gradient_accumulation
                else contextlib.nullcontext()
            )
            with sync:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                    losses = renderer_training_losses(
                        model,
                        codec,
                        latent,
                        conditions,
                        rgb,
                        batch["pixel_region_mask"].to(device),
                        player_region_mask=batch["player_region_mask"].to(device),
                        region_weight=build_flow_region_weight(
                            batch["region_weight"].to(device),
                            batch["player_region_mask"].to(device),
                            player_upweight=args.latent_player_region_upweight,
                            probability=(
                                args.counterfactual_player_mask_probability
                                if counterfactual_step or args.appearance_counterfactual
                                else args.player_mask_probability
                            ),
                            randomize=True,
                        ),
                        player_identity_mask=batch["player_identity_mask"].to(device),
                        player_identity_frame=batch["player_identity_frame"].to(device),
                        player_identity_slot=batch["player_identity_slot"].to(device),
                        player_identity_valid=batch["player_identity_valid"].to(device),
                        profile_timings=profile_timings if profile_this_step else None,
                        **{
                            **loss_kwargs,
                            "counterfactual_group_size": getattr(
                                active_dataset, "group_size", 1
                            ),
                            "counterfactual_player_difference_weight": (
                                auxiliary_weights[
                                    "counterfactual_player_difference_weight"
                                ]
                                if counterfactual_step or args.appearance_counterfactual
                                else 0.0
                            ),
                        },
                    )
                    loss = losses["total_loss"] / args.gradient_accumulation
                _finish_profile_range(
                    profile_timings, "loss_forward_total", loss_forward_started, device
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite M3 loss at step {step}")
                backward_started = _start_profile_range(profile_this_step, device)
                loss.backward()
                _finish_profile_range(
                    profile_timings, "backward", backward_started, device
                )
            for name in LOSS_NAMES:
                accumulated[name] += float(losses[name].detach()) / args.gradient_accumulation
        clip_started = _start_profile_range(profile_this_step, device)
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
        _finish_profile_range(profile_timings, "grad_clip", clip_started, device)
        optimizer_started = _start_profile_range(profile_this_step, device)
        optimizer.step()
        _finish_profile_range(profile_timings, "optimizer_step", optimizer_started, device)
        if profile_this_step:
            _finish_profile_range(
                profile_timings, "step_wall_gpu_synchronized", profile_step_started, device
            )
            loss_forward = profile_timings["loss_forward_total"]
            nested = (
                profile_timings.get("m3_forward", 0.0)
                + profile_timings.get("vae_decode_for_loss", 0.0)
            )
            profile_timings["loss_forward_other"] = max(0.0, loss_forward - nested)
            exclusive_names = (
                "rgb_to_gpu_and_prefix",
                "vae_encode",
                "m3_forward",
                "vae_decode_for_loss",
                "loss_forward_other",
                "backward",
                "grad_clip",
                "optimizer_step",
            )
            exclusive_total = sum(
                profile_timings.get(name, 0.0) for name in exclusive_names
            )
            wall_total = profile_timings["step_wall_gpu_synchronized"]
            accounted = exclusive_total + profile_timings.get("data_wait_cpu", 0.0)
            profile_timings["unattributed_wall"] = max(0.0, wall_total - accounted)
            report_names = ("data_wait_cpu", *exclusive_names, "unattributed_wall")
            report = {
                "profiled_global_step": step,
                "profiled_local_step": step - start_step,
                "loss_mode": args.loss_mode,
                "batch_size": args.batch_size,
                "seconds": profile_timings,
                "wall_percent": {
                    name: 100.0 * profile_timings.get(name, 0.0) / max(wall_total, 1e-12)
                    for name in report_names
                },
            }
            print("PROFILE_STEP_TIMING " + json.dumps(report, sort_keys=True), flush=True)
            if rank == 0 and run:
                run.finish()
            if world > 1:
                dist.destroy_process_group()
            return
        reduced = torch.tensor([accumulated[name] for name in LOSS_NAMES], device=device)
        if world > 1:
            dist.all_reduce(reduced)
            reduced /= world
        if rank == 0:
            reduced_total_loss = float(reduced[0])
            train_total_loss_ema = (
                reduced_total_loss
                if train_total_loss_ema is None
                else train_total_loss_ema_decay * train_total_loss_ema
                + (1 - train_total_loss_ema_decay) * reduced_total_loss
            )
        log_now = step == 1 or step % args.log_every == 0
        if log_now:
            local_memory = torch.tensor(
                [
                    torch.cuda.memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_allocated(device) / 2**30,
                    torch.cuda.max_memory_reserved(device) / 2**30,
                ],
                device=device,
            )
            if world > 1:
                memory_by_rank = [torch.zeros_like(local_memory) for _ in range(world)]
                dist.all_gather(memory_by_rank, local_memory)
                memory_by_rank = torch.stack(memory_by_rank)
            else:
                memory_by_rank = local_memory[None]
        if rank == 0 and log_now:
            train_log = {
                **{f"train/{name}": float(reduced[index])
                   for index, name in enumerate(LOSS_NAMES)},
                "train/total_loss_ema_099": train_total_loss_ema,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
                "train/cuda_memory_allocated_max_gib": float(memory_by_rank[:, 0].max()),
                "train/cuda_peak_allocated_max_gib": float(memory_by_rank[:, 1].max()),
                "train/cuda_peak_reserved_max_gib": float(memory_by_rank[:, 2].max()),
                "train/counterfactual_step": float(counterfactual_step),
            }
            memory_text = " ".join(
                f"rank{index}_peak_reserved_gib={float(values[2]):.2f}"
                for index, values in enumerate(memory_by_rank)
            )
            print(
                f"step={step} source={'s11' if counterfactual_step else 'ordinary'} "
                f"loss={train_log['train/total_loss']:.6f} "
                f"flow={train_log['train/flow_loss']:.6f} "
                f"entity_l1={train_log['train/entity_pixel_l1']:.6f} "
                f"health_l1={train_log['train/health_pixel_l1']:.6f} "
                f"identity={train_log['train/player_identity_loss']:.6f} "
                f"identity_sim={train_log['train/player_identity_similarity']:.4f} "
                f"player_l1={train_log['train/player_pixel_l1']:.6f} "
                f"counterfactual={train_log['train/counterfactual_player_difference_loss']:.6f} "
                f"peak_reserved_max_gib={train_log['train/cuda_peak_reserved_max_gib']:.2f} "
                f"{memory_text}",
                flush=True,
            )
            if run:
                run.log(train_log, step=step)
        if val_loader is not None and (step % args.validate_every == 0 or step == args.steps):
            raw_model.eval()
            values = []
            with torch.no_grad():
                for number, val in enumerate(val_loader):
                    if number >= args.val_batches:
                        break
                    rgb = val["rgb"].to(device)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"
                    ):
                        latent = codec.encode(rgb)
                        condition = {k: v.to(device) for k, v in val["conditions"].items()}
                        val_loss_kwargs = dict(
                            loss_kwargs,
                            counterfactual_group_size=1,
                            counterfactual_player_difference_weight=0.0,
                        )
                        val_losses = renderer_training_losses(
                            raw_model,
                            codec,
                            latent,
                            condition,
                            rgb,
                            val["pixel_region_mask"].to(device),
                            player_region_mask=val["player_region_mask"].to(device),
                            region_weight=build_flow_region_weight(
                                val["region_weight"].to(device),
                                val["player_region_mask"].to(device),
                                player_upweight=args.latent_player_region_upweight,
                                probability=args.player_mask_probability,
                                randomize=False,
                            ),
                            player_identity_mask=val["player_identity_mask"].to(device),
                            player_identity_frame=val["player_identity_frame"].to(device),
                            player_identity_slot=val["player_identity_slot"].to(device),
                            player_identity_valid=val["player_identity_valid"].to(device),
                            generator=torch.Generator(device=device).manual_seed(
                                args.seed + number
                            ),
                            **val_loss_kwargs,
                        )
                        values.append(torch.stack([val_losses[name] for name in LOSS_NAMES]))
            metric = (
                torch.stack(values).mean(0)
                if values
                else torch.full((len(LOSS_NAMES),), float("nan"), device=device)
            )
            if world > 1:
                dist.all_reduce(metric)
                metric /= world
            if rank == 0:
                val_metrics = {
                    name: float(metric[index]) for index, name in enumerate(LOSS_NAMES)
                }
                with (output / "validation.jsonl").open("a") as handle:
                    handle.write(json.dumps({"step": step, **val_metrics}) + "\n")
                if run:
                    run.log({f"val/{name}": value for name, value in val_metrics.items()}, step=step)
            raw_model.train()
        visualize = bool(
            args.val_window_index
            and args.visualize_every
            and (step % args.visualize_every == 0 or step == args.steps)
        )
        if visualize:
            # Only rank zero renders; the barrier prevents other ranks from
            # entering the next DDP backward while it owns the model caches.
            if world > 1:
                dist.barrier()
            if rank == 0:
                raw_model.eval()
                visual_log = {}
                visual_dir = output / "visualizations" / f"step_{step:07d}"
                visual_dir.mkdir(parents=True, exist_ok=True)
                for probe_number, probe in enumerate(probes):
                    try:
                        probe_sample = TextAgentRendererDataset.read_window(
                            probe.episode,
                            val_dataset.vocabulary,
                            start=probe.start,
                            target=probe.target,
                            context_frames=args.context_frames,
                        )
                        sample = collate_renderer([probe_sample])
                        video_path = visual_dir / f"{probe.name}.mp4"
                        metrics = render_probe(
                            raw_model,
                            codec,
                            sample,
                            video_path,
                            seed=args.seed + probe_number,
                            denoising_steps=args.visualization_denoising_steps,
                            precision=args.precision,
                        )
                        visual_log.update(
                            {f"visual/{probe.name}_{key}": value for key, value in metrics.items()}
                        )
                        if run:
                            import wandb

                            visual_log[f"visual/{probe.name}"] = wandb.Video(
                                str(video_path),
                                fps=8,
                                format="mp4",
                                caption=(
                                    f"{probe.scenario_id}, target agent{probe.target}, "
                                    f"event={probe.event or 'movement'}"
                                ),
                            )
                    except Exception as error:
                        raw_model.clear_cache()
                        visual_log[f"visual/{probe.name}_failure"] = 1
                        print(
                            f"warning: visualization probe {probe.name} failed: {error}", flush=True
                        )
                try:
                    (visual_dir / "metrics.json").write_text(
                        json.dumps(visual_log, indent=2, default=lambda value: "wandb.Video")
                    )
                    if run:
                        run.log(visual_log, step=step)
                except Exception as error:
                    print(f"warning: visualization logging failed: {error}", flush=True)
                raw_model.train()
            if world > 1:
                dist.barrier()
        save_now = step % args.save_every == 0 or step == args.steps
        if save_now and world > 1:
            dist.barrier()
        checkpoint_error = None
        if rank == 0 and save_now:
            destination = output / f"step_{step:07d}.pt"
            try:
                staged_torch_save(
                    {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "config": run_config,
                    },
                    destination,
                    staging_dir=args.checkpoint_staging_dir,
                )
                if run:
                    run.log({"checkpoint/saved_step": step}, step=step)
            except Exception as error:
                try:
                    record_checkpoint_failure(output, destination, error)
                except Exception as record_error:
                    print(
                        f"warning: could not write checkpoint failure record: {record_error}",
                        flush=True,
                    )
                print(f"warning: checkpoint save failed at step {step}: {error}", flush=True)
                if run:
                    run.log(
                        {
                            "checkpoint/save_failure": 1,
                            "checkpoint/save_failure_message": str(error),
                        },
                        step=step,
                    )
                if args.checkpoint_errors == "raise":
                    checkpoint_error = str(error)
        if save_now and world > 1:
            status = [checkpoint_error]
            dist.broadcast_object_list(status, src=0)
            checkpoint_error = status[0]
        if checkpoint_error is not None:
            raise RuntimeError(f"checkpoint save failed: {checkpoint_error}")
    if world > 1:
        dist.barrier()
    if rank == 0 and run:
        run.finish()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
