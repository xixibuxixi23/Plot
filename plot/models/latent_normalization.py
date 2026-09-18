"""Portable fixed Pixel VAE statistics shared by M3 training and inference."""
from __future__ import annotations

from copy import deepcopy
import math


# From 2daction/datasets/stage2_remote_full_47676/latent_stats.npz, not
# estimated per batch or from the new fixed-skins release. Same frozen VAE.
_PIXEL_VAE_STATS = {
    "mode": "per_channel",
    "mean": [
        -0.7020555138587952, 0.8461390733718872, -0.917613685131073,
        -1.5030848979949951, -1.698622465133667, -1.8949798345565796,
        -1.1478651762008667, 0.6958867311477661, 2.513483762741089,
        1.2163547277450562, 1.9755494594573975, 0.7205855250358582,
        -0.5866942405700684, 0.3998744487762451, -2.7082550525665283,
        -2.014042615890503,
    ],
    "std": [
        2.0923426151275635, 2.398167848587036, 2.862182140350342,
        1.8666865825653076, 2.9419124126434326, 2.9730942249298096,
        2.1236183643341064, 2.4726924896240234, 2.607072353363037,
        2.1573493480682373, 2.7498817443847656, 2.7455546855926514,
        2.5569992065429688, 2.051330804824829, 2.2603418827056885,
        2.367582321166992,
    ],
    "source": "2daction/stage2_remote_full_47676/latent_stats.npz",
    "source_sha256": "2746a106b60e8ac20a2e6bd622e226a6db6d0e3aa4d4c79d5a1a1566d454161a",
    "vae_sha256": "eb634803c94aeea980046961382f2ab67157aa71e92c5a183a35e4f61f8cbc36",
    "sampled_files": 256,
    "frames_per_file": 8,
    "scalar_samples_per_channel": 9437184,
    "seed": 20260823,
}


def pixel_vae_latent_stats():
    """Return a fresh JSON-serializable copy, requiring no shared-disk assets."""
    return deepcopy(_PIXEL_VAE_STATS)


def validate_latent_normalization(stats, channels=16):
    """Canonicalize legacy raw mode or validate fixed per-channel statistics."""
    if stats is None or stats == {"mode": "none"}:
        return {"mode": "none"}
    if not isinstance(stats, dict) or stats.get("mode") != "per_channel":
        raise ValueError("latent normalization requires mode none or per_channel")
    result = deepcopy(stats)
    for key in ("mean", "std"):
        values = result.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != channels:
            raise ValueError(f"latent normalization {key} must have {channels} channels")
        values = [float(value) for value in values]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"latent normalization {key} must be finite")
        if key == "std" and not all(value > 0 for value in values):
            raise ValueError("latent normalization std must be positive")
        result[key] = values
    return result


def normalization_from_run_config(config):
    # Absence means a historical raw-latent checkpoint, including old Simple.
    return validate_latent_normalization(config.get("codec", {}).get("latent_normalization"))


def resolve_training_normalization(mode, *, simple_m3, checkpoint_config=None):
    """Default new Simple runs to normalized latents; inherit on resume/warm start."""
    if mode not in (None, "none", "pixel-vae"):
        raise ValueError(f"unknown latent normalization: {mode}")
    requested = pixel_vae_latent_stats() if mode == "pixel-vae" else {"mode": "none"}
    if checkpoint_config is not None:
        saved = normalization_from_run_config(checkpoint_config)
        # Compare the transform, not provenance labels.
        keys = ("mode", "mean", "std")
        if mode is not None and any(requested.get(k) != saved.get(k) for k in keys):
            raise ValueError(
                "latent normalization differs from checkpoint; start a fresh run "
                "without --resume/--warm-start to change latent scale"
            )
        return saved
    return pixel_vae_latent_stats() if mode is None and simple_m3 else requested
