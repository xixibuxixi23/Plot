"""Strict model-only migration from M3-Simple without QK norm to QK norm."""

from dataclasses import asdict
import json

import torch


def load_qk_warm_start(model, checkpoint, *, item_vocabulary, class_to_raw):
    """Copy all existing tensors; initialize only new attention norm gains.

    This deliberately does not load optimizer, RNG or data iterator state.
    It is an architecture-changing warm start, not an exact resume.
    """
    current_config = asdict(model.cfg)
    saved_config = checkpoint.get("config", {})
    old_renderer = saved_config.get("renderer", {})
    if not current_config.get("simple_conditioning") or not current_config.get("qk_rms_norm"):
        raise ValueError("QK warm start requires M3-Simple with QK RMSNorm enabled")
    if not old_renderer or old_renderer.get("qk_rms_norm", False):
        raise ValueError("parent must record a renderer config with QK RMSNorm disabled")
    old_renderer = {k: v for k, v in old_renderer.items() if k != "qk_rms_norm"}
    new_renderer = {k: v for k, v in current_config.items() if k != "qk_rms_norm"}
    # Checkpoint configs may serialize tuple-valued fields as lists.
    if json.dumps(old_renderer, sort_keys=True) != json.dumps(new_renderer, sort_keys=True):
        raise ValueError("QK warm start permits no other renderer configuration changes")
    for key, expected in (("item_vocabulary", item_vocabulary), ("class_to_raw", class_to_raw)):
        if saved_config.get(key) != expected:
            raise ValueError(f"QK warm start requires identical {key}")
    step = checkpoint.get("step")
    if not isinstance(step, int) or isinstance(step, bool) or step < 1:
        raise ValueError("parent must contain a positive completed training step")

    expected_new = {
        f"core.blocks.{i}.{attention}.{norm}.gamma"
        for i in range(model.cfg.depth)
        for attention in ("s_attn", "t_attn")
        for norm in ("q_rms_norm", "k_rms_norm")
    }
    current = model.state_dict()
    saved = checkpoint["model"]
    missing = set(current) - set(saved)
    unexpected = set(saved) - set(current)
    mismatched = [k for k in set(saved) & set(current) if saved[k].shape != current[k].shape]
    if missing != expected_new or unexpected or mismatched:
        raise RuntimeError(
            "incompatible QK warm start: "
            f"missing={sorted(missing)}, expected_new={sorted(expected_new)}, "
            f"unexpected={sorted(unexpected)}, shape_mismatch={sorted(mismatched)}"
        )
    # Validate everything before mutating model parameters. Setting gain=1 is
    # explicit; normalization still changes the model function immediately.
    complete = dict(saved)
    complete.update({name: torch.ones_like(current[name]) for name in expected_new})
    model.load_state_dict(complete, strict=True)
    return {
        "parent_step": step,
        "initialized_parameters": sorted(expected_new),
        "copied_tensor_count": len(saved),
        "optimizer_state": "reset",
        "rng_and_data_iterator": "reset",
        "migration": "enable_qk_rms_norm",
    }
