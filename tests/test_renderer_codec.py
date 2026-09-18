"""The same fixed latent coordinate system must be used throughout M3."""
from dataclasses import asdict
import json

import pytest
import torch

from plot.models.latent_normalization import (
    normalization_from_run_config,
    pixel_vae_latent_stats,
    resolve_training_normalization,
    validate_latent_normalization,
)
from plot.models.renderer_backbone.vae_pixel import ViTVae, ViTVaeArgs
from plot.models.renderer_codec import RendererCodec


torch.set_num_threads(2)


@pytest.fixture
def codec_pair():
    cfg = ViTVaeArgs(input_height=40, input_width=40, enc_dim=32, enc_depth=1,
                     enc_heads=4, dec_dim=32, dec_depth=1, dec_heads=4)
    weights = ViTVae(**asdict(cfg)).state_dict()
    return (
        RendererCodec(weights, cfg),
        RendererCodec(weights, cfg, latent_normalization=pixel_vae_latent_stats()),
        weights,
        cfg,
    )


def test_encode_normalizes_and_decode_inverts(codec_pair):
    raw, normalized, _, _ = codec_pair
    rgb = torch.rand(2, 9, 3, 40, 40)
    latent = raw.encode(rgb)
    encoded = normalized.encode(rgb)
    expected = (latent - normalized.latent_mean) / normalized.latent_std
    torch.testing.assert_close(encoded, expected)
    torch.testing.assert_close(normalized.denormalize_latent(encoded), latent)
    torch.testing.assert_close(normalized.decode(encoded), raw.decode(latent), atol=1e-6, rtol=1e-5)
    assert not encoded.requires_grad
    assert raw.get_config() == {"latent_normalization": {"mode": "none"}}


def test_decode_for_loss_preserves_gradients(codec_pair):
    raw, normalized, _, _ = codec_pair
    latent = raw.encode(torch.rand(1, 2, 3, 40, 40)).requires_grad_()
    encoded = normalized.normalize_latent(latent.detach()).requires_grad_()
    raw.decode_for_loss(latent).sum().backward()
    normalized.decode_for_loss(encoded).sum().backward()
    torch.testing.assert_close(encoded.grad, latent.grad * normalized.latent_std, atol=1e-5, rtol=1e-4)
    assert encoded.grad.abs().sum() > 0
    assert all(p.grad is None for p in normalized.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_affine_is_frame_independent_and_preserves_dtype(codec_pair, dtype):
    _, codec, _, _ = codec_pair
    latent = torch.randn(2, 17, 16, 4, 4).to(dtype)
    encoded = codec.normalize_latent(latent)
    torch.testing.assert_close(encoded[:1, :1], codec.normalize_latent(latent[:1, :1]))
    assert encoded.dtype == dtype
    decoded = codec.denormalize_latent(encoded)
    assert decoded.dtype == dtype
    tolerance = 0.04 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(decoded, latent, atol=tolerance, rtol=0)


def test_config_roundtrip_and_legacy_scale(codec_pair):
    raw, codec, weights, cfg = codec_pair
    config = json.loads(json.dumps({"codec": codec.get_config()}))
    restored = RendererCodec.from_run_config(weights, config, cfg)
    latent = torch.randn(1, 9, 16, 4, 4)
    torch.testing.assert_close(restored.normalize_latent(latent), codec.normalize_latent(latent))
    legacy = RendererCodec.from_run_config(weights, {"renderer": {"simple_conditioning": True}}, cfg)
    assert legacy.normalize_latent(latent) is latent
    assert legacy.get_config() == raw.get_config()
    config["codec"]["latent_normalization"]["mean"][0] = 100
    assert codec.get_config()["latent_normalization"]["mean"][0] != 100
    assert restored.get_config()["latent_normalization"]["mean"][0] != 100


def test_training_scale_defaults_and_checkpoint_inheritance():
    stats = pixel_vae_latent_stats()
    assert resolve_training_normalization(None, simple_m3=True) == stats
    assert resolve_training_normalization(None, simple_m3=False) == {"mode": "none"}
    assert resolve_training_normalization("none", simple_m3=True) == {"mode": "none"}
    assert resolve_training_normalization("pixel-vae", simple_m3=False) == stats
    assert resolve_training_normalization(None, simple_m3=True, checkpoint_config={}) == {"mode": "none"}
    saved = {"codec": {"latent_normalization": stats}}
    assert resolve_training_normalization(None, simple_m3=False, checkpoint_config=saved) == stats
    assert resolve_training_normalization("pixel-vae", simple_m3=True, checkpoint_config=saved) == stats
    with pytest.raises(ValueError, match="differs from checkpoint"):
        resolve_training_normalization("pixel-vae", simple_m3=True, checkpoint_config={})
    with pytest.raises(ValueError, match="differs from checkpoint"):
        resolve_training_normalization("none", simple_m3=True, checkpoint_config=saved)


@pytest.mark.parametrize("key,value,match", [
    ("mean", [0.0], "16 channels"),
    ("std", [1.0] * 15, "16 channels"),
    ("mean", [float("nan")] * 16, "finite"),
    ("std", [float("inf")] * 16, "finite"),
    ("std", [0.0] * 16, "positive"),
    ("std", [-1.0] * 16, "positive"),
    ("mode", "batch", "mode"),
])
def test_invalid_stats_fail_before_constructing_vae(key, value, match):
    stats = pixel_vae_latent_stats()
    stats[key] = value
    with pytest.raises(ValueError, match=match):
        RendererCodec({}, latent_normalization=stats)


def test_stats_are_fresh_and_legacy_default_is_explicit():
    stats = pixel_vae_latent_stats()
    stats["mean"][0] = 1234
    assert pixel_vae_latent_stats()["mean"][0] != 1234
    assert validate_latent_normalization(None) == {"mode": "none"}
    assert normalization_from_run_config({}) == {"mode": "none"}
