import torch
from torch import nn

from plot.models.player_identity import (
    PlayerIdentityEncoder,
    crop_masked_players,
    multi_positive_contrastive_loss,
)
from train_scripts.train_renderer import build_flow_region_weight
from plot.training.renderer_trainer import renderer_player_identity_loss
from scripts.audit_m3_reference_condition import _identity_metrics


def test_masked_player_crop_preserves_gradient_and_alpha():
    image = torch.rand(2, 3, 24, 32, requires_grad=True)
    mask = torch.zeros(2, 1, 24, 32)
    mask[0, :, 4:20, 10:18] = 1
    crop, valid = crop_masked_players(image, mask, output_size=(32, 16))
    assert crop.shape == (2, 4, 32, 16)
    assert valid.tolist() == [True, False]
    assert crop[0, 3].max() == 1
    assert crop[1].count_nonzero() == 0
    crop.sum().backward()
    assert image.grad is not None and image.grad[0].abs().sum() > 0
    assert image.grad[1].count_nonzero() == 0


def test_player_identity_encoder_contract():
    model = PlayerIdentityEncoder(32)
    crop = torch.rand(3, 4, 64, 32)
    reference = torch.rand(3, 4, 4, 128, 64)
    crop_embedding, reference_embedding = model(crop, reference)
    assert crop_embedding.shape == reference_embedding.shape == (3, 32)
    torch.testing.assert_close(crop_embedding.norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(reference_embedding.norm(dim=-1), torch.ones(3))


def test_contrastive_loss_rewards_correct_pairing():
    identity = torch.arange(4)
    valid = torch.ones(4, dtype=torch.bool)
    embedding = torch.eye(4)
    matched = multi_positive_contrastive_loss(embedding, embedding, identity, valid)
    shuffled = multi_positive_contrastive_loss(
        embedding, embedding.roll(1, 0), identity, valid
    )
    assert matched < shuffled


def test_player_flow_mask_is_dynamic_and_spatially_local():
    base = torch.ones(2, 3, 1, 2, 2)
    mask = torch.zeros(2, 3, 1, 4, 4)
    mask[0, :, :, :2, :2] = 1
    focused = build_flow_region_weight(
        base,
        mask,
        player_upweight=4,
        probability=1,
        randomize=False,
    )
    assert focused[0, :, :, 0, 0].eq(5).all()
    assert focused[0, :, :, 1, 1].eq(1).all()
    assert focused[1].eq(1).all()


class _IdentityCodec:
    def decode_for_loss(self, latent, chunk_size=1):
        return latent


class _ColorIdentity(nn.Module):
    crop_size = (8, 8)

    @staticmethod
    def _embedding(value):
        rgb = value[:, :3].mean(dim=(-2, -1))
        return torch.nn.functional.normalize(rgb, dim=-1, eps=1e-6)

    def encode_crop(self, crop):
        return self._embedding(crop)

    def encode_reference(self, reference):
        return self._embedding(reference.mean(1))


def test_renderer_identity_loss_uses_correct_reference_and_backpropagates():
    prediction = torch.zeros(1, 2, 3, 8, 8, requires_grad=True)
    with torch.no_grad():
        prediction[:, 1, 0] = 0.8
        prediction[:, 1, 1] = 0.2
    reference = torch.zeros(1, 2, 4, 4, 8, 8)
    reference[:, 0, :, 0] = 1
    reference[:, 0, :, 3] = 1
    reference[:, 1, :, 1] = 1
    reference[:, 1, :, 3] = 1
    mask = torch.ones(1, 1, 8, 8)
    result = renderer_player_identity_loss(
        _IdentityCodec(),
        _ColorIdentity(),
        prediction,
        reference,
        mask,
        torch.tensor([1]),
        torch.tensor([0]),
        torch.tensor([True]),
        player_appearance_valid=torch.ones(1, 2, 4, dtype=torch.bool),
    )
    assert result["player_identity_similarity"] > 0.9
    assert result["player_identity_ranking_accuracy"] == 1
    result["player_identity_loss"].backward()
    assert prediction.grad[:, 1].abs().sum() > 0


def test_reference_audit_identity_metric_maps_future_frame_index():
    prediction = torch.zeros(3, 3, 8, 8)
    prediction[1, 0] = 1
    prediction[2, 1] = 1
    references = torch.zeros(2, 4, 4, 8, 8)
    references[0, :, 0] = 1
    references[0, :, 3] = 1
    references[1, :, 1] = 1
    references[1, :, 3] = 1
    raw = {
        "player_identity_valid": torch.tensor(True),
        # Original clip frame 2 is prediction index 1 after dropping frame 0.
        "player_identity_frame": torch.tensor(2),
        "player_identity_slot": torch.tensor(0),
        "player_identity_mask": torch.ones(1, 8, 8),
        "conditions": {
            "player_reference": references,
            "player_appearance_valid": torch.ones(2, 4, dtype=torch.bool),
        },
    }
    result = _identity_metrics(_ColorIdentity(), prediction, raw)
    assert result["identity_similarity"] > 0.99
    assert result["identity_wrong_similarity"] < 0.01
    assert result["identity_ranking_correct"] is True
