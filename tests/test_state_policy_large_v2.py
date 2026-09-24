import torch
from torch import nn

from plot.models.state_policy_large_v2 import (
    ZombieStatePolicyV2, ZombieStatePolicyV2Args,
    ZombieStatePolicyV3, ZombieStatePolicyV3Args,
)


def inputs(batch=2):
    return dict(
        resident_state=torch.randn(batch, 8, 3, 18),
        resident_actions=torch.zeros(batch, 8, 3, 23),
        resident_valid=torch.ones(batch, 8, 3, dtype=torch.bool),
        held_item=torch.zeros(batch, 8, 3, dtype=torch.long),
        resident_type=torch.zeros(batch, 8, 3, dtype=torch.long),
        history_valid=torch.ones(batch, 8, dtype=torch.bool),
        target_agent=torch.zeros(batch, dtype=torch.long),
        voxel_classes=torch.zeros(batch, 48, 48, 48, dtype=torch.long),
        voxel_known=torch.ones(batch, 48, 48, 48, dtype=torch.bool),
        raster_camera=torch.tensor([[0., 1., 0., 0., 0., -1., 0., 0., 0., 1.2]]).repeat(batch, 1))


class StubGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(1, 4, 40))

    def forward(self, values):
        return self.value.expand(len(values["target_agent"]), -1, -1)


def tiny(dropout=0.0):
    model = ZombieStatePolicyV2(ZombieStatePolicyV2Args(
        3, 4, hidden=40, heads=4, depth=2, dropout=dropout,
        gradient_checkpointing=True, image_h=4, image_w=4))
    model.geometry = StubGeometry()
    return model


def test_v2_split_heads_loss_and_roundtrip(tmp_path):
    torch.manual_seed(31)
    model = tiny()
    data = inputs()
    actions = torch.zeros(2, 8, 23)
    actions[0, 2:5, 8] = 1
    actions[1, :, 21] = 0.05
    logits = model(data)
    assert logits["keys"].shape == (2, 8, 10)
    assert logits["attack"].shape == (2, 8)
    assert logits["mouse_x_move"].shape == (2, 8)
    torch.testing.assert_close(logits["keys"][..., 7], logits["attack"])
    loss, parts = model.loss(logits, actions, horizon_weights=torch.linspace(1, 0.25, 8))
    assert set(parts) == {"move_keys", "attack", "hotbar", "mouse_x", "mouse_y",
                          "mouse_x_gate", "mouse_y_gate"}
    loss.backward()
    assert model.attack_head[-1].weight.grad.abs().sum() > 0
    # Slot 7 of the common key head is replaced before loss and must not receive gradients.
    assert model.common_head.keys.weight.grad.view(1, 10, 40)[0, 7].abs().sum() == 0
    assert model.mouse_move_head[-1].weight.grad.abs().sum() > 0
    decoded = model.decode(logits)
    assert decoded.shape == (2, 8, 23)
    torch.save(model.state_dict(), tmp_path / "v2.pt")
    restored = tiny().eval()
    restored.load_state_dict(torch.load(tmp_path / "v2.pt", weights_only=True))
    model.eval()
    for key, value in model(data).items():
        torch.testing.assert_close(value, restored(data)[key])


def test_v2_soft_mouse_targets_credit_neighbor_bins():
    target = torch.tensor([[8]])
    center = torch.full((1, 1, 17), -8.0)
    center[..., 8] = 8.0
    adjacent = torch.full((1, 1, 17), -8.0)
    adjacent[..., 9] = 8.0
    far = torch.full((1, 1, 17), -8.0)
    far[..., 14] = 8.0
    center_loss = ZombieStatePolicyV2._soft_bin_loss(center, target, 0.2)
    adjacent_loss = ZombieStatePolicyV2._soft_bin_loss(adjacent, target, 0.2)
    far_loss = ZombieStatePolicyV2._soft_bin_loss(far, target, 0.2)
    assert center_loss < adjacent_loss < far_loss


def test_v2_dropout_disabled_in_eval():
    torch.manual_seed(32)
    model = tiny(dropout=0.2).eval()
    data = inputs(batch=1)
    first = model(data)
    second = model(data)
    for key in first:
        torch.testing.assert_close(first[key], second[key])


def test_v3_explicit_combat_context_and_auxiliary_loss():
    torch.manual_seed(33)
    model = ZombieStatePolicyV3(ZombieStatePolicyV3Args(
        3, 4, hidden=40, heads=4, depth=2, dropout=0.0,
        gradient_checkpointing=True, image_h=4, image_w=4))
    model.geometry = StubGeometry()
    data = inputs()
    data["resident_type"][:, :, 0] = 2
    data["resident_valid"][:, :, 2] = False
    data["resident_state"][..., :3] = 0
    data["resident_state"][:, -2:, 1, 0] = torch.tensor([[2.5 / 24], [3.5 / 24]])
    actions = torch.zeros(2, 8, 23)
    actions[0, :, 0] = 1
    actions[0, 2, 8] = 1
    logits = model(data)
    assert logits["combat_range"].shape == (2, 8)
    assert logits["combat_bearing"].shape == (2, 8, 3)
    torch.testing.assert_close(logits["combat_distance"], torch.tensor([2.5, 3.5]))
    loss, parts = model.loss(logits, actions, positive_weight=4.0)
    assert {"locomotion", "combat_range", "combat_bearing"} <= set(parts)
    loss.backward()
    assert model.combat_encoder[0].weight.grad.abs().sum() > 0
    assert model.range_head[-1].weight.grad.abs().sum() > 0
    decoded = model.decode(logits, attack_threshold=0.0)
    assert decoded[..., 8].all()
