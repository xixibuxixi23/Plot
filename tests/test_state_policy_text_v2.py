import torch
from torch import nn

from plot.models.state_policy_text_v2 import (
    IndependentStatePolicyV4, TextStatePolicyV2, TextStatePolicyV2Args,
    UnifiedStatePolicyV4, UnifiedStatePolicyV4Args,
)


class StubGeometry(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.randn(1, 4, 40))

    def forward(self, inputs):
        return self.value.expand(len(inputs["target_agent"]), -1, -1)


def make_model():
    model = TextStatePolicyV2(TextStatePolicyV2Args(
        3, 4, hidden=40, heads=4, depth=2, dropout=0.0,
        text_hidden_size=24, image_h=4, image_w=4))
    model.geometry = StubGeometry()
    return model


def make_inputs(batch=2):
    values = dict(
        resident_state=torch.randn(batch, 8, 3, 18),
        resident_actions=torch.zeros(batch, 8, 3, 23),
        resident_valid=torch.ones(batch, 8, 3, dtype=torch.bool),
        held_item=torch.zeros(batch, 8, 3, dtype=torch.long),
        resident_type=torch.zeros(batch, 8, 3, dtype=torch.long),
        history_valid=torch.ones(batch, 8, dtype=torch.bool),
        target_agent=torch.zeros(batch, dtype=torch.long),
        shared_text=torch.randn(batch, 5, 24),
        shared_text_mask=torch.ones(batch, 5, dtype=torch.bool),
        current_text=torch.randn(batch, 7, 24),
        current_text_mask=torch.ones(batch, 7, dtype=torch.bool))
    return values


def test_text_v2_split_heads_and_text_gradients():
    torch.manual_seed(61)
    model = make_model()
    inputs = make_inputs()
    actions = torch.zeros(2, 8, 23)
    actions[0, 1:4, 8] = 1
    actions[1, 3:7, 9] = 1
    actions[:, :, 21] = 0.05
    logits = model(inputs)
    torch.testing.assert_close(logits["keys"][..., 7], logits["attack"])
    torch.testing.assert_close(logits["keys"][..., 8], logits["place"])
    loss, parts = model.loss(logits, actions, horizon_weights=torch.linspace(1, 0.25, 8))
    assert set(parts) == {"move_keys", "attack", "place", "hotbar", "mouse_x",
                          "mouse_y", "mouse_x_gate", "mouse_y_gate"}
    loss.backward()
    assert model.text_projection.weight.grad.abs().sum() > 0
    assert model.attack_head[-1].weight.grad.abs().sum() > 0
    assert model.place_head[-1].weight.grad.abs().sum() > 0
    common = model.common_head.keys.weight.grad.view(1, 10, 40)[0]
    assert common[7].abs().sum() == 0
    assert common[8].abs().sum() == 0
    assert model.mouse_move_head[-1].weight.grad.abs().sum() > 0
    assert model.decode(logits).shape == (2, 8, 23)


def test_text_changes_prediction():
    torch.manual_seed(62)
    model = make_model().eval()
    first = make_inputs(batch=1)
    second = {key: value.clone() for key, value in first.items()}
    second["current_text"].add_(2.0)
    output_a = model(first)["place"]
    output_b = model(second)["place"]
    assert not torch.allclose(output_a, output_b)


def make_unified_model():
    model = UnifiedStatePolicyV4(UnifiedStatePolicyV4Args(
        3, 4, hidden=40, heads=4, depth=2, dropout=0.0,
        text_hidden_size=24, image_h=4, image_w=4))
    model.geometry = StubGeometry()
    return model


def test_unified_v4_shares_actor_and_accepts_null_text():
    torch.manual_seed(63)
    model = make_unified_model()
    inputs = make_inputs()
    inputs["policy_role"] = torch.tensor([0, 1])
    for key in ("shared_text", "shared_text_mask", "current_text", "current_text_mask"):
        del inputs[key]
    logits = model(inputs)
    assert logits["keys"].shape == (2, 8, 10)
    actions = torch.zeros(2, 8, 23)
    actions[0, :, 9] = 1
    actions[1, :, 8] = 1
    loss, _ = model.loss(logits, actions)
    loss.backward()
    assert model.policy_role.weight.grad.abs().sum() > 0
    assert model.common_head.hotbar.weight.grad.abs().sum() > 0
    assert model.attack_head[-1].weight.grad.abs().sum() > 0
    assert model.place_head[-1].weight.grad.abs().sum() > 0


def test_unified_v4_role_changes_prediction():
    torch.manual_seed(64)
    model = make_unified_model().eval()
    first = make_inputs(batch=1)
    second = {key: value.clone() for key, value in first.items()}
    first["policy_role"] = torch.tensor([0])
    second["policy_role"] = torch.tensor([1])
    assert not torch.allclose(model(first)["keys"], model(second)["keys"])


def test_independent_v4_has_same_architecture_but_private_parameters():
    builder = IndependentStatePolicyV4(TextStatePolicyV2Args(
        3, 4, profile="language_builder", hidden=40, heads=4, depth=2,
        dropout=0.0, text_hidden_size=24, image_h=4, image_w=4))
    zombie = IndependentStatePolicyV4(TextStatePolicyV2Args(
        3, 4, profile="zombie_melee", hidden=40, heads=4, depth=2,
        dropout=0.0, text_hidden_size=24, image_h=4, image_w=4))
    builder_state = builder.state_dict()
    zombie_state = zombie.state_dict()
    assert list(builder_state) == list(zombie_state)
    assert all(builder_state[key].shape == zombie_state[key].shape for key in builder_state)
    assert all(left.data_ptr() != right.data_ptr()
               for left, right in zip(builder.parameters(), zombie.parameters()))

    zombie.geometry = StubGeometry()
    inputs = make_inputs(batch=1)
    for key in ("shared_text", "shared_text_mask", "current_text", "current_text_mask"):
        del inputs[key]
    assert zombie(inputs)["keys"].shape == (1, 8, 10)


def test_target_pointer_is_end_to_end_and_uses_fixed_null_label():
    torch.manual_seed(65)
    model = TextStatePolicyV2(TextStatePolicyV2Args(
        3, 4, hidden=40, heads=4, depth=2, dropout=0.0,
        text_hidden_size=24, image_h=4, image_w=4, target_pointer=True))
    model.geometry = StubGeometry()
    inputs = make_inputs()
    inputs["voxel_classes"] = torch.zeros(2, 48, 48, 48, dtype=torch.long)
    inputs["voxel_known"] = torch.ones(2, 48, 48, 48, dtype=torch.bool)
    inputs["grid_offset"] = torch.zeros(2, 3)
    actions = torch.zeros(2, 8, 23)
    # First row points to an exact voxel; 2197 is the sidecar's fixed null.
    target = torch.tensor([1098, 2197])
    logits = model(inputs)
    assert logits["target"].shape == (2, 2197 + 3 + 1)
    assert logits["target_entity"].shape == (2, 3 + 1)
    loss, parts = model.loss(logits, actions, target_address=target)
    assert torch.isfinite(parts["target"])
    loss.backward()
    assert model.target_query[-1].weight.grad.abs().sum() > 0
    assert model.target_voxel.weight.grad.abs().sum() > 0
    assert model.target_entity_query[-1].weight.grad.abs().sum() > 0
    assert model.blocks[0].ego_attn.q.weight.grad.abs().sum() > 0
    assert model.blocks[0].target_attn.q.weight.grad.abs().sum() > 0
    assert model.query.grad.abs().sum() > 0
