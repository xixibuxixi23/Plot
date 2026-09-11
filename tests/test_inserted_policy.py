import json
import numpy as np
import torch

from plot.data.inserted_policy_dataset import InsertedPolicyDataset
from plot.data.renderer_dataset import TextAgentRendererDataset
from plot.models.inserted_policy import InsertedInhabitantPolicy, InsertedPolicyArgs
from plot.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID
from plot.models.structured_action import StructuredActionHead, constrain_peaceful_logits
from tests.test_renderer import conditions, tiny_model


def test_structured_action_roundtrip_and_peaceful_constraints():
    head = StructuredActionHead(16, horizons=8)
    actions = torch.zeros(2, 8, 23)
    actions[0, :, [0, 8, 10]] = 1
    actions[0, :, 14] = 1
    actions[..., 21] = .05
    actions[..., 22] = -.08
    target = head.targets(actions)
    assert target["keys"][0, 0].tolist() == [1., 0., 0., 0., 0., 0., 0., 1., 0., 1.]
    assert target["hotbar"][0, 0].item() == 3
    logits = head(torch.randn(2, 16))
    constrained = constrain_peaceful_logits(logits, torch.tensor([True, False]))
    decoded = head.decode(constrained)
    assert not decoded[0, :, [8, 9, 10]].any()
    assert not decoded[0, :, 12:21].any()
    loss, pieces = head.loss(logits, actions)
    assert torch.isfinite(loss) and set(pieces) == {"keys", "hotbar", "mouse_x", "mouse_y"}


def _three_family_conditions():
    result = conditions(8)
    for key, value in list(result.items()):
        if torch.is_tensor(value) and value.shape[:1] == (1,):
            result[key] = value.expand(3, *value.shape[1:]).clone()
    result["target_agent"] = torch.zeros(3, dtype=torch.long)
    result["condition_mask"] = torch.ones(3, 8, dtype=torch.bool)
    result["action_prefix_mask"] = torch.zeros(3, 8, dtype=torch.bool)
    return result


def test_inserted_policy_is_exactly_eight_to_eight_and_freezes_m3():
    renderer = tiny_model()
    model = InsertedInhabitantPolicy(
        renderer, InsertedPolicyArgs(num_policy_blocks=2, text_hidden_size=16)).train()
    cond = _three_family_conditions()
    family = torch.tensor([FAMILY_TO_ID[name] for name in
                           ("language_builder", "villager", "combat")])
    profile = torch.tensor([PROFILE_TO_ID[name] for name in
                            ("language_builder", "villager_peaceful", "zombie_melee")])
    text = torch.randn(3, 5, 16)
    mask = torch.ones(3, 5, dtype=torch.bool)
    video, logits = model(torch.randn(3, 8, 16, 4, 4), cond, family, profile,
                          shared_text=text, shared_text_mask=mask,
                          current_text=text, current_text_mask=mask)
    assert video.shape == (3, 8, 16, 4, 4)
    assert logits["keys"].shape == (3, 8, 10)
    assert logits["hotbar"].shape == (3, 8, 10)
    actions = model.decode(logits)
    assert actions.shape == (3, 8, 23)
    assert not actions[1, :, [8, 9, 10]].any() and not actions[1, :, 12:21].any()
    loss, _ = model.loss(logits, torch.zeros_like(actions))
    loss.backward()
    assert all(parameter.grad is None for parameter in renderer.parameters())
    assert all(any(p.grad is not None for p in branch.parameters())
               for branch in model.families.values())
    try:
        model(torch.randn(3, 1, 16, 4, 4), cond, family, profile,
              shared_text=text, shared_text_mask=mask,
              current_text=text, current_text_mask=mask)
    except ValueError as error:
        assert "complete M3 context" in str(error)
    else:
        raise AssertionError("one-frame M4 input must be rejected")


def test_family_profile_mismatch_is_rejected():
    model = InsertedInhabitantPolicy(
        tiny_model(), InsertedPolicyArgs(num_policy_blocks=1, text_hidden_size=16))
    cond = conditions(8)
    text = torch.zeros(1, 1, 16); mask = torch.zeros(1, 1, dtype=torch.bool)
    try:
        model(torch.randn(1, 8, 16, 4, 4), cond,
              torch.tensor([FAMILY_TO_ID["villager"]]),
              torch.tensor([PROFILE_TO_ID["zombie_melee"]]), shared_text=text,
              shared_text_mask=mask, current_text=text, current_text_mask=mask)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("invalid family/profile route must be rejected")


def test_long_m3_context_is_causal_but_policy_selects_only_eight_frames():
    model = InsertedInhabitantPolicy(
        tiny_model(), InsertedPolicyArgs(num_policy_blocks=2, text_hidden_size=16)).eval()
    cond = conditions(65)
    cond["condition_mask"][:] = True
    index = torch.arange(1, 9)[None]
    text = torch.randn(1, 3, 16); mask = torch.ones(1, 3, dtype=torch.bool)
    latent = torch.randn(1, 65, 16, 4, 4)
    args = (torch.tensor([FAMILY_TO_ID["language_builder"]]),
            torch.tensor([PROFILE_TO_ID["language_builder"]]))
    with torch.no_grad():
        _, first = model(latent, cond, *args, shared_text=text, shared_text_mask=mask,
                         current_text=text, current_text_mask=mask, policy_indices=index)
        latent[:, 9:] += 100
        cond["hp"][:, 9:] = 1
        _, second = model(latent, cond, *args, shared_text=text, shared_text_mask=mask,
                          current_text=text, current_text_mask=mask, policy_indices=index)
    for key in first: torch.testing.assert_close(first[key], second[key])


def test_dataset_aligns_eight_completed_frames_to_next_chunk(tmp_path, monkeypatch):
    episode = tmp_path / "episodes" / "episode"; episode.mkdir(parents=True)
    actions = np.arange(80 * 2 * 23, dtype=np.float32).reshape(80, 2, 23)
    np.savez(episode / "data.npz", action_continuous=actions,
             cam_pos=np.zeros((81, 2, 3)), action_source=np.full((80, 2), "policy"))
    (episode / "manifest.json").write_text(json.dumps(
        {"num_agents": 2, "model_start_observation": 0}))
    captured = {}

    def fake_window(cls, path, vocabulary, **kwargs):
        captured.update(kwargs)
        return {"rgb": torch.zeros(65, 3, 4, 4), "region_weight": torch.ones(65, 1, 1, 1),
                "conditions": {"action": torch.zeros(65, 2, 23)}}
    monkeypatch.setattr(TextAgentRendererDataset, "read_window", classmethod(fake_window))
    index = tmp_path / "index.jsonl"
    index.write_text(json.dumps({
        "episode_path": "episode", "anchor": 64, "agent_slot": 1,
        "family": "language_builder", "profile": "language_builder",
        "task_text": "build a wall", "sample_weight": 2.,
    }) + "\n")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"texts": [
        {"text_id": 0, "text": ""}, {"text_id": 1, "text": "build a wall"}] }))
    sample = InsertedPolicyDataset(index, object(), catalog, dataset_root=episode.parent)[0]
    assert captured == {"start": 0, "target": 1, "context_frames": 65}
    assert sample["policy_indices"].tolist() == list(range(57, 65))
    torch.testing.assert_close(sample["target_actions"], torch.from_numpy(actions[64:72, 1]))
    assert sample["shared_text_id"].item() == sample["current_text_id"].item() == 1
