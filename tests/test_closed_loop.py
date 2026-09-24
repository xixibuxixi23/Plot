from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from plot.pipelines.closed_loop_pipeline import ClosedLoopPipeline
from plot.pipelines.transition_pipeline import TransitionCommitter
from plot.transition_state import CharRow
from plot.world_memory import WorldMemory


class FakeTransition(nn.Module):
    def forward(self, inputs):
        b, t, a = inputs["actions"].shape[:3]
        n = 2197 + a + 1
        logits = torch.zeros(b, t, a, n, device=inputs["actions"].device)
        return {
            "address_logits": logits, "occurrence_logits": logits[..., 0] - 20,
            "pose": inputs["initial_pose"][:, None].expand(-1, t, -1, -1),
            "velocity": torch.zeros(b, t, a, 3, device=logits.device),
            "held_logits": torch.zeros(b, t, a, 2, device=logits.device),
            "camera_relative": inputs["camera_relative"][:, None].expand(-1, t, -1, -1),
            "camera_direction": inputs["camera_direction"][:, None].expand(-1, t, -1, -1),
        }

    def payloads(self, output, address):
        shape = address.shape
        return torch.zeros(*shape, 2, device=address.device), torch.zeros(shape, device=address.device)


class FakeFill:
    def __init__(self, memory):
        self.memory = memory

    def fill_resident_windows(self, centers, images):
        for center in centers:
            self.memory.commit_tile(center, np.zeros((48, 48, 48), np.int32),
                                    np.ones((48, 48, 48), bool))


class FakeRollout:
    def __init__(self, agents):
        self.model = SimpleNamespace(cfg=SimpleNamespace(in_channels=16, input_h=4, input_w=4))
        self.last_policy_features = torch.randn(agents, 8, 128)
        self.last_policy_layers = None
        self.condition = None

    def start(self, latent, condition):
        pass

    def generate(self, noise, condition):
        self.condition = condition
        self.last_policy_features = torch.randn(len(noise), 8, 128)
        self.last_policy_layers = tuple(torch.randn(len(noise), 8, 2, 2, 8) for _ in range(4))
        return noise


class FakeCodec:
    def decode(self, latent):
        return torch.zeros(len(latent), 8, 3, 16, 16)


class FakeInsertedPolicy(nn.Module):
    def __init__(self):
        super().__init__(); self.calls = 0

    def forward(self, latents, conditions, family_id, profile_id, **text):
        self.calls += 1
        assert latents is None and conditions["condition_mask"].all()
        assert len(text["video_layers"]) == 4
        return None, torch.ones(len(family_id), 8, 23)

    def decode(self, logits): return logits


def test_one_closed_loop_block_has_authoritative_order():
    a = 2
    memory = WorldMemory()
    rows = [CharRow(f"agent{i}", i, (float(i), 0., 0.), 0., 0., 20., 0, 0,
                    (0., 0., 1.5), (0., 1., 0.)) for i in range(a)]
    committer = TransitionCommitter(memory, rows)
    policy = FakeInsertedPolicy()
    rollout = FakeRollout(a)
    pipeline = ClosedLoopPipeline(
        policy=policy, transition=FakeTransition(), committer=committer,
        renderer_rollout=rollout, codec=FakeCodec(), fill_pipeline=FakeFill(memory),
        resident_ids=["agent0", "agent1"], skins=torch.zeros(a, 4, 4, 8, 8),
        appearance_valid=torch.ones(a, 4, dtype=torch.bool), fov_x=np.ones(a), device="cpu",
    )
    pipeline.prime(first_latents=torch.zeros(a, 1, 16, 4, 4), first_conditions={},
                   action_history=torch.zeros(a, 8, 23), latest_rgb=torch.zeros(a, 3, 16, 16))
    transition_inputs = {
        "initial_pose": torch.tensor([[[0., 0., 0., 0., 0.], [1., 0., 0., 0., 0.]]]),
        "camera_relative": torch.tensor([[[0., 0., 1.5], [0., 0., 1.5]]]),
        "camera_direction": torch.tensor([[[0., 1., 0.], [0., 1., 0.]]]),
    }
    policy_state = {
        "family_id": torch.zeros(a, dtype=torch.long), "profile_id": torch.zeros(a, dtype=torch.long),
        "shared_text": torch.ones(a, 4, 8), "shared_text_mask": torch.ones(a, 4, dtype=torch.bool),
        "current_text": torch.ones(a, 4, 8), "current_text_mask": torch.ones(a, 4, dtype=torch.bool),
    }
    result = pipeline.run_block(
        policy_state=policy_state, external_actions=torch.zeros(a, 8, 23),
        controlled=torch.tensor([True, False]), transition_inputs=transition_inputs,
        anchors=np.zeros((a, 3), np.int64),
    )
    assert result.rgb.shape == (a, 8, 3, 16, 16)
    assert len(result.snapshots) == 8 and committer.next_transition == 8
    assert result.snapshots[-1]["agent0"].velocity_xyz == (0.0, 0.0, 0.0)
    assert rollout.condition["voxel_known"].all()
    assert rollout.condition["target_agent"].tolist() == [0, 1]
    assert result.actions[1].sum() == 0
    assert policy.calls == 0  # one-frame initialization uses the external bootstrap chunk
    second = pipeline.run_block(
        policy_state=policy_state, external_actions=torch.zeros(a, 8, 23),
        controlled=torch.tensor([True, False]), transition_inputs=transition_inputs,
        anchors=np.zeros((a, 3), np.int64),
    )
    assert policy.calls == 1 and second.actions[0].sum() == 8 * 23
    assert second.actions[1].sum() == 0
