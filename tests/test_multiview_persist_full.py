import numpy as np
import pytest
import torch
from torch.utils.data import DistributedSampler

from experiments.m1.train_multiview_persist_full import Source


def test_high_noise_mixture_preserves_global_rng_and_distribution():
    from experiments.m1.train_multiview_persist_full import mix_high_noise_time

    t = torch.randn(100000).sigmoid()
    before = torch.random.get_rng_state().clone()
    a = mix_high_noise_time(t, torch.Generator().manual_seed(71))
    b = mix_high_noise_time(t, torch.Generator().manual_seed(71))
    assert torch.equal(a, b)
    assert torch.equal(before, torch.random.get_rng_state())
    assert 0.48 < float((a == t).float().mean()) < 0.52
    assert 0.52 < float((a >= 0.8).float().mean()) < 0.56
    assert ((a[a != t] >= 0.8) & (a[a != t] <= 1)).all()


def test_step_cap_handles_partial_epochs_and_resume():
    from experiments.m1.train_multiview_persist_full import step_limited_end_epoch

    assert step_limited_end_epoch(20, 0, 14440, 1_000_000, 722) == 1386
    assert step_limited_end_epoch(20, 0, 14440, 14441, 722) == 21
    assert step_limited_end_epoch(20, 1, 14441, 15162, 722) == 21
    assert step_limited_end_epoch(20, 1, 14441, 15163, 722) == 22
    with pytest.raises(ValueError):
        step_limited_end_epoch(20, 0, 14440, 14440, 722)


def test_full_direct_prediction_matches_pilot_and_uses_fixed_queries():
    from experiments.m1.train_multiview_persist_full import predict_clean
    from experiments.m1.train_multiview_objective_compare import direct_prediction

    class Spy(torch.nn.Module):
        def forward(self, x, t, cond, valid):
            assert torch.count_nonzero(x) == 0
            assert torch.count_nonzero(t) == 0
            return x + cond[:, :1, :1, :1, :1]

    cond = torch.ones(2, 1, 1, 1, 1, requires_grad=True)
    valid = torch.ones(2, 1, dtype=torch.bool)
    a = predict_clean(Spy(), cond, valid)
    assert torch.equal(a, direct_prediction(Spy(), cond, valid))
    a.mean().backward()
    assert torch.count_nonzero(cond.grad) == 2


def test_resume_lr_preserves_adam_moments():
    from experiments.m1.train_multiview_persist_full import restore_optimizer

    p = torch.nn.Parameter(torch.tensor([2.0]))
    old = torch.optim.AdamW([p], lr=1e-4)
    p.square().sum().backward()
    old.step()
    state = old.state_dict()
    new = torch.optim.AdamW([p], lr=9e-3)
    restore_optimizer(new, state, 3e-5)
    assert new.param_groups[0]["lr"] == 3e-5
    for key in ("step", "exp_avg", "exp_avg_sq"):
        assert torch.equal(new.state[p][key], old.state[p][key])
    restore_optimizer(new, old.state_dict())
    assert new.param_groups[0]["lr"] == 1e-4


def test_unknown_state_keeps_material_and_known_states(tmp_path):
    tile = np.array([[[[1, 0], [1, 7], [1, 19]]]])
    np.savez(tmp_path / "m1_initial.npz", voxel_tiles=tile[None])

    class Fake:
        index = [(0, 0, 0)]
        episodes = [(tmp_path, {})]

        def __getitem__(self, i):
            return {
                k: torch.zeros(1)
                for k in (
                    "images",
                    "camera_position",
                    "camera_direction",
                    "fov_x",
                    "fov_y",
                    "agent_mask",
                    "camera_valid",
                )
            }

    source = Source.__new__(Source)
    source.datasets = [Fake()]
    source.items = [(0, 0)]
    source.lut = np.full((3, 256), -1)
    source.lut[1, 0], source.lut[1, 7] = 10, 11
    source.fallback = np.array([-1, 10, -1])
    result = source[0]
    assert result["ids"].flatten().tolist() == [10, 11, 10]
    assert result["raw"].flatten().tolist() == [1, 1, 1]
    assert "19" in result["aliases"]
    tile[0, 0, 0, 0] = 2
    np.savez(tmp_path / "m1_initial.npz", voxel_tiles=tile[None])
    with pytest.raises(ValueError, match="Unknown raw block ID"):
        source[0]


def test_full_training_and_evaluation_cover_every_sample_once():
    n, world, chunk = 46200, 8, 32
    sampled = []
    evaluated = []
    for rank in range(world):
        sampled.extend(
            DistributedSampler(range(n), num_replicas=world, rank=rank, seed=20260910, shuffle=True)
        )
        for start in range(rank * chunk, n, world * chunk):
            evaluated.extend(range(start, min(start + chunk, n)))
    assert sorted(sampled) == list(range(n))
    assert sorted(evaluated) == list(range(n))


def test_batched_sampling_uses_global_sample_seed_without_target_input():
    from scripts.evaluate_multiview_persist_full import sample

    class Velocity(torch.nn.Module):
        def forward(self, x, t, cond, valid):
            return x * 0.2 + cond[:, :1, :1, :1, :1]

    model = Velocity()
    cond = torch.tensor([1.0, 2.0]).reshape(2, 1, 1, 1, 1)
    valid = torch.ones(2, 1, dtype=torch.bool)
    combined = sample(model, cond, valid, [10, 37])
    separate = torch.cat(
        [sample(model, cond[i : i + 1], valid[i : i + 1], [idx]) for i, idx in enumerate([10, 37])]
    )
    assert torch.equal(combined, separate)
    swapped = sample(model, cond.flip(0), valid, [37, 10])
    assert torch.equal(combined.flip(0), swapped)
