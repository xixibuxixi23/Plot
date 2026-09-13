import torch
from contextlib import nullcontext
from copy import deepcopy

from experiments.m1.train_projective_memorization import region_loss, backward_batch
from plot.models import ProjectiveFillArgs
from torch.utils.data import DistributedSampler
from scripts.evaluate_projective_memorization import batch_metrics


def test_region_loss_is_mean_of_scenes_and_excludes_known():
    torch.manual_seed(4)
    logits = torch.randn(2, 3, 4, 4, 4, requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 4, 4))
    valid = torch.ones_like(target, dtype=torch.bool)
    surface = torch.zeros_like(valid)
    surface[0, 0, 0, 0] = True
    surface[1, :2] = True
    free = ~surface
    known = torch.zeros_like(valid)
    known[:, 3] = True
    b = dict(target=target, target_valid=valid, fill_mask=valid, known_mask=known,
             surface=surface, free=free)
    loss = region_loss(logits, b, .1)
    separate = sum(region_loss(logits[i:i+1], {k: v[i:i+1] for k, v in b.items()}, .1)
                   for i in range(2)) / 2
    assert torch.allclose(loss, separate)
    loss.backward()
    assert (logits.grad.movedim(1, -1)[known] == 0).all()
    assert logits.grad.isfinite().all()


def test_exhaustive_metrics_oracle_and_error_partition():
    target = torch.zeros(2, 8, 8, 8, dtype=torch.long)
    target[:, 6, :, :] = 1
    surface = target.bool()
    pos = torch.tensor([[[-.4, 0., 0.]], [[-.4, 0., 0.]]])
    direction = torch.tensor([[[1., 0., 0.]], [[1., 0., 0.]]])
    b = dict(target=target, target_valid=torch.ones_like(surface), surface=surface, free=~surface,
        gt_position=pos, camera_position=pos, gt_direction=direction, camera_direction=direction,
        images=torch.zeros(2, 1, 3, 32, 32), fov_x=torch.ones(2, 1), fov_y=torch.ones(2, 1),
        camera_valid=torch.ones(2, 1, dtype=torch.bool), agent_mask=torch.ones(2, 1, dtype=torch.bool),
        sample_index=torch.tensor([5, 9]))
    rows = batch_metrics(target, b, 0)
    for row in rows:
        assert row["surface_exact_recall"] == 1
        assert row["visible_exact_precision"] == 1
        assert row["pred_pose/half_block_hit"] == 1
        assert row["pred_pose/hit_or_missing_mae"] == 0
    pred = target.clone()
    pred[0] = 0
    pred[1, 4, :, :] = 1
    rows = batch_metrics(pred, b, 0)
    assert rows[0]["pred_pose/missing_hit"] == 1
    for row in rows:
        assert abs(sum(row["pred_pose/" + key] for key in
                       ("half_block_hit", "early_hit", "late_hit", "missing_hit")) - 1) < 1e-6


def test_uneven_microbatches_preserve_loss_and_gradients(monkeypatch):
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    torch.manual_seed(3)
    full = ProjectiveFillArgs(5, voxel_size=8, image_channels=8, channels=8).build()
    split = deepcopy(full)
    target = torch.randint(0, 5, (3, 8, 8, 8))
    valid = torch.ones_like(target, dtype=torch.bool)
    b = dict(target=target, target_valid=valid, surface=target != 0, free=target == 0,
        voxel_context=torch.zeros_like(target), known_mask=~valid, fill_mask=valid,
        images=torch.randn(3, 1, 3, 32, 32), agent_mask=torch.ones(3, 1, dtype=torch.bool),
        camera_position=torch.tensor([[[-.6, 0., 0.]]]).expand(3, -1, -1),
        camera_direction=torch.tensor([[[1., 0., 0.]]]).expand(3, -1, -1),
        fov_x=torch.ones(3, 1), fov_y=torch.ones(3, 1))
    a = backward_batch(full, b, .1, 0)
    c = backward_batch(split, b, .1, 2)
    assert torch.allclose(a, c, atol=1e-5, rtol=1e-5)
    for p, q in zip(full.parameters(), split.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5, rtol=1e-4)


def test_world_resize_preserves_global_batch_cursor_without_padding():
    dataset = range(46200)
    def batches(world, local_batch):
        shards = []
        for rank in range(world):
            sampler = DistributedSampler(dataset, num_replicas=world, rank=rank, seed=20260910)
            sampler.set_epoch(5)
            shards.append(list(sampler))
        return [sorted(i for shard in shards for i in shard[start:start+local_batch])
                for start in range(0, len(shards[0]), local_batch)]
    assert batches(4, 32) == batches(8, 16)
