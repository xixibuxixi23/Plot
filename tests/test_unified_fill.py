import numpy as np
import torch
from torch.utils.data import DataLoader

from plot.data import SyntheticFillDataset
from plot.data.fill_dataset import select_frontier_observation, use_image_condition
from plot.geometry import coverage_mask
from plot.models import FillNetwork, masked_fill_loss, projection_consistency_loss
from plot.models.projection import first_hit_projection_loss
from plot.training import FillTrainer, FillTrainerConfig


def test_all_image_conditioning_disabled_is_finite():
    model = FillNetwork(num_block_classes=7, base_channels=8, image_feature_dim=32)
    batch = 2
    output = model(
        torch.zeros(batch, 48, 48, 48, dtype=torch.long),
        torch.zeros(batch, 48, 48, 48, dtype=torch.bool),
        torch.ones(batch, 48, 48, 48, dtype=torch.bool),
        torch.zeros(batch, 2, 3, 32, 32),
        torch.ones(batch, 2, dtype=torch.bool),
        torch.zeros(batch, dtype=torch.bool),
        return_aux=True,
    )
    assert torch.isfinite(output["voxel_logits"]).all()
    assert torch.isfinite(output["camera_position"]).all()
    assert torch.isfinite(output["camera_direction"]).all()


def test_frontier_sampling_selects_first_unseen_fringe():
    centers = np.asarray(
        [
            [[0, 0, 0], [100, 0, 0]],
            [[0, 0, 0], [100, 0, 0]],
            [[1, 0, 0], [100, 0, 0]],
        ],
        dtype=np.int64,
    )
    observation = select_frontier_observation(centers, 0, 0, 1, 2)
    assert observation == 2
    known = coverage_mask(centers[observation, 0], centers[:observation].reshape(-1, 3))
    assert 0 < known.sum() < known.size


def test_frontier_sampling_falls_back_to_initial_state_without_motion():
    centers = np.zeros((4, 2, 3), dtype=np.int64)
    assert select_frontier_observation(centers, 0, 0, 1, 2) == 0


def test_optional_image_schedule_keeps_initial_and_one_of_three_frontiers():
    assert [use_image_condition(slot, 4, 1 / 3) for slot in range(4)] == [
        True, False, False, True
    ]


def test_first_hit_loss_prefers_a_surface_at_the_target_depth():
    size = 8
    target = torch.zeros(1, size, size, size, dtype=torch.long)
    target[:, 5] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    fill = valid.clone()
    correct = torch.full_like(target, -8.0, dtype=torch.float32)
    correct[:, 5] = 8.0
    missing = torch.full_like(correct, -8.0)
    camera_position = torch.zeros(1, 1, 3)
    camera_direction = torch.tensor([[[1.0, 0.0, 0.0]]])
    camera_valid = torch.ones(1, 1, dtype=torch.bool)
    fov = torch.full((1, 1), 0.1)

    def loss(logits):
        return first_hit_projection_loss(
            logits, target, fill, valid, camera_position, camera_direction,
            camera_valid, fov, fov, air_class=0, height=1, width=1,
            samples=32, max_distance=4.0, supervised_height_fraction=1.0,
        )["projection_surface_loss"]

    assert loss(correct) < loss(missing)


def test_projection_loss_is_differentiable():
    size = 8
    logits = torch.zeros(1, 3, size, size, size, requires_grad=True)
    target = torch.zeros(1, size, size, size, dtype=torch.long)
    target[:, :, 5, :] = 1
    fill = torch.ones_like(target, dtype=torch.bool)
    valid = torch.ones_like(fill)
    camera_position = torch.tensor([[[0.0, -0.25, 0.0]]])
    camera_direction = torch.tensor([[[0.0, 1.0, 0.0]]])
    camera_valid = torch.ones(1, 1, dtype=torch.bool)
    fov = torch.ones(1, 1)
    silhouette, depth = projection_consistency_loss(
        logits, target, fill, valid, camera_position, camera_direction,
        camera_valid, fov, fov, air_class=0, height=6, width=8,
        samples=16, max_distance=10.0,
    )
    (silhouette + depth).backward()
    assert torch.isfinite(silhouette) and torch.isfinite(depth)
    assert float(logits.grad.abs().sum()) > 0.0


def test_forward_loss_and_update():
    dataset = SyntheticFillDataset(length=1, size=8, agents=2, classes=6)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = FillNetwork(6, voxel_embedding_dim=4, image_feature_dim=8, base_channels=4)
    logits = model(
        batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
        batch["images"], batch["agent_mask"],
    )
    assert logits.shape == (1, 6, 8, 8, 8)
    output = model(
        batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
        batch["images"], batch["agent_mask"], return_aux=True,
    )
    assert output["camera_position"].shape == (1, 2, 3)
    assert output["camera_direction"].shape == (1, 2, 3)
    loss = masked_fill_loss(logits, batch["target"], batch["fill_mask"], batch["target_valid"])
    assert torch.isfinite(loss)
    trainer = FillTrainer(model, FillTrainerConfig(device="cpu"))
    assert trainer.train_step(batch) > 0
