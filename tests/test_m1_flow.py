import unittest

import torch
from torch.utils.data import DataLoader

from plot.data import SyntheticFillDataset
from plot.models import FillNetwork, masked_fill_loss, projection_consistency_loss
from plot.models.projection import first_hit_projection_loss
from plot.training import FillTrainer, FillTrainerConfig


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


class M1FlowTest(unittest.TestCase):
    def test_projection_loss_is_differentiable(self):
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
            logits,
            target,
            fill,
            valid,
            camera_position,
            camera_direction,
            camera_valid,
            fov,
            fov,
            air_class=0,
            height=6,
            width=8,
            samples=16,
            max_distance=10.0,
        )
        (silhouette + depth).backward()
        self.assertTrue(torch.isfinite(silhouette) and torch.isfinite(depth))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_forward_loss_and_update(self):
        dataset = SyntheticFillDataset(length=1, size=8, agents=2, classes=6)
        batch = next(iter(DataLoader(dataset, batch_size=1)))
        model = FillNetwork(6, voxel_embedding_dim=4, image_feature_dim=8, base_channels=4)
        logits = model(
            batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
            batch["images"], batch["agent_mask"],
        )
        self.assertEqual(logits.shape, (1, 6, 8, 8, 8))
        output = model(
            batch["voxel_context"], batch["known_mask"], batch["fill_mask"],
            batch["images"], batch["agent_mask"], return_aux=True,
        )
        self.assertEqual(output["camera_position"].shape, (1, 2, 3))
        self.assertEqual(output["camera_direction"].shape, (1, 2, 3))
        loss = masked_fill_loss(
            logits, batch["target"], batch["fill_mask"], batch["target_valid"]
        )
        self.assertTrue(torch.isfinite(loss))
        trainer = FillTrainer(model, FillTrainerConfig(device="cpu"))
        self.assertGreater(trainer.train_step(batch), 0)


if __name__ == "__main__":
    unittest.main()
