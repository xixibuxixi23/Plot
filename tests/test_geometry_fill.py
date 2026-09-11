import unittest

import torch
import numpy as np

from plot.models import GeometryConditionedFillNetwork, decode_voxel_prediction, trilinear_splat
from plot.training import GeometryFillTrainerConfig, geometry_fill_loss


class GeometryFillTest(unittest.TestCase):
    def test_multiscale_only_training_freezes_geometry_backbone(self):
        model = GeometryConditionedFillNetwork(
            num_block_classes=5, voxel_size=8, max_distance=8.0,
            voxel_embedding_dim=8, splat_channels=4, base_channels=4,
            geometry_feature_dim=24, geometry_attention_heads=4,
            geometry_stages=1, geometry_output_height=4,
            geometry_output_width=6, max_views=2,
            geometry_multiscale_highres=True,
        )
        model.train_geometry_multiscale_only()
        geometry_trainable = {
            name for name, parameter in model.geometry.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(geometry_trainable)
        self.assertTrue(all(
            name.startswith(("shallow_adapter.", "middle_adapter.", "multiscale_fusion."))
            for name in geometry_trainable
        ))
        self.assertFalse(model.geometry.image_encoder[0].weight.requires_grad)

    def test_explicit_occupancy_decodes_before_material(self):
        logits = torch.tensor([[[[[4.0]]], [[[1.0]]], [[[2.0]]]]])
        output = {"voxel_logits": logits, "occupancy_logits": torch.ones(1, 1, 1, 1, 1)}
        self.assertEqual(int(decode_voxel_prediction(output, air_class=0).item()), 2)
        output["occupancy_logits"] = -torch.ones(1, 1, 1, 1, 1)
        self.assertEqual(int(decode_voxel_prediction(output, air_class=0).item()), 0)

    def test_highres_semantic_only_freezes_everything_else(self):
        model = GeometryConditionedFillNetwork(
            num_block_classes=5, voxel_size=8, max_distance=8.0,
            voxel_embedding_dim=8, splat_channels=4, base_channels=4,
            geometry_feature_dim=24, geometry_attention_heads=4,
            geometry_stages=1, geometry_output_height=4,
            geometry_output_width=6, max_views=2,
            geometry_multiscale_highres=True, highres_pixel_semantic_head=True,
        )
        model.train_highres_semantic_only()
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        self.assertTrue(trainable)
        self.assertTrue(all("geometry.highres_semantic_" in name for name in trainable))
        self.assertFalse(model.geometry.image_encoder[0].weight.requires_grad)
        self.assertFalse(model.classifier.weight.requires_grad)

    def test_highres_semantic_splat_backpropagates_to_embedding(self):
        model = GeometryConditionedFillNetwork(
            num_block_classes=5, voxel_size=8, max_distance=8.0,
            voxel_embedding_dim=8, splat_channels=4, base_channels=4,
            geometry_feature_dim=24, geometry_attention_heads=4,
            geometry_stages=1, geometry_output_height=4,
            geometry_output_width=6, max_views=2,
            geometry_multiscale_highres=True, highres_pixel_semantic_head=True,
            highres_pixel_semantic_splat=True,
        )
        shape = (1, 8, 8, 8)
        output = model(
            torch.zeros(shape, dtype=torch.long),
            torch.zeros(shape, dtype=torch.bool), torch.ones(shape, dtype=torch.bool),
            torch.rand(1, 2, 3, 24, 32), torch.ones(1, 2, dtype=torch.bool),
            return_aux=True,
        )
        output["voxel_logits"].square().mean().backward()
        self.assertGreater(
            float(model.highres_pixel_semantic_embedding.grad.abs().sum()), 0.0
        )

    def test_direct_visible_head_is_zero_start_and_isolated(self):
        kwargs = dict(
            num_block_classes=5, voxel_size=8, max_distance=8.0,
            voxel_embedding_dim=8, splat_channels=4, base_channels=4,
            geometry_feature_dim=24, geometry_attention_heads=4,
            geometry_stages=1, geometry_output_height=4,
            geometry_output_width=6, max_views=2,
            geometry_multiscale_highres=True, highres_pixel_semantic_head=True,
        )
        torch.manual_seed(11)
        baseline = GeometryConditionedFillNetwork(**kwargs).eval()
        torch.manual_seed(11)
        direct = GeometryConditionedFillNetwork(
            **kwargs, direct_highres_visible_head=True
        ).eval()
        direct.load_state_dict(baseline.state_dict(), strict=False)
        shape = (1, 8, 8, 8)
        inputs = (
            torch.zeros(shape, dtype=torch.long), torch.zeros(shape, dtype=torch.bool),
            torch.ones(shape, dtype=torch.bool), torch.rand(1, 2, 3, 24, 32),
            torch.ones(1, 2, dtype=torch.bool),
        )
        torch.testing.assert_close(baseline(*inputs), direct(*inputs))
        direct.train_direct_visible_only()
        trainable = {name for name, p in direct.named_parameters() if p.requires_grad}
        self.assertEqual(trainable, {
            "direct_visible_embedding", "direct_visible_classifier.weight",
            "direct_visible_classifier.bias",
        })

    def test_canonical_yaw_rotation_preserves_lattice(self):
        direction = np.asarray([[0.0, 1.0, -0.25], [0.0, -1.0, -0.25]], dtype=np.float32)
        yaw = np.arctan2(direction[0, 1], direction[0, 0])
        quarter_turns = int(np.rint(yaw / (0.5 * np.pi)))
        angle = -quarter_turns * 0.5 * np.pi
        rotation = np.asarray(
            ((np.cos(angle), -np.sin(angle), 0.0),
             (np.sin(angle), np.cos(angle), 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        )
        canonical = direction @ rotation.T
        self.assertGreater(canonical[0, 0], 0.0)
        self.assertAlmostEqual(float(canonical[0, 1]), 0.0, places=6)

    def test_trilinear_splat_conserves_weight_and_interpolates(self):
        features = torch.tensor([[[2.0, 4.0]]])
        positions = torch.tensor([[[1.5, 1.5, 1.5]]])
        weights = torch.ones(1, 1)
        volume, support = trilinear_splat(features, positions, weights, (4, 4, 4))
        self.assertAlmostEqual(float(support.sum()), 1.0, places=6)
        self.assertEqual(int((support > 0).sum()), 8)
        for channel, expected in enumerate((2.0, 4.0)):
            torch.testing.assert_close(
                volume[0, channel][support[0, 0] > 0],
                torch.full((8,), expected),
            )

    def test_geometry_conditioned_fill_shapes_and_backward(self):
        model = GeometryConditionedFillNetwork(
            num_block_classes=5, voxel_size=8, max_distance=8.0,
            voxel_embedding_dim=8, splat_channels=4, base_channels=4,
            geometry_feature_dim=24, geometry_attention_heads=4,
            geometry_stages=1, geometry_output_height=4,
            geometry_output_width=6, max_views=2, full_resolution_surface=True,
            explicit_occupancy=True,
            visibility_evidence=True, free_space_samples=4,
            adaptive_visibility_fusion=True,
            pixel_semantic_head=True, pixel_semantic_splat=True,
        )
        context = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        known = torch.zeros_like(context, dtype=torch.bool)
        fill = torch.ones_like(known)
        images = torch.rand(1, 2, 3, 24, 32)
        view_mask = torch.ones(1, 2, dtype=torch.bool)
        camera_position = torch.tensor([[[0.05, 0.0, 0.1], [0.2, 0.0, 0.1]]])
        camera_direction = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]])
        output = model(
            context, known, fill, images, view_mask,
            splat_camera_position=camera_position,
            splat_camera_direction=camera_direction,
            return_aux=True,
        )
        self.assertEqual(output["voxel_logits"].shape, (1, 5, 8, 8, 8))
        self.assertEqual(output["occupancy_logits"].shape, (1, 1, 8, 8, 8))
        self.assertEqual(output["splat_features"].shape, (1, 4, 4, 4, 4))
        self.assertEqual(output["surface_support"].shape, (1, 1, 8, 8, 8))
        self.assertEqual(output["free_space_support"].shape, (1, 1, 8, 8, 8))
        self.assertEqual(output["visibility_residual"].shape, (1, 1, 8, 8, 8))
        self.assertEqual(output["camera_position"].shape, (1, 2, 3))
        loss = output["voxel_logits"].square().mean()
        loss.backward()
        self.assertGreater(float(model.splat_projection.weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(model.pixel_semantic_embedding.grad.abs().sum()), 0.0)

        batch = {
            "target": torch.randint(0, 5, (1, 8, 8, 8)),
            "target_valid": torch.ones_like(fill),
            "fill_mask": fill,
            "camera_position": camera_position,
            "camera_direction": camera_direction,
            "camera_valid": view_mask,
            "agent_mask": view_mask,
        }
        total, metrics = geometry_fill_loss(
            output, batch, GeometryFillTrainerConfig(device="cpu", air_class=0)
        )
        self.assertTrue(torch.isfinite(total))
        self.assertIn("occupancy_loss", metrics)


if __name__ == "__main__":
    unittest.main()
