import math
import unittest

import torch
from torch.nn import functional as F

from plot.models import GeometryBootstrap, geometry_targets_from_voxels
from plot.camera_geometry import canonicalize_cameras
from plot.training import GeometryBootstrapTrainerConfig, geometry_bootstrap_loss


class GeometryBootstrapTest(unittest.TestCase):
    def test_zero_initialized_multiscale_branch_preserves_warm_start(self):
        kwargs = dict(
            feature_dim=24, attention_heads=4, stages=1,
            patch_height=3, patch_width=4, output_height=6, output_width=8,
            max_views=2,
        )
        torch.manual_seed(7)
        baseline = GeometryBootstrap(**kwargs).eval()
        torch.manual_seed(7)
        multiscale = GeometryBootstrap(**kwargs, multiscale_highres=True).eval()
        incompatible = multiscale.load_state_dict(baseline.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        images = torch.rand(1, 2, 3, 24, 32)
        mask = torch.ones(1, 2, dtype=torch.bool)
        baseline_output = baseline(images, mask)
        multiscale_output = multiscale(images, mask)
        for key in ("depth", "point_map", "visibility_logits", "dense_features"):
            torch.testing.assert_close(baseline_output[key], multiscale_output[key])

    def test_first_camera_gauge_is_invariant_to_global_yaw_and_translation(self):
        position = torch.tensor([[[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]]])
        direction = torch.tensor([[[0.6, 0.8, -0.1], [-0.2, 0.9, 0.3]]])
        translation, rotation = canonicalize_cameras(position, direction)
        angle = 0.7
        yaw = torch.tensor([
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        shifted_position = torch.einsum("ij,bvj->bvi", yaw, position)
        shifted_position = shifted_position + torch.tensor([[[7.0, -3.0, 2.0]]])
        shifted_direction = torch.einsum("ij,bvj->bvi", yaw, direction)
        shifted_translation, shifted_rotation = canonicalize_cameras(
            shifted_position, shifted_direction
        )
        torch.testing.assert_close(translation, shifted_translation, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(rotation, shifted_rotation, atol=1e-5, rtol=1e-5)

    def test_highres_semantic_head_preserves_half_resolution_features(self):
        model = GeometryBootstrap(
            feature_dim=24, attention_heads=4, stages=1,
            patch_height=3, patch_width=4, output_height=6, output_width=8,
            max_views=2, num_block_classes=5, multiscale_highres=True,
            highres_semantic_head=True,
        )
        output = model(torch.rand(1, 2, 3, 24, 32), torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(output["highres_pixel_semantic_logits"].shape, (1, 2, 5, 12, 16))

    def test_highres_point_refiner_starts_as_bilinear_point_map(self):
        model = GeometryBootstrap(
            feature_dim=24, attention_heads=4, stages=1,
            patch_height=3, patch_width=4, output_height=6, output_width=8,
            max_views=2, num_block_classes=5, multiscale_highres=True,
            highres_semantic_head=True, highres_point_refinement=True,
        )
        output = model(torch.rand(1, 2, 3, 24, 32), torch.ones(1, 2, dtype=torch.bool))
        expected = F.interpolate(
            output["point_map"].reshape(2, 3, 6, 8), (12, 16),
            mode="bilinear", align_corners=False,
        ).reshape(1, 2, 3, 12, 16)
        torch.testing.assert_close(output["highres_point_map"], expected)

    def test_targets_model_and_backward(self):
        size = 8
        target = torch.zeros(1, size, size, size, dtype=torch.long)
        target[:, :, 5, :] = 1
        valid = torch.ones_like(target, dtype=torch.bool)
        camera_position = torch.tensor([[[0.0, -0.25, 0.0], [0.125, -0.25, 0.0]]])
        camera_direction = torch.tensor([[[0.0, 1.0, 0.0], [0.1, 1.0, 0.0]]])
        camera_valid = torch.ones(1, 2, dtype=torch.bool)
        fov = torch.ones(1, 2)
        geometry_target = geometry_targets_from_voxels(
            target, valid, camera_position, camera_direction, camera_valid, fov, fov,
            air_class=0, height=6, width=8, samples=16, max_distance=10.0,
            return_semantics=True,
        )
        torch.testing.assert_close(
            geometry_target["relative_translation"][:, 0], torch.zeros(1, 3)
        )
        torch.testing.assert_close(
            geometry_target["relative_rotation"][:, 0], torch.eye(3)[None],
            atol=1e-5, rtol=1e-5,
        )
        self.assertEqual(geometry_target["point_map"].shape, (1, 2, 3, 6, 8))
        self.assertEqual(geometry_target["semantic_class"].shape, (1, 2, 6, 8))
        self.assertTrue((geometry_target["semantic_class"][geometry_target["visibility"].bool()] == 1).all())
        self.assertTrue(torch.isfinite(geometry_target["point_map"]).all())

        model = GeometryBootstrap(
            feature_dim=24, attention_heads=4, stages=1,
            patch_height=3, patch_width=4, output_height=6, output_width=8,
            max_views=2, num_block_classes=5,
        )
        images = torch.rand(1, 2, 3, 24, 32)
        output = model(images, camera_valid)
        self.assertEqual(output["depth"].shape, (1, 2, 6, 8))
        self.assertEqual(output["point_map"].shape, (1, 2, 3, 6, 8))
        self.assertEqual(output["pixel_semantic_logits"].shape, (1, 2, 5, 6, 8))
        total, metrics = geometry_bootstrap_loss(
            output, geometry_target, GeometryBootstrapTrainerConfig(device="cpu")
        )
        total.backward()
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(model.image_encoder[0].weight.grad.abs().sum()), 0.0)
        self.assertIn("rotation_loss", metrics)


if __name__ == "__main__":
    unittest.main()
