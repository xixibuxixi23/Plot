import tempfile
import unittest
from pathlib import Path

import numpy as np

from plot.geometry import coverage_mask, crop_raw_49_to_tile_48, tile_bounds
from plot.world_memory import WorldMemory


class GeometryMemoryTest(unittest.TestCase):
    def test_persist_half_open_convention(self):
        lower, upper = tile_bounds((10, 20, 30))
        np.testing.assert_array_equal(lower, (-14, -4, 6))
        np.testing.assert_array_equal(upper, (34, 44, 54))
        raw = np.arange(49**3).reshape(49, 49, 49, 1)
        cropped = crop_raw_49_to_tile_48(raw)
        self.assertEqual(cropped.shape, (48, 48, 48, 1))
        self.assertEqual(cropped[24, 24, 24, 0], raw[24, 24, 24, 0])

    def test_irregular_memory_round_trip(self):
        memory = WorldMemory(chunk_size=4)
        values = np.full((8, 8, 8), 3, dtype=np.int32)
        mask = np.zeros_like(values, dtype=bool)
        mask[:2, :, :] = True
        self.assertEqual(memory.commit_tile((0, 0, 0), values, mask), int(mask.sum()))
        blocks, known = memory.read_tile((0, 0, 0), size=8)
        np.testing.assert_array_equal(known, mask)
        np.testing.assert_array_equal(blocks[known], 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.npz"
            memory.save(path)
            restored = WorldMemory.load(path)
            self.assertEqual(restored.known_voxel_count, memory.known_voxel_count)

    def test_coverage_union(self):
        mask = coverage_mask(np.asarray([1, 0, 0]), np.asarray([[0, 0, 0]]), size=8)
        self.assertEqual(mask.shape, (8, 8, 8))
        self.assertEqual(int(mask.sum()), 7 * 8 * 8)

        # Repeated centers from a long trajectory must not change the union.
        repeated = coverage_mask(
            np.asarray([1, 0, 0]), np.repeat([[0, 0, 0]], 1_000, axis=0), size=8
        )
        np.testing.assert_array_equal(repeated, mask)


if __name__ == "__main__":
    unittest.main()
