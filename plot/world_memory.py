from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import TILE_SIZE, tile_bounds


@dataclass
class _Chunk:
    block_ids: np.ndarray
    known: np.ndarray


class WorldMemory:
    """Unbounded sparse voxel memory backed by fixed-size dense chunks.

    Block IDs are model vocabulary IDs. Unknown and air are distinct: unknown is
    represented by ``known=False`` while air is an ordinary known block class.
    """

    def __init__(self, chunk_size: int = 16):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = int(chunk_size)
        self._chunks: dict[tuple[int, int, int], _Chunk] = {}

    def _chunk(self, key: tuple[int, int, int], create: bool = False) -> _Chunk | None:
        chunk = self._chunks.get(key)
        if chunk is None and create:
            shape = (self.chunk_size,) * 3
            chunk = _Chunk(np.zeros(shape, dtype=np.int32), np.zeros(shape, dtype=bool))
            self._chunks[key] = chunk
        return chunk

    def _point_key(self, xyz: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        chunk_coord = np.floor_divide(xyz, self.chunk_size)
        local = xyz - chunk_coord * self.chunk_size
        return tuple(int(v) for v in chunk_coord), tuple(int(v) for v in local)

    def read_tile(
        self, center: np.ndarray | tuple[int, int, int], size: int = TILE_SIZE
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, _ = tile_bounds(center, size)
        return self.read_region(lower, (size, size, size))

    def read_region(self, lower, shape):
        """Read arbitrary half-open integer regions, including M2's odd 13³ crop."""
        lower = np.asarray(lower, dtype=np.int64)
        shape = tuple(int(v) for v in shape)
        if lower.shape != (3,) or len(shape) != 3 or min(shape) < 1:
            raise ValueError("region requires a 3D lower coordinate and positive shape")
        blocks = np.zeros(shape, dtype=np.int32)
        known = np.zeros_like(blocks, dtype=bool)
        for local in np.ndindex(blocks.shape):
            xyz = lower + np.asarray(local)
            key, offset = self._point_key(xyz)
            chunk = self._chunk(key)
            if chunk is not None and chunk.known[offset]:
                blocks[local] = chunk.block_ids[offset]
                known[local] = True
        return blocks, known

    def commit_points(
        self,
        coordinates: np.ndarray,
        block_ids: np.ndarray,
        *,
        allow_overwrite: bool = False,
    ) -> int:
        coords = np.asarray(coordinates, dtype=np.int64).reshape(-1, 3)
        values = np.asarray(block_ids, dtype=np.int32).reshape(-1)
        if len(coords) != len(values):
            raise ValueError("coordinates and block_ids have different lengths")
        written = 0
        for xyz, value in zip(coords, values):
            key, offset = self._point_key(xyz)
            chunk = self._chunk(key, create=True)
            assert chunk is not None
            if chunk.known[offset] and not allow_overwrite:
                continue
            chunk.block_ids[offset] = value
            if not chunk.known[offset]:
                written += 1
            chunk.known[offset] = True
        return written

    def commit_tile(
        self,
        center: np.ndarray | tuple[int, int, int],
        block_ids: np.ndarray,
        fill_mask: np.ndarray,
    ) -> int:
        values = np.asarray(block_ids)
        mask = np.asarray(fill_mask, dtype=bool)
        if values.shape != mask.shape or values.ndim != 3:
            raise ValueError("block_ids and fill_mask must be matching 3D arrays")
        lower, _ = tile_bounds(center, values.shape[0])
        local = np.argwhere(mask)
        return self.commit_points(lower + local, values[mask])

    @property
    def known_voxel_count(self) -> int:
        return sum(int(chunk.known.sum()) for chunk in self._chunks.values())

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def save(self, path: str | Path) -> None:
        keys, blocks, known = [], [], []
        for key in sorted(self._chunks):
            keys.append(key)
            blocks.append(self._chunks[key].block_ids)
            known.append(self._chunks[key].known)
        shape = (0, self.chunk_size, self.chunk_size, self.chunk_size)
        np.savez_compressed(
            path,
            chunk_size=np.asarray(self.chunk_size),
            keys=np.asarray(keys, dtype=np.int64).reshape(-1, 3),
            block_ids=np.stack(blocks) if blocks else np.empty(shape, dtype=np.int32),
            known=np.stack(known) if known else np.empty(shape, dtype=bool),
        )

    @classmethod
    def load(cls, path: str | Path) -> "WorldMemory":
        with np.load(path, allow_pickle=False) as data:
            memory = cls(int(data["chunk_size"]))
            for key, blocks, known in zip(data["keys"], data["block_ids"], data["known"]):
                memory._chunks[tuple(int(v) for v in key)] = _Chunk(blocks.copy(), known.copy())
        return memory
