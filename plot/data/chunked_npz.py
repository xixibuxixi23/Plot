"""Frame-chunked NPZ sidecars for M3's large temporal arrays.

The source archive is never modified.  Large arrays are split only along the
time axis, while all other arrays retain ordinary NPZ semantics.  This keeps
the cache rebuildable and lets M3 decompress just the chunks intersecting a
training window.
"""
from __future__ import annotations

import os
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1
DEFAULT_CHUNKED_KEYS = ("instance_mask", "obs_voxel_mt")
_SCHEMA_KEY = "__m3_chunk_schema__"
_CHUNK_FRAMES_KEY = "__m3_chunk_frames__"
_CHUNKED_KEYS_KEY = "__m3_chunked_keys__"


def _shape_key(key):
    return f"__m3_shape_{key}__"


def _chunk_key(key, index):
    return f"{key}__m3_chunk_{index:06d}"


def _write_npy(archive, key, value):
    with archive.open(f"{key}.npy", "w", force_zip64=True) as handle:
        np.lib.format.write_array(handle, np.asanyarray(value), allow_pickle=False)


class ChunkedArray:
    """Lazy first-axis view over one chunked array in an NPZ archive."""

    def __init__(self, archive, key, shape, chunk_frames):
        self._archive = archive
        self._key = str(key)
        self.shape = tuple(int(value) for value in shape)
        self.ndim = len(self.shape)
        self._chunk_frames = int(chunk_frames)

    def __len__(self):
        return self.shape[0]

    @property
    def dtype(self):
        if not self.shape[0]:
            raise ValueError("empty chunked arrays do not expose a lazy dtype")
        return self._archive[_chunk_key(self._key, 0)].dtype

    def _read_span(self, start, stop):
        if start == stop:
            exemplar = self._archive[_chunk_key(self._key, 0)]
            return np.empty((0, *self.shape[1:]), dtype=exemplar.dtype)
        first = start // self._chunk_frames
        last = (stop - 1) // self._chunk_frames
        pieces = [self._archive[_chunk_key(self._key, index)]
                  for index in range(first, last + 1)]
        joined = pieces[0] if len(pieces) == 1 else np.concatenate(pieces, axis=0)
        offset = start - first * self._chunk_frames
        return joined[offset:offset + stop - start]

    def __getitem__(self, selector):
        selectors = selector if isinstance(selector, tuple) else (selector,)
        if not selectors:
            return self[:]
        first, rest = selectors[0], selectors[1:]
        if first is Ellipsis:
            return self[:][selector]
        if isinstance(first, (int, np.integer)):
            index = int(first)
            if index < 0:
                index += self.shape[0]
            if not 0 <= index < self.shape[0]:
                raise IndexError(index)
            result = self._read_span(index, index + 1)[0]
            return result[rest] if rest else result
        if not isinstance(first, slice):
            raise TypeError("chunked arrays support an integer or slice on the time axis")
        start, stop, step = first.indices(self.shape[0])
        if step != 1:
            raise ValueError("chunked arrays require a contiguous first-axis slice")
        result = self._read_span(start, stop)
        return result[(slice(None), *rest)] if rest else result


class ChunkedNpzFile(Mapping):
    """Mapping-compatible reader for an M3 frame-chunked NPZ archive."""

    def __init__(self, archive):
        self._archive = archive
        self.schema_version = int(np.asarray(archive[_SCHEMA_KEY]).item())
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported M3 chunk schema {self.schema_version}; expected {SCHEMA_VERSION}"
            )
        self.chunk_frames = int(np.asarray(archive[_CHUNK_FRAMES_KEY]).item())
        self.chunked_keys = tuple(np.asarray(archive[_CHUNKED_KEYS_KEY]).astype(str).tolist())
        hidden = {_SCHEMA_KEY, _CHUNK_FRAMES_KEY, _CHUNKED_KEYS_KEY}
        hidden.update(_shape_key(key) for key in self.chunked_keys)
        self.files = [key for key in archive.files
                      if key not in hidden and "__m3_chunk_" not in key]
        self.files.extend(key for key in self.chunked_keys if key not in self.files)

    def __getitem__(self, key):
        if key in self.chunked_keys:
            shape = np.asarray(self._archive[_shape_key(key)], dtype=np.int64)
            return ChunkedArray(self._archive, key, shape, self.chunk_frames)
        return self._archive[key]

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def close(self):
        self._archive.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open_npz(path):
    """Open either a standard NPZ or an M3 chunked NPZ with one API."""
    archive = np.load(path, allow_pickle=False)
    if _SCHEMA_KEY in archive.files:
        try:
            return ChunkedNpzFile(archive)
        except Exception:
            archive.close()
            raise
    return archive


def write_chunked_npz(source, destination, *, chunk_frames=8,
                      chunked_keys=DEFAULT_CHUNKED_KEYS, compression_level=6):
    """Materialize a standalone, atomic chunk cache beside immutable source data."""
    source, destination = Path(source), Path(destination)
    chunk_frames = int(chunk_frames)
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    if not 0 <= int(compression_level) <= 9:
        raise ValueError("compression_level must be between 0 and 9")
    chunked_keys = tuple(str(key) for key in chunked_keys)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with np.load(source, allow_pickle=False) as source_data:
            missing = [key for key in chunked_keys if key not in source_data.files]
            if missing:
                raise KeyError(f"source archive lacks chunk keys: {missing}")
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=int(compression_level), allowZip64=True,
            ) as archive:
                _write_npy(archive, _SCHEMA_KEY, np.asarray(SCHEMA_VERSION, np.int64))
                _write_npy(archive, _CHUNK_FRAMES_KEY, np.asarray(chunk_frames, np.int64))
                _write_npy(archive, _CHUNKED_KEYS_KEY, np.asarray(chunked_keys, dtype=np.str_))
                for key in source_data.files:
                    value = source_data[key]
                    if key not in chunked_keys:
                        _write_npy(archive, key, value)
                        continue
                    if value.ndim < 1:
                        raise ValueError(f"chunked array {key!r} needs a time axis")
                    _write_npy(archive, _shape_key(key), np.asarray(value.shape, np.int64))
                    for index, start in enumerate(range(0, len(value), chunk_frames)):
                        _write_npy(
                            archive, _chunk_key(key, index),
                            value[start:start + chunk_frames],
                        )
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def validate_chunked_npz(path, *, expected_chunk_frames=None,
                         expected_keys=DEFAULT_CHUNKED_KEYS):
    """Return True only for a complete cache matching the requested layout."""
    try:
        with open_npz(path) as archive:
            if not isinstance(archive, ChunkedNpzFile):
                return False
            if expected_chunk_frames is not None and archive.chunk_frames != int(expected_chunk_frames):
                return False
            if tuple(archive.chunked_keys) != tuple(expected_keys):
                return False
            raw_files = set(archive._archive.files)
            for key in archive.chunked_keys:
                shape = archive[key].shape
                count = (shape[0] + archive.chunk_frames - 1) // archive.chunk_frames
                if _shape_key(key) not in raw_files:
                    return False
                if any(_chunk_key(key, index) not in raw_files for index in range(count)):
                    return False
        return True
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False
