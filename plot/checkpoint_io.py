"""Checkpoint writes that remain safe on restricted FUSE/OSS mounts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
import zipfile

import torch


def _sync_file(path: Path, *, required: bool = True) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        if required:
            raise


def _verify_archive(path: Path) -> None:
    """Check every zip member without allocating a second model/optimizer."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            corrupt = archive.testzip()
        if corrupt is not None:
            raise OSError(f"corrupt checkpoint member: {corrupt}")
    else:
        torch.load(path, map_location="cpu", weights_only=False)


def staged_torch_save(
    payload: Any,
    destination: str | Path,
    *,
    staging_dir: str | Path | None = None,
    verify: bool = True,
) -> Path:
    """Serialize on a normal local/PFS filesystem, then copy the finished file.

    Some OSS FUSE clients do not implement the seek/mmap/rename behavior used by
    tensor serializers.  Serialization therefore never touches ``destination``.
    Only a completed ordinary file is copied to that mount.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(
        staging_dir or os.environ.get("PLOT_CHECKPOINT_STAGING_DIR", tempfile.gettempdir())
    )
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="plot-checkpoint-", dir=root) as temporary_dir:
        local = Path(temporary_dir) / destination.name
        torch.save(payload, local)
        _sync_file(local)
        if verify:
            _verify_archive(local)

        remote_partial = destination.with_name(destination.name + ".uploading")
        try:
            shutil.copyfile(local, remote_partial)
            _sync_file(remote_partial, required=False)
            try:
                os.replace(remote_partial, destination)
            except OSError:
                # Restricted object-store mounts may reject rename.  A plain
                # copy is still safe because ``local`` is already complete.
                shutil.copyfile(local, destination)
                _sync_file(destination, required=False)
                remote_partial.unlink(missing_ok=True)
        finally:
            remote_partial.unlink(missing_ok=True)
    return destination


def record_checkpoint_failure(
    output_dir: str | Path, destination: str | Path, error: Exception
) -> Path:
    """Persist a machine-readable failure without hiding the training problem."""
    path = Path(output_dir) / "checkpoint_failures.jsonl"
    row = {
        "time_unix": time.time(),
        "destination": str(destination),
        "error_type": type(error).__name__,
        "error": str(error),
    }
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    return path
