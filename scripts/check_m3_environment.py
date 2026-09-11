#!/usr/bin/env python3
"""Fail-fast checks for moving an M3 run to a CUDA/Blackwell machine."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from plot.checkpoint_io import staged_torch_save


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("+")[0].split(".")[:3])


def inspect_environment(device: str, *, exercise_rasterizer: bool = True) -> dict:
    modules = ("cv2", "einops", "loguru", "safetensors", "timm", "utils3d", "wandb")
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "imports": {},
        "errors": [],
        "warnings": [],
    }
    for name in modules:
        try:
            importlib.import_module(name)
            report["imports"][name] = "ok"
        except Exception as error:
            report["imports"][name] = f"{type(error).__name__}: {error}"
            report["errors"].append(f"cannot import {name}")
    if version_tuple(torch.__version__) < (2, 7):
        report["errors"].append("M3 B200 recipe requires PyTorch >=2.7 with a CUDA 12.8 build")

    requested = torch.device(device)
    if requested.type == "cuda":
        if not torch.cuda.is_available():
            report["errors"].append("CUDA was requested but torch.cuda.is_available() is false")
        else:
            index = requested.index or 0
            capability = torch.cuda.get_device_capability(index)
            report.update(
                {
                    "device_name": torch.cuda.get_device_name(index),
                    "device_capability": list(capability),
                    "compiled_arches": torch.cuda.get_arch_list(),
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
            architecture = f"sm_{capability[0]}{capability[1]}"
            if architecture not in report["compiled_arches"]:
                report["errors"].append(
                    f"this PyTorch wheel lacks {architecture}; install its CUDA 12.8 build"
                )
            if not report["bf16_supported"]:
                report["errors"].append("selected device does not report BF16 support")
            if exercise_rasterizer:
                try:
                    from plot.models.renderer_backbone.voxel_rasterizer import RastContext

                    with torch.cuda.device(index):
                        RastContext(backend="cuda", device=index)
                    report["nvdiffrast_context"] = "ok"
                except Exception as error:
                    report["nvdiffrast_context"] = f"{type(error).__name__}: {error}"
                    report["errors"].append("nvdiffrast CUDA context could not be created")
    else:
        report["warnings"].append("CPU mode checks imports and checkpoint I/O only")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Also test completed-file copy to the actual output/OSS mount",
    )
    parser.add_argument("--checkpoint-staging-dir", type=Path)
    parser.add_argument("--skip-rasterizer", action="store_true")
    args = parser.parse_args()
    report = inspect_environment(args.device, exercise_rasterizer=not args.skip_rasterizer)
    output_root = args.output_dir or Path(tempfile.mkdtemp(prefix="plot-m3-io-check-"))
    destination = output_root / ".m3_checkpoint_io_probe.pt"
    try:
        staged_torch_save(
            {"probe": torch.arange(16)}, destination, staging_dir=args.checkpoint_staging_dir
        )
        loaded = torch.load(destination, map_location="cpu", weights_only=True)
        if not torch.equal(loaded["probe"], torch.arange(16)):
            raise RuntimeError("checkpoint content differs after copy")
        report["checkpoint_copy"] = "ok"
    except Exception as error:
        report["checkpoint_copy"] = f"{type(error).__name__}: {error}"
        report["errors"].append("staged checkpoint copy failed")
    finally:
        destination.unlink(missing_ok=True)
    print(json.dumps(report, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
