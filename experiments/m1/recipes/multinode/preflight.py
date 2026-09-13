"""Local provenance checks and optional NCCL smoke test before original M1 entry."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
from datetime import timedelta


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local():
    import numpy as np
    import torch

    root = Path(os.environ["PLOT_ROOT"])
    sys.path.insert(0, str(root))
    from experiments.m1.train_multiview_persist_full import Cache

    cache_path = Path(os.environ["CACHE_DIR"])
    c = Cache(cache_path)
    assert np.load(cache_path / "ready.npy", mmap_mode="r").all(), "Incomplete cache"
    assert 0 < int(os.environ["AUDIT_SAMPLES"]) <= len(c)
    assert 0 <= int(os.environ["FIXED_LOSS_SAMPLES"]) <= len(c)
    assert all(len(a) == len(c) for a in c.arrays.values()), "Cache length mismatch"
    assert torch.cuda.device_count() >= int(os.environ["GPUS_PER_NODE"]), "Not enough visible GPUs"
    assert all(
        torch.cuda.get_device_capability(i)[0] >= 8 for i in range(int(os.environ["GPUS_PER_NODE"]))
    ), "bf16 requires suitable GPUs"
    path = Path(os.environ["RESUME_CHECKPOINT"])
    s = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    assert s.get("objective", "flow") == "flow"
    assert s.get("time_sampling", "original") == "original"
    topology_matches = s.get("batch_size") == int(os.environ["BATCH_PER_GPU"]) and s.get(
        "world_size"
    ) == int(os.environ["WORLD_SIZE_EXPECTED"])
    discard_partial = os.environ.get("DISCARD_PARTIAL_EPOCH", "0") == "1"
    assert s.get("batch_in_epoch", 0) == 0 or topology_matches or discard_partial, (
        "Changing topology or batch requires an epoch-boundary checkpoint or "
        "DISCARD_PARTIAL_EPOCH=1"
    )
    assert s["step"] < int(os.environ["MAX_STEPS"]), "max_steps is a cumulative cap"
    assert "optimizer" in s, "Optimizer state required"
    output = Path(os.environ["OUTPUT_DIR"]).resolve()
    assert output != path.parent.resolve(), "Use a separate output directory"
    assert not output.exists() or not any(output.iterdir()), "Choose a new empty output directory"
    decoder = Path(os.environ["PERSIST_ROOT"]) / "data/checkpoints/voxel_decoder/model.safetensors"
    assert decoder.is_file(), decoder
    # Full cache hashing intentionally explicit: expensive, but detects mismatched copies.
    hashes = {p.name: digest(p) for p in sorted(cache_path.glob("*.npy"))}
    hashes["metadata.json"] = digest(cache_path / "metadata.json")
    code = {
        str(p.relative_to(root)): digest(p)
        for folder in ["polis", "experiments/m1"]
        for p in sorted((root / folder).rglob("*.py"))
    }
    return dict(
        host=socket.gethostname(),
        gpus=[torch.cuda.get_device_name(i) for i in range(int(os.environ["GPUS_PER_NODE"]))],
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        checkpoint=digest(path),
        decoder=digest(decoder),
        cache=hashes,
        code=code,
        step=s["step"],
        epoch=s["epoch"],
        batch=int(os.environ["BATCH_PER_GPU"]),
        max_steps=int(os.environ["MAX_STEPS"]),
        audit=int(os.environ["AUDIT_SAMPLES"]),
        fixed=int(os.environ["FIXED_LOSS_SAMPLES"]),
        eval_every_steps=int(os.environ.get("EVAL_EVERY_STEPS", "0")),
        checkpoint_every_steps=int(os.environ.get("CHECKPOINT_EVERY_STEPS", "0")),
        discard_partial_epoch=discard_partial,
    )


def collective(train=False):
    import torch
    import torch.distributed as dist

    sys.path.insert(0, os.environ["PLOT_ROOT"])
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    # Hash once per node, before process-group initialization; failures abort torchrun.
    report = local() if rank == 0 else None
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    assert dist.get_world_size() == int(os.environ["WORLD_SIZE_EXPECTED"])
    reports = [None] * dist.get_world_size()
    dist.all_gather_object(reports, report)
    nodes = [r for r in reports if r is not None]
    keys = [
        "torch",
        "cuda",
        "checkpoint",
        "decoder",
        "cache",
        "code",
        "step",
        "epoch",
        "batch",
        "max_steps",
        "audit",
        "fixed",
        "eval_every_steps",
        "checkpoint_every_steps",
        "discard_partial_epoch",
    ]
    assert all(all(r[k] == nodes[0][k] for k in keys) for r in nodes), (
        "Nodes have different data, code, configuration or checkpoint"
    )
    x = torch.full((1024 * 1024,), float(dist.get_rank() + 1), device="cuda")
    dist.all_reduce(x)
    n = dist.get_world_size()
    assert torch.all(x == n * (n + 1) / 2), "NCCL all-reduce mismatch"
    if dist.get_rank() == 0:
        print(json.dumps(dict(status="PASS", world=n, nodes=nodes), indent=2), flush=True)
    dist.barrier()
    if not train:
        dist.destroy_process_group()
    if train:
        root = Path(os.environ["PLOT_ROOT"])
        sys.argv = [
            "train_multiview_persist_full.py",
            "--phase",
            "train",
            "--objective",
            "flow",
            "--time-sampling",
            "original",
            "--cache",
            os.environ["CACHE_DIR"],
            "--persist",
            os.environ["PERSIST_ROOT"],
            "--resume",
            os.environ["RESUME_CHECKPOINT"],
            "--output",
            os.environ["OUTPUT_DIR"],
            "--max-steps",
            os.environ["MAX_STEPS"],
            "--batch-size",
            os.environ["BATCH_PER_GPU"],
            "--audit-samples",
            os.environ["AUDIT_SAMPLES"],
            "--fixed-loss-samples",
            os.environ["FIXED_LOSS_SAMPLES"],
            "--eval-every-steps",
            os.environ.get("EVAL_EVERY_STEPS", "0"),
            "--checkpoint-every-steps",
            os.environ.get("CHECKPOINT_EVERY_STEPS", "0"),
            "--workers",
            os.environ.get("WORKERS", "2"),
        ]
        if os.environ.get("DISCARD_PARTIAL_EPOCH", "0") == "1":
            sys.argv.append("--discard-partial-epoch")
        runpy.run_path(
            str(root / "experiments/m1/train_multiview_persist_full.py"), run_name="__main__"
        )


if __name__ == "__main__":
    if sys.argv[1] == "local":
        print(json.dumps(local(), indent=2))
    elif sys.argv[1] in ("collective", "train"):
        collective(train=sys.argv[1] == "train")
    else:
        raise SystemExit("Expected local, collective or train")
