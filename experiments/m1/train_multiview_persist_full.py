"""Full S01 memorization with GT cameras: distributed codec cache, DiT fit, full audit."""

# ruff: noqa: E402
import argparse
import json
import os
import time
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plot.data import TextAgentFillDataset
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from plot.models.projection import camera_rays
from plot.models.renderer_backbone.vae_pixel import ViTVae
from experiments.m1.train_multiview_voxel_dit import codec, load_weights, atomic_save, evaluate


FIELDS = dict(
    latent=((48, 12, 12, 12), "float16"),
    image_rays=((2, 22, 36, 64), "float16"),
    raw=((48, 48, 48), "int16"),
    camera_position=((2, 3), "float32"),
    camera_direction=((2, 3), "float32"),
    fov_x=((2,), "float32"),
    fov_y=((2,), "float32"),
    agent_mask=((2,), "bool"),
    camera_valid=((2,), "bool"),
)


class Source(Dataset):
    def __init__(self, args):
        self.datasets = [
            TextAgentFillDataset(
                args.dataset_root,
                ROOT / "datasets/s01_block_vocabulary.json",
                split=s,
                samples_per_agent=1,
                image_size=(360, 640),
                max_agents=2,
                num_views=2,
                initial_only=True,
                episode_index=ROOT / "datasets/s01_episode_index.json",
                canonical_yaw=False,
            )
            for s in ("train", "val_id", "test_id")
        ]
        self.items = [(s, i) for s, ds in enumerate(self.datasets) for i in range(len(ds))]
        c = json.loads(
            (args.persist / "data/persist-eval-sample/mt_voxel_classdict.json").read_text()
        )["node_classes"]
        self.lut = np.full((max(v[0] for v in c.values()) + 1, 256), -1, dtype=np.int64)
        for k, (node, param) in c.items():
            self.lut[node, param] = int(k)
        self.fallback = np.full(len(self.lut), -1, dtype=np.int64)
        for k, (node, param) in c.items():
            if self.fallback[node] < 0 or param == 0:
                self.fallback[node] = int(k)
        self.raw_lut = [c[str(i)][0] for i in range(len(c))]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        s, i = self.items[index]
        ds = self.datasets[s]
        b = ds[i]
        ep, agent, _ = ds.index[i]
        path, _ = ds.episodes[ep]
        with np.load(path / "m1_initial.npz") as f:
            tile = f["voxel_tiles"][agent].astype(np.int64)
        if tile[..., 0].max() >= len(self.lut) or tile[..., 1].max() >= 256 or tile.min() < 0:
            raise ValueError(f"Node/state outside PERSIST lookup: {path}")
        ids = self.lut[tile[..., 0], tile[..., 1]]
        missing = ids < 0
        pairs, counts = np.unique(tile[missing], axis=0, return_counts=True)
        ids[missing] = self.fallback[tile[..., 0][missing]]
        if (ids < 0).any():
            raise ValueError(
                f"Unknown raw block ID cannot be mapped without changing material: {path}"
            )
        return dict(
            index=index,
            aliases=json.dumps(
                [
                    dict(node=int(v[0]), param2=int(v[1]), count=int(n))
                    for v, n in zip(pairs, counts)
                ]
            ),
            ids=torch.from_numpy(ids),
            raw=torch.from_numpy(tile[..., 0]),
            **{
                k: b[k]
                for k in (
                    "images",
                    "camera_position",
                    "camera_direction",
                    "fov_x",
                    "fov_y",
                    "agent_mask",
                    "camera_valid",
                )
            },
        )


class Cache(Dataset):
    def __init__(self, path, mode="r"):
        self.meta = json.loads((path / "metadata.json").read_text())
        self.arrays = {k: np.load(path / (k + ".npy"), mmap_mode=mode) for k in FIELDS}

    def __len__(self):
        return len(self.arrays["latent"])

    def __getitem__(self, i):
        return {
            k: torch.from_numpy(np.array(self.arrays[k][i], copy=True))
            for k in ("latent", "image_rays", "agent_mask", "camera_valid")
        }

    def evaluation_cache(self, indices):
        return dict(
            data={
                k: torch.from_numpy(np.array(v[indices], copy=True)).float()
                if v.dtype == np.float16
                else torch.from_numpy(np.array(v[indices], copy=True))
                for k, v in self.arrays.items()
            },
            stats=self.meta["stats"],
            raw_lut=torch.tensor(self.meta["raw_lut"]),
        )


@torch.no_grad()
def prepare(args, rank, world):
    source = Source(args)
    n = len(source)
    meta = dict(
        count=n,
        dataset_root=str(args.dataset_root.resolve()),
        splits={s: len(d) for s, d in zip(("train", "val_id", "test_id"), source.datasets)},
        raw_lut=source.raw_lut,
        stats=json.loads((args.persist / "pipelines/persist_S_pipeline/pipeline.json").read_text())[
            "args"
        ]["latent_normalization_stats"],
    )
    if rank == 0:
        args.cache.mkdir(parents=True, exist_ok=True)
        if (args.cache / "metadata.json").exists():
            if json.loads((args.cache / "metadata.json").read_text()) != meta:
                raise ValueError("Cache provenance mismatch")
        else:
            for k, (shape, dtype) in FIELDS.items():
                a = np.lib.format.open_memmap(
                    args.cache / (k + ".npy"), mode="w+", dtype=dtype, shape=(n, *shape)
                )
                a.flush()
                del a
            a = np.lib.format.open_memmap(
                args.cache / "ready.npy", mode="w+", dtype="bool", shape=(n,)
            )
            a[:] = False
            a.flush()
            (args.cache / "metadata.json").write_text(json.dumps(meta, indent=2))
    dist.barrier()
    arrays = {k: np.load(args.cache / (k + ".npy"), mmap_mode="r+") for k in FIELDS}
    ready = np.load(args.cache / "ready.npy", mmap_mode="r+")
    indices = [i for i in range(rank, n, world) if not ready[i]]
    if indices:
        encoder = codec(args.persist)
        pixel = load_weights(
            ViTVae(
                latent_dim=16,
                input_height=360,
                input_width=640,
                patch_size=10,
                enc_dim=1024,
                enc_depth=6,
                enc_heads=16,
                dec_dim=1024,
                dec_depth=12,
                dec_heads=16,
                mlp_ratio=4.0,
                use_variational=True,
                qk_rms_norm=True,
            ),
            args.persist / "data/checkpoints/pixel_vae/model.safetensors",
        )

        def norm(x, kind):
            sh = (1, -1, *([1] * (x.ndim - 2)))
            return (
                x - torch.tensor(meta["stats"][kind]["mean"], device="cuda").reshape(sh)
            ) / torch.tensor(meta["stats"][kind]["std"], device="cuda").reshape(sh)

        loader = DataLoader(
            source,
            batch_size=args.prepare_batch,
            sampler=indices,
            num_workers=args.workers,
            pin_memory=True,
        )
        done = 0
        tic = time.time()
        for b in loader:
            idx = b.pop("index").numpy()
            batch = len(idx)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = norm(encoder(b["ids"].cuda()), "voxel")
                pix = pixel.encode(b["images"].flatten(0, 1).cuda() * 2 - 1).mode()
                pix = norm(pix.reshape(batch * 2, 36, 64, 16).permute(0, 3, 1, 2), "pixel")
            rays = camera_rays(
                b["camera_direction"].flatten(0, 1).cuda(),
                b["fov_x"].flatten().cuda(),
                b["fov_y"].flatten().cuda(),
                36,
                64,
            )
            origins = (
                b["camera_position"].flatten(0, 1).cuda()[:, None, None].expand(-1, 36, 64, -1)
            )
            cond = torch.cat(
                [pix.float(), torch.cat([origins, rays], -1).permute(0, 3, 1, 2)], 1
            ).reshape(batch, 2, 22, 36, 64)
            arrays["latent"][idx] = latent.float().cpu().numpy()
            arrays["image_rays"][idx] = cond.cpu().numpy()
            for k in FIELDS:
                if k not in ("latent", "image_rays"):
                    arrays[k][idx] = b[k].numpy()
            with (args.cache / f"aliases_rank_{rank}.jsonl").open("a") as f:
                for i, aliases in zip(idx, b["aliases"]):
                    if aliases != "[]":
                        f.write(json.dumps(dict(index=int(i), aliases=json.loads(aliases))) + "\n")
            # Data is flushed before marking records reusable after interruption.
            for a in arrays.values():
                a.flush()
            ready[idx] = True
            ready.flush()
            done += batch
            if done % 100 < batch:
                print(
                    json.dumps(
                        dict(
                            phase="prepare",
                            rank=rank,
                            done=done,
                            total=len(indices),
                            seconds=time.time() - tic,
                        )
                    ),
                    flush=True,
                )
        del encoder, pixel
        torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        if not ready.all():
            raise RuntimeError("Incomplete cache")
        (args.cache / "completion.json").write_text(json.dumps(dict(count=n, complete=True)))


def restore_optimizer(optimizer, state, resume_lr=None):
    optimizer.load_state_dict(state)
    if resume_lr is not None:
        for group in optimizer.param_groups:
            group["lr"] = resume_lr


@torch.no_grad()
def fixed_loss_audit(model, cache, count, rank, world):
    """Fixed scene/noise/time grid; independent of the training RNG and split."""
    model.eval()
    times = [0.1, 0.3, 0.5, 0.7, 0.9]
    totals = torch.zeros(len(times) + 1, device="cuda", dtype=torch.float64)
    indices = np.linspace(0, len(cache) - 1, count).round().astype(int)[rank::world]
    for index in indices:
        b = cache[int(index)]
        target = b["latent"][None].cuda().float()
        cond = b["image_rays"][None].cuda().float()
        valid = (b["agent_mask"] & b["camera_valid"])[None].cuda()
        generator = torch.Generator(device="cuda").manual_seed(90210 + int(index))
        noise = torch.randn(target.shape, generator=generator, device="cuda")
        velocity = (1 - 1e-5) * noise - target
        for j, time_value in enumerate(times):
            x = (1 - time_value) * target + (1e-5 + (1 - 1e-5) * time_value) * noise
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x, torch.tensor([time_value], device="cuda"), cond, valid)
            totals[j] += (pred.float() - velocity).square().mean().double()
        totals[-1] += 1
    dist.all_reduce(totals)
    values = (totals[:-1] / totals[-1]).tolist()
    return dict(
        count=int(totals[-1]),
        mean=sum(values) / len(values),
        by_time=dict(zip(map(str, times), values)),
        protocol="equally spaced fitted windows; noise seed 90210+index; uniform fixed time grid, not training-distribution mean",
    )


def predict_clean(model, cond, valid):
    return model(
        torch.zeros(len(cond), 48, 12, 12, 12, device=cond.device),
        torch.zeros(len(cond), device=cond.device),
        cond,
        valid,
    )


@torch.no_grad()
def direct_cache(model, cache, indices):
    model.eval()
    c = cache.evaluation_cache(indices)
    d = c["data"]
    outputs = []
    for start in range(0, len(indices), 8):
        cond = d["image_rays"][start : start + 8].cuda()
        valid = (d["agent_mask"] & d["camera_valid"])[start : start + 8].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs.append(predict_clean(model, cond, valid).float().cpu())
    d["sampled_latent"] = torch.cat(outputs)
    return c


def mix_high_noise_time(base_time, generator):
    replace = torch.rand(base_time.shape, device=base_time.device, generator=generator) < 0.5
    high = 0.8 + 0.2 * torch.rand(base_time.shape, device=base_time.device, generator=generator)
    return torch.where(replace, high, base_time)


def step_limited_end_epoch(start, batch_offset, step, max_steps, batches_per_epoch):
    if max_steps <= step:
        raise ValueError("max_steps must exceed the checkpoint step")
    return start + (batch_offset + max_steps - step + batches_per_epoch - 1) // batches_per_epoch


def train(args, rank, world):
    cache = Cache(args.cache)
    if not np.load(args.cache / "ready.npy", mmap_mode="r").all():
        raise ValueError("Cache is incomplete")
    model = MultiViewVoxelDiT().cuda()
    state = torch.load(args.resume or args.initialize, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start = 0
    step = 0
    batch_offset = 0
    if args.resume:
        if state.get("objective", "flow") != args.objective:
            raise ValueError(
                "Resume objective differs from checkpoint; use --initialize for objective adaptation"
            )
        restore_optimizer(optimizer, state["optimizer"], args.resume_lr)
        start = state["epoch"]
        step = state["step"]
        batch_offset = state.get("batch_in_epoch", 0)
        if args.discard_partial_epoch and batch_offset:
            start += 1
            batch_offset = 0
    del state
    ddp = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])
    sampler = DistributedSampler(cache, shuffle=True, seed=20260910, drop_last=False)
    loader = DataLoader(
        cache,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    if args.max_steps is not None:
        args.epochs = step_limited_end_epoch(start, batch_offset, step, args.max_steps, len(loader))
    decoder = codec(args.persist, decoder=True)
    audit = np.linspace(0, len(cache) - 1, args.audit_samples).round().astype(int)[rank::world]
    if rank == 0:
        (args.output / "manifest.json").write_text(
            json.dumps(
                dict(
                    protocol="all S01 memorization, supplied GT cameras, two players, frozen codecs",
                    objective=args.objective,
                    time_sampling=args.time_sampling,
                    count=len(cache),
                    epochs=args.epochs,
                    max_steps=args.max_steps,
                    eval_every_steps=args.eval_every_steps,
                    checkpoint_every_steps=args.checkpoint_every_steps,
                    start_step=step,
                    effective_batch=args.batch_size * world,
                    initialize=str(args.initialize),
                    lr=[group["lr"] for group in optimizer.param_groups],
                    resume=str(args.resume),
                    start_epoch=start,
                    seed=20260910,
                    config=model.config,
                ),
                indent=2,
            )
        )
    tic = time.time()
    high_time_generator = torch.Generator(device="cuda").manual_seed(88011 + rank)

    def write_fixed(epoch):
        if args.fixed_loss_samples:
            if args.objective == "direct":
                ix = (
                    np.linspace(0, len(cache) - 1, args.fixed_loss_samples)
                    .round()
                    .astype(int)[rank::world]
                )
                d = direct_cache(model, cache, ix)["data"]
                errors = (d["sampled_latent"] - d["latent"]).square().flatten(1).mean(1)
                totals = torch.tensor(
                    [float(errors.sum()), len(errors)], device="cuda", dtype=torch.float64
                )
                dist.all_reduce(totals)
                result = dict(
                    count=int(totals[1]),
                    mean=float(totals[0] / totals[1]),
                    protocol="fixed evenly spaced fitted windows; deterministic clean latent MSE",
                )
            else:
                result = fixed_loss_audit(model, cache, args.fixed_loss_samples, rank, world)
            if rank == 0:
                row = dict(epoch=epoch, step=step, **result)
                (args.output / f"fixed_loss_epoch_{epoch:03d}.json").write_text(
                    json.dumps(row, indent=2)
                )
                print("FIXED_LOSS " + json.dumps(row), flush=True)

    def save_checkpoint(resume_epoch, resume_batch):
        if rank == 0:
            atomic_save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    epoch=resume_epoch,
                    batch_in_epoch=resume_batch,
                    batch_size=args.batch_size,
                    world_size=world,
                    step=step,
                    config=model.config,
                    objective=args.objective,
                    time_sampling=args.time_sampling,
                ),
                args.output / "checkpoint_latest.pt",
            )
        dist.barrier()

    def run_audit(audit_epoch):
        write_fixed(audit_epoch)
        audit_cache = (
            direct_cache(model, cache, audit)
            if args.objective == "direct"
            else cache.evaluation_cache(audit)
        )
        result = evaluate(model, decoder, audit_cache, step, args.output / f"rank_{rank}")
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            rows = [r for part in gathered for r in part["samples"]]
            summary = dict(
                epoch=audit_epoch,
                step=step,
                mean={k: sum(r[k] for r in rows) / len(rows) for k in rows[0]},
            )
            (args.output / f"audit_epoch_{audit_epoch:03d}.json").write_text(
                json.dumps(summary, indent=2)
            )
            print("AUDIT " + json.dumps(summary), flush=True)
        dist.barrier()
        ddp.train()

    write_fixed(start)
    for epoch in range(start, args.epochs):
        sampler.set_epoch(epoch)
        ddp.train()
        epoch_totals = torch.zeros(2, device="cuda", dtype=torch.float64)
        for batch_index, b in enumerate(loader):
            if epoch == start and batch_index < batch_offset:
                continue
            target = b["latent"].cuda().float()
            cond = b["image_rays"].cuda().float()
            valid = (b["agent_mask"] & b["camera_valid"]).cuda()
            noise = torch.randn_like(target)
            t = torch.randn(len(target), device="cuda").sigmoid()
            if args.time_sampling == "high_noise":
                t = mix_high_noise_time(t, high_time_generator)
            tt = t[:, None, None, None, None]
            x = (1 - tt) * target + (1e-5 + (1 - 1e-5) * tt) * noise
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.objective == "direct":
                    loss = (predict_clean(ddp, cond, valid).float() - target).square().mean()
                else:
                    loss = (
                        (ddp(x, t, cond, valid).float() - ((1 - 1e-5) * noise - target))
                        .square()
                        .mean()
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_totals[0] += loss.detach().double() * len(target)
            epoch_totals[1] += len(target)
            step += 1
            if rank == 0 and step % 20 == 0:
                row = dict(epoch=epoch + 1, step=step, loss=float(loss), seconds=time.time() - tic)
                with (args.output / "train.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            final_step = (args.max_steps is not None and step >= args.max_steps) or (
                epoch == args.epochs - 1 and batch_index + 1 == len(loader)
            )
            resume_epoch = epoch + 1 if batch_index + 1 == len(loader) else epoch
            resume_batch = 0 if resume_epoch == epoch + 1 else batch_index + 1
            if args.checkpoint_every_steps and (step % args.checkpoint_every_steps == 0 or final_step):
                save_checkpoint(resume_epoch, resume_batch)
            if args.eval_every_steps and (step % args.eval_every_steps == 0 or final_step):
                run_audit(epoch + 1)
            if final_step:
                break
        next_epoch = epoch + 1 if batch_index + 1 == len(loader) else epoch
        next_batch = 0 if next_epoch == epoch + 1 else batch_index + 1
        dist.all_reduce(epoch_totals)
        if rank == 0:
            row = dict(
                epoch=epoch + 1,
                step=step,
                loss=float(epoch_totals[0] / epoch_totals[1]),
                samples=int(epoch_totals[1]),
                lr=optimizer.param_groups[0]["lr"],
            )
            with (args.output / "epoch_loss.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        if args.checkpoint_every_steps == 0:
            save_checkpoint(next_epoch, next_batch)
        if args.eval_every_steps == 0:
            run_audit(epoch + 1)
        if args.max_steps is not None and step >= args.max_steps:
            break
    if rank == 0:
        (args.output / "training_complete.json").write_text(
            json.dumps(dict(epoch=next_epoch, batch_in_epoch=next_batch, step=step))
        )


def full_eval(args, rank, world):
    from scripts.evaluate_multiview_persist_full import sample

    cache = Cache(args.cache)
    model = MultiViewVoxelDiT().cuda()
    state = torch.load(args.output / "checkpoint_latest.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    step = state["step"]
    objective = state.get("objective", "flow")
    del state
    decoder = codec(args.persist, decoder=True)
    rows = []
    for start in range(rank * args.eval_chunk, len(cache), world * args.eval_chunk):
        indices = list(range(start, min(start + args.eval_chunk, len(cache))))
        chunk = cache.evaluation_cache(indices)
        data = chunk["data"]
        sampled = []
        for offset in range(0, len(indices), args.batch_size):
            selected = indices[offset : offset + args.batch_size]
            cond = data["image_rays"][offset : offset + len(selected)].cuda()
            valid = (data["agent_mask"] & data["camera_valid"])[
                offset : offset + len(selected)
            ].cuda()
            if objective == "direct":
                model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    sampled.append(predict_clean(model, cond, valid).float().cpu())
            else:
                sampled.append(sample(model, cond, valid, selected).cpu())
        data["sampled_latent"] = torch.cat(sampled)
        result = evaluate(
            model,
            decoder,
            chunk,
            start,
            args.output / f"full_rank_{rank}",
            seed_base=1234 + start,
        )
        rows.extend([dict(index=i, **r) for i, r in zip(indices, result["samples"])])
    gathered = [None] * world
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        rows = sorted([r for part in gathered for r in part], key=lambda r: r["index"])
        if [r["index"] for r in rows] != list(range(len(cache))):
            raise RuntimeError("Full eval coverage mismatch")
        result = dict(
            step=step,
            count=len(rows),
            sampling_batch=args.batch_size,
            objective=objective,
            seed_protocol="none: deterministic zero queries"
            if objective == "direct"
            else "1234 + global sample index",
            mean={k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != "index"},
            samples=rows,
        )
        (args.output / "full_fit_metrics.json").write_text(json.dumps(result, indent=2))
        (args.output / "completion.json").write_text(
            json.dumps(dict(step=step, count=len(rows), mean=result["mean"]), indent=2)
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=["prepare", "train", "evaluate", "all"], default="all")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT.parent / "textagent/data/batches/s01_v1_mask_20260907",
    )
    p.add_argument("--persist", type=Path, default=ROOT.parent / "PERSIST")
    p.add_argument("--cache", type=Path, default=ROOT / "outputs/m1_multiview_persist_s01_cache")
    p.add_argument(
        "--output", type=Path, default=ROOT / "outputs/m1_multiview_persist_gtcam_s01_all_v1"
    )
    p.add_argument(
        "--initialize",
        type=Path,
        default=ROOT / "outputs/m1_multiview_persist_gtcam_fit8_v1/checkpoint_latest.pt",
    )
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--discard-partial-epoch",
        action="store_true",
        help="Resume weights and optimizer at the next epoch when batch topology changes",
    )
    p.add_argument("--objective", choices=["flow", "direct"], default="flow")
    p.add_argument("--time-sampling", choices=["original", "high_noise"], default="original")
    p.add_argument(
        "--resume-lr",
        type=float,
        help="Override checkpoint learning rate while retaining Adam moments",
    )
    p.add_argument("--fixed-loss-samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument(
        "--max-steps",
        type=int,
        help="Total update cap including checkpoint steps; overrides epochs",
    )
    p.add_argument("--eval-every-steps", type=int, default=0, help="0: every epoch; otherwise global step interval")
    p.add_argument("--checkpoint-every-steps", type=int, default=0, help="0: every epoch; otherwise global step interval")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prepare-batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--audit-samples", type=int, default=32)
    p.add_argument("--eval-chunk", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    args = p.parse_args()
    if args.eval_every_steps < 0 or args.checkpoint_every_steps < 0:
        p.error("Step intervals must be nonnegative")
    torch.set_num_threads(4)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    # The multinode preflight may already own the validated process group.
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.manual_seed(20260910 + rank)
    args.output.mkdir(parents=True, exist_ok=True)
    for name in (f"rank_{rank}", f"full_rank_{rank}"):
        (args.output / name).mkdir(exist_ok=True)
    if args.phase in ("prepare", "all"):
        prepare(args, rank, world)
    if args.phase in ("train", "all"):
        train(args, rank, world)
    if args.phase in ("evaluate", "all"):
        full_eval(args, rank, world)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
