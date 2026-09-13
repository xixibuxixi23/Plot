"""Fit static multi-player PERSIST with supplied GT cameras and frozen VAEs."""

# ruff: noqa: E402
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from plot.data import TextAgentFillDataset
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from plot.models.projection import camera_rays, raycast_voxel_targets
from plot.models.visible_supervision import visible_voxel_masks
from plot.models.renderer_backbone.vae_voxel import ResNet3dEncoder, ResNet3dDecoder
from plot.models.renderer_backbone.vae_pixel import ViTVae


def load_weights(model, path):
    model.load_state_dict(
        {k.removeprefix("_orig_mod."): v for k, v in load_file(str(path)).items()}, strict=True
    )
    return model.eval().requires_grad_(False).cuda()


def codec(persist, decoder=False):
    model = (
        ResNet3dDecoder(
            out_channels=2138,
            latent_channels=48,
            channels=(512, 128, 32),
            num_res_blocks=2,
            num_res_blocks_middle=2,
        )
        if decoder
        else ResNet3dEncoder(
            input_x=48,
            input_y=48,
            input_z=48,
            in_channels=2138,
            latent_channels=48,
            channels=(32, 128, 512),
            num_res_blocks=2,
            num_res_blocks_middle=2,
        )
    )
    name = "voxel_decoder" if decoder else "voxel_encoder"
    return load_weights(model, persist / f"data/checkpoints/{name}/model.safetensors")


def atomic_save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


@torch.no_grad()
def prepare(args):
    ds = TextAgentFillDataset(
        args.dataset_root,
        ROOT / "datasets/s01_block_vocabulary.json",
        split="val_id",
        samples_per_agent=1,
        image_size=(360, 640),
        max_agents=args.num_views,
        num_views=args.num_views,
        initial_only=True,
        episode_index=ROOT / "datasets/s01_episode_index.json",
        canonical_yaw=False,
    )
    classes = json.loads(
        (args.persist / "data/persist-eval-sample/mt_voxel_classdict.json").read_text()
    )["node_classes"]
    pairs = {tuple(v): int(k) for k, v in classes.items()}
    raw_lut = torch.tensor([classes[str(i)][0] for i in range(len(classes))])
    stats = json.loads((args.persist / "pipelines/persist_S_pipeline/pipeline.json").read_text())[
        "args"
    ]["latent_normalization_stats"]
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
    entries, records = [], []
    # One target resident per episode, so different crops of one episode do not duplicate fitted scenes.
    eligible = [i for i, (_, agent, _) in enumerate(ds.index) if agent == 0]
    selected = [
        eligible[i] for i in np.linspace(0, len(eligible) - 1, args.samples).round().astype(int)
    ]

    def norm(z, kind):
        shape = (1, -1, *([1] * (z.ndim - 2)))
        return (z - torch.tensor(stats[kind]["mean"], device="cuda").reshape(shape)) / torch.tensor(
            stats[kind]["std"], device="cuda"
        ).reshape(shape)

    for idx in selected:
        batch = ds[idx]
        episode, resident, _ = ds.index[idx]
        path, _ = ds.episodes[episode]
        with np.load(path / "m1_initial.npz") as data:
            tile = data["voxel_tiles"][resident]
        ids = np.vectorize(lambda n, p: pairs.get((int(n), int(p)), -1))(tile[..., 0], tile[..., 1])
        if (ids < 0).any():
            raise ValueError(f"Unsupported original node+param2 pair in {path}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = norm(encoder(torch.tensor(ids[None], device="cuda")), "voxel")
            pix = pixel.encode(batch["images"].cuda() * 2 - 1).mode()
            pix = norm(pix.reshape(args.num_views, 36, 64, 16).permute(0, 3, 1, 2), "pixel")
        rays = camera_rays(
            batch["camera_direction"].cuda(), batch["fov_x"].cuda(), batch["fov_y"].cuda(), 36, 64
        )
        origins = batch["camera_position"].cuda()[:, None, None].expand(-1, 36, 64, -1)
        cond = torch.cat([pix.float(), torch.cat([origins, rays], -1).permute(0, 3, 1, 2)], 1)
        raw = torch.tensor(tile[..., 0].astype(np.int64))
        entry = {
            k: batch[k]
            for k in (
                "images",
                "camera_position",
                "camera_direction",
                "fov_x",
                "fov_y",
                "agent_mask",
                "camera_valid",
            )
        }
        entry.update(latent=z[0].float().cpu(), image_rays=cond.cpu(), raw=raw)
        entries.append(entry)
        records.append(dict(dataset_index=idx, episode=str(path), target_resident=resident))
        print(f"prepared {len(entries)}/{args.samples}", flush=True)
    cache = dict(
        data={k: torch.stack([e[k] for e in entries]) for k in entries[0]},
        records=records,
        stats=stats,
        raw_lut=raw_lut,
    )
    atomic_save(cache, args.output / "cache.pt")
    del encoder, pixel
    torch.cuda.empty_cache()
    return cache


@torch.no_grad()
def evaluate(model, decoder, cache, step, output, steps=20, seed_base=1234):
    model.eval()
    data = cache["data"]
    vm = torch.tensor(cache["stats"]["voxel"]["mean"], device="cuda").reshape(1, 48, 1, 1, 1)
    vs = torch.tensor(cache["stats"]["voxel"]["std"], device="cuda").reshape(1, 48, 1, 1, 1)
    raw_lut = cache["raw_lut"].cuda()
    rows, preds = [], []
    schedule = np.linspace(1, 0, steps + 1)
    schedule = 3 * schedule / (1 + 2 * schedule)
    for i in range(len(data["raw"])):
        # Local generator ensures repeated evals use identical noise without altering training RNG.
        g = torch.Generator(device="cuda").manual_seed(seed_base + i)
        x = torch.randn(1, 48, 12, 12, 12, device="cuda", generator=g)
        cond = data["image_rays"][i : i + 1].cuda()
        valid = (data["agent_mask"] & data["camera_valid"])[i : i + 1].cuda()
        condition_valid = data.get(
            "condition_view_valid", data["agent_mask"] & data["camera_valid"]
        )[i : i + 1].cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if "sampled_latent" in data:
                x = data["sampled_latent"][i : i + 1].cuda()
            else:
                for t, tp in zip(schedule[:-1], schedule[1:]):
                    x = x - float(t - tp) * model(
                        x, torch.full((1,), t, device="cuda"), cond, condition_valid
                    )
            pred = raw_lut[decoder(x * vs + vm).argmax(-1)]
        batch = {
            k: v[i : i + 1].cuda()
            for k, v in data.items()
            if k in ("camera_position", "camera_direction", "fov_x", "fov_y", "raw")
        }
        gt = batch["raw"]
        surface, free = visible_voxel_masks(
            gt,
            torch.ones_like(gt, dtype=torch.bool),
            batch["camera_position"],
            batch["camera_direction"],
            valid,
            batch["fov_x"],
            batch["fov_y"],
            126,
        )
        correct = (pred == gt) & surface
        visible_pred = (pred != 126) & (surface | free)
        row = dict(
            surface_exact_recall=float(correct.sum() / surface.sum().clamp_min(1)),
            visible_exact_precision=float(correct.sum() / visible_pred.sum().clamp_min(1)),
            free_accuracy=float(((pred == 126) & free).sum() / free.sum().clamp_min(1)),
            latent_mse=float((x - data["latent"][i : i + 1].cuda()).square().mean()),
        )
        hits = count = 0
        for view in range(valid.shape[1]):
            if not valid[0, view]:
                continue
            opts = dict(height=60, width=106, samples=192, max_distance=32.0)
            ph, pd, _ = raycast_voxel_targets(
                (pred != 126).float(),
                batch["camera_position"][:, view],
                batch["camera_direction"][:, view],
                batch["fov_x"][:, view],
                batch["fov_y"][:, view],
                **opts,
            )
            gh, gd, _ = raycast_voxel_targets(
                (gt != 126).float(),
                batch["camera_position"][:, view],
                batch["camera_direction"][:, view],
                batch["fov_x"][:, view],
                batch["fov_y"][:, view],
                **opts,
            )
            active = gh.bool()
            active[:, round(60 * 0.82) :] = False
            hits += int((active & ph.bool() & ((pd - gd).abs() <= 0.5)).sum())
            count += int(active.sum())
        row["gt_camera_half_block_hit"] = hits / max(count, 1)
        rows.append(row)
        preds.append(pred[0].cpu())
    result = dict(
        step=step, samples=rows, mean={k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    )
    (output / f"metrics_{step:06d}.json").write_text(json.dumps(result, indent=2))
    atomic_save(dict(pred=torch.stack(preds), metrics=result), output / f"prediction_{step:06d}.pt")
    print("EVAL " + json.dumps(result["mean"]), flush=True)
    model.train()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT.parent / "textagent/data/batches/s01_v1_mask_20260907",
    )
    p.add_argument("--persist", type=Path, default=ROOT.parent / "PERSIST")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--num-views", type=int, default=2)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(20260910)
    cache = (
        torch.load(args.output / "cache.pt", weights_only=False)
        if (args.output / "cache.pt").exists()
        else prepare(args)
    )
    if len(cache["records"]) != args.samples:
        raise ValueError("Cache sample count mismatch")
    if cache["data"]["images"].shape[1] != args.num_views:
        raise ValueError("Cache view count mismatch")
    model = MultiViewVoxelDiT()
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("PERSIST-team/persist-voxel-denoiser-s", "model.safetensors")
    transfer = model.load_persist(load_file(path))
    model.cuda().train()
    decoder = codec(args.persist, decoder=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["step"]
        if "rng_cpu" in state:
            torch.set_rng_state(state["rng_cpu"])
            torch.cuda.set_rng_state(state["rng_cuda"])
    manifest = dict(
        protocol="memorization; 2 RGB views + GT cameras; frozen PERSIST codecs; no scene IDs; no mask supervision; flow velocity MSE only",
        config=model.config,
        transfer=transfer,
        source_checkpoint=path,
        records=cache["records"],
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with (args.output / "run_history.jsonl").open("a") as f:
        f.write(json.dumps(dict(start_step=start, **manifest["args"])) + "\n")
    (args.output / "run_status.json").write_text(
        json.dumps(dict(status="running", start_step=start, target_step=args.steps))
    )
    print("MODEL " + json.dumps(transfer), flush=True)
    evaluate(model, decoder, cache, start, args.output)
    data = cache["data"]
    tic = time.time()
    for step in range(start + 1, args.steps + 1):
        idx = torch.randperm(args.samples)[: args.batch_size]
        target = data["latent"][idx].cuda()
        cond = data["image_rays"][idx].cuda()
        valid = (data["agent_mask"][idx] & data["camera_valid"][idx]).cuda()
        noise = torch.randn_like(target)
        t = torch.randn(len(idx), device="cuda").sigmoid()
        tt = t[:, None, None, None, None]
        x = (1 - tt) * target + (1e-5 + (1 - 1e-5) * tt) * noise
        velocity = (1 - 1e-5) * noise - target
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = (model(x, t, cond, valid).float() - velocity).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 20 == 0:
            record = dict(step=step, loss=float(loss), elapsed=time.time() - tic)
            with (args.output / "train.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            result = evaluate(model, decoder, cache, step, args.output)
            atomic_save(
                dict(
                    model=model.state_dict(),
                    optimizer=optimizer.state_dict(),
                    step=step,
                    config=model.config,
                    rng_cpu=torch.get_rng_state(),
                    rng_cuda=torch.cuda.get_rng_state(),
                ),
                args.output / "checkpoint_latest.pt",
            )
    (args.output / "completion.json").write_text(
        json.dumps(dict(step=args.steps, metrics=result["mean"]), indent=2)
    )
    (args.output / "run_status.json").write_text(
        json.dumps(dict(status="complete", step=args.steps))
    )


if __name__ == "__main__":
    main()
