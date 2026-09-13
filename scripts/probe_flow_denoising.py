"""Separate teacher-forced denoising from image-conditioned noise generation."""

# ruff: noqa: E402
import json
from pathlib import Path
import sys
import datetime
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.m1.multiview_voxel_dit import MultiViewVoxelDiT
from experiments.m1.train_multiview_persist_full import Cache
from experiments.m1.train_multiview_voxel_dit import codec, evaluate


def noise_for(indices, device):
    return torch.cat(
        [
            torch.randn(
                1,
                48,
                12,
                12,
                12,
                device=device,
                generator=torch.Generator(device=device).manual_seed(1234 + i),
            )
            for i in indices
        ]
    )


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    out = ROOT / "outputs/m1_flow_denoising_probe_v1"
    out.mkdir(exist_ok=False)
    path = ROOT / "outputs/m1_flow_full_1000k_v1/checkpoint_latest.pt"
    # mmap retains this checkpoint inode if the trainer atomically replaces the path.
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    step, epoch = state["step"], state["epoch"]
    torch.save(
        dict(model=state["model"], step=step, epoch=epoch, objective="flow"),
        out / "model_snapshot.pt",
    )
    model = MultiViewVoxelDiT().cuda().eval()
    model.load_state_dict(state["model"])
    del state
    decoder = codec(ROOT.parent / "PERSIST", decoder=True)
    source = Cache(ROOT / "outputs/m1_multiview_persist_s01_cache")
    indices = np.linspace(0, len(source) - 1, 64).round().astype(int).tolist()
    cache = source.evaluation_cache(indices)
    data = cache["data"]
    assert not (data["raw"] == 127).any()
    manifest = dict(
        step=step,
        epoch=epoch,
        indices=indices,
        source=str(path),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        noise="1234+global index; same across conditions and start times",
        steps=20,
        batch=4,
        conditions="original versus circular shift of RGB latent only; ray features/cameras/GT fixed",
        partial="GT latent plus noise is privileged diagnostic input, not image-only reconstruction",
        time_grid=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    clean = data["image_rays"].clone()
    results = {}
    one_step = {}
    for condition in ["original", "shuffled_images"]:
        cond = clean.clone()
        if condition == "shuffled_images":
            cond[:, :, :16] = cond[:, :, :16].roll(1, 0)
        for start_t in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            key = f"{condition}_t{start_t:.1f}"
            preds = []
            errors = []
            restored = []
            for offset in range(0, len(indices), 4):
                ix = indices[offset : offset + 4]
                target = data["latent"][offset : offset + 4].cuda()
                features = cond[offset : offset + 4].cuda()
                valid = (data["agent_mask"] & data["camera_valid"])[offset : offset + 4].cuda()
                noise = noise_for(ix, "cuda")
                a = 1 - 1e-5
                sigma = 1e-5 + a * start_t
                x = (1 - start_t) * target + sigma * noise
                time = torch.full((len(ix),), start_t, device="cuda")
                model.eval()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v = model(x, time, features, valid).float()
                errors.extend((v - (a * noise - target)).square().flatten(1).mean(1).cpu().tolist())
                z_est = a * x - sigma * v
                restored.extend((z_est - target).square().flatten(1).mean(1).cpu().tolist())
                schedule = np.linspace(1, 0, 21)
                schedule = start_t * 3 * schedule / (1 + 2 * schedule)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for t, tp in zip(schedule[:-1], schedule[1:]):
                        x = x - float(t - tp) * model(
                            x, torch.full((len(ix),), t, device="cuda"), features, valid
                        )
                preds.append(x.float().cpu())
            one_step[key] = dict(
                velocity_mse=float(np.mean(errors)),
                clean_estimate_mse=float(np.mean(restored)),
                velocity_per_sample=errors,
                clean_per_sample=restored,
            )
            data["sampled_latent"] = torch.cat(preds)
            folder = out / key
            folder.mkdir()
            result = evaluate(model, decoder, cache, step, folder)
            results[key] = result
            (out / "partial_results.json").write_text(
                json.dumps(dict(manifest=manifest, one_step=one_step, results=results), indent=2)
            )
            print(
                key,
                json.dumps(dict(one_step=one_step[key]["velocity_mse"], mean=result["mean"])),
                flush=True,
            )
    # Codec upper bound, plus uncorrected noisy latent: isolate decoder robustness.
    for t in [0.0, 0.1, 0.5, 0.9]:
        if t == 0:
            pred = data["latent"]
        else:
            pred = (1 - t) * data["latent"] + (1e-5 + (1 - 1e-5) * t) * noise_for(indices, "cpu")
        data["sampled_latent"] = pred
        folder = out / f"no_model_t{t:.1f}"
        folder.mkdir()
        results[f"no_model_t{t:.1f}"] = evaluate(model, decoder, cache, step, folder)
    (out / "results.json").write_text(
        json.dumps(dict(manifest=manifest, one_step=one_step, results=results), indent=2)
    )
    (out / "completion.json").write_text(json.dumps(dict(step=step, count=len(indices))))


if __name__ == "__main__":
    main()
