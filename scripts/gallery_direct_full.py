"""Use the same eight windows as the earlier full-flow gallery."""

import json
from pathlib import Path
import shutil
import torch

ROOT = Path(__file__).resolve().parents[1]
old = ROOT / "outputs/m1_multiview_persist_gtcam_s01_all_v1/gallery"
out = ROOT / "outputs/m1_direct_full_s01_v1"
assert (out / "completion.json").exists()
indices = json.loads((old / "indices.json").read_text())
pred = []
for i in indices:
    start = i // 32 * 32
    rank = (start // 32) % 8
    chunk = torch.load(out / f"full_rank_{rank}/prediction_{start:06d}.pt", weights_only=False)[
        "pred"
    ]
    pred.append(chunk[i - start])
gallery = out / "gallery"
gallery.mkdir(exist_ok=True)
shutil.copy2(old / "cache.pt", gallery / "cache.pt")
shutil.copy2(old / "indices.json", gallery / "indices.json")
torch.save(dict(pred=torch.stack(pred)), gallery / "prediction_014440.pt")
print(indices)
