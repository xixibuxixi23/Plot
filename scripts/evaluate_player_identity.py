"""Evaluate cross-domain skin retrieval from rendered player crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from plot.data.player_identity_dataset import PlayerIdentityDataset
from plot.models.player_identity import PlayerIdentityEncoder


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PlayerIdentityEncoder(checkpoint.get("embedding_dim", 128)).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    dataset = PlayerIdentityDataset(
        args.dataset_root, args.window_index, canonical_only=False, min_pixels=96, retry=16
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    total = torch.zeros(5, device=device)
    for number, batch in enumerate(loader):
        if number >= args.batches:
            break
        valid = batch["valid"].to(device)
        identity = batch["identity"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            crop, reference = model(
                batch["crop"].to(device, non_blocking=True),
                batch["reference"].to(device, non_blocking=True),
            )
        scores = crop.float() @ reference.float().transpose(0, 1)
        keep = valid.bool()
        if not keep.any():
            continue
        positive = scores.diag()[keep]
        negatives = scores.masked_fill(
            identity[:, None] == identity[None, :], -torch.inf
        )[keep].amax(1)
        predicted_identity = identity[scores[keep].argmax(1)]
        total += torch.tensor(
            [
                float((predicted_identity == identity[keep]).float().sum()),
                float(keep.sum()),
                float(positive.sum()),
                float(negatives[torch.isfinite(negatives)].sum()),
                float(torch.isfinite(negatives).sum()),
            ],
            device=device,
        )
    report = {
        "checkpoint": args.checkpoint,
        "samples": int(total[1]),
        "retrieval_at_1": float(total[0] / total[1].clamp_min(1)),
        "positive_similarity": float(total[2] / total[1].clamp_min(1)),
        "hard_negative_similarity": float(total[3] / total[4].clamp_min(1)),
    }
    Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
