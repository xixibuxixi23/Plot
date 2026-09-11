#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data import SyntheticFillDataset
from plot.models import FillNetwork
from plot.training import FillTrainer, FillTrainerConfig


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    batch = next(iter(DataLoader(SyntheticFillDataset(length=1, size=8, agents=2, classes=6))))
    model = FillNetwork(
        6, voxel_embedding_dim=4, image_feature_dim=16, base_channels=4,
        attention_heads=2, max_views=2,
    ).to(local_rank)
    model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    trainer = FillTrainer(model, FillTrainerConfig(device=f"cuda:{local_rank}", precision="bf16"))
    loss = trainer.train_step(batch)
    value = torch.tensor(loss, device=local_rank)
    dist.all_reduce(value)
    if rank == 0:
        print(f"ddp_world_size={dist.get_world_size()} mean_loss={value.item()/dist.get_world_size():.6f}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
