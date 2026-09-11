from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data import SyntheticFillDataset
from plot.models import FillNetwork
from plot.pipelines import PlotPipeline
from plot.training import FillTrainer, FillTrainerConfig


def main() -> None:
    torch.manual_seed(0)
    dataset = SyntheticFillDataset(length=4, size=16, agents=2, classes=8)
    batch = next(iter(DataLoader(dataset, batch_size=1)))
    model = FillNetwork(8, voxel_embedding_dim=4, image_feature_dim=16, base_channels=4)
    trainer = FillTrainer(model, FillTrainerConfig(device="cpu"))
    loss = trainer.train_step(batch)
    metrics = trainer.evaluate_step(batch)

    # The production pipeline uses 48-cubed tiles. Use tiny images here while
    # exercising real overlapping global coordinates and sparse-memory commit.
    pipeline_model = FillNetwork(8, voxel_embedding_dim=4, image_feature_dim=16, base_channels=4)
    pipeline = PlotPipeline(pipeline_model, device="cpu")
    images = torch.rand(2, 3, 32, 48)
    result = pipeline.fill_resident_windows(np.asarray([[0, 0, 0], [4, 0, 0]]), images)
    print({"train_loss": loss, **metrics, "fill_result": result})


if __name__ == "__main__":
    main()
