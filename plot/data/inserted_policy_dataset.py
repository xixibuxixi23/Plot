"""Continuous episodes -> integrated read-only M3/M4 eight-frame chunks."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from .renderer_dataset import TextAgentRendererDataset, collate_renderer
from .fill_dataset import TextAgentFillDataset
from plot.policy_schema import FAMILY_TO_ID, PROFILE_TO_ID


def canonical_text(value):
    return " ".join(str(value or "").strip().split())


class InsertedPolicyDataset(Dataset):
    def __init__(self, index, vocabulary, text_catalog, dataset_root=None):
        self.rows = [json.loads(line) for line in Path(index).open() if line.strip()]
        if not self.rows: raise ValueError("M4 index is empty")
        self.vocabulary = vocabulary
        self.dataset_root = Path(dataset_root) if dataset_root is not None else None
        catalog = json.loads(Path(text_catalog).read_text())
        self.text_ids = {canonical_text(row["text"]): int(row["text_id"])
                         for row in catalog["texts"]}
        if "" not in self.text_ids: raise ValueError("text catalog must contain an empty entry")

    def __len__(self): return len(self.rows)

    def __getitem__(self, item):
        row = self.rows[item]; anchor, agent = int(row["anchor"]), int(row["agent_slot"])
        episode = Path(row["episode_path"])
        if not episode.is_absolute():
            if self.dataset_root is None:
                raise ValueError("relative M4 episode paths require dataset_root")
            episode = self.dataset_root / episode
        manifest = json.loads((episode / "manifest.json").read_text())
        with np.load(episode / manifest.get("training_data_file", "data.npz"),
                     allow_pickle=False) as data:
            model_start = TextAgentFillDataset._model_start(data, manifest)
            frames = len(data["cam_pos"])
            if frames - model_start < 65:
                raise ValueError("M4 requires an episode with one complete 65-frame M3 context")
            # Early decisions use the episode prefix; later decisions use the
            # longest 65-frame causal history ending at the current observation.
            context_start = int(row.get(
                "context_start", min(max(model_start, anchor - 64), frames - 65)))
            target = data["action_continuous"][anchor:anchor+8, agent].astype(np.float32)
        renderer = TextAgentRendererDataset.read_window(
            episode, self.vocabulary, start=context_start, target=agent,
            context_frames=65)
        renderer["conditions"]["condition_mask"] = torch.ones(65, dtype=torch.bool)
        policy_indices = torch.arange(anchor - 7 - context_start,
                                      anchor + 1 - context_start)
        shared = canonical_text(row.get("shared_task_text", row.get("task_text", "")))
        current = canonical_text(row.get("current_task_text", shared))
        if row["family"] != "language_builder": shared = current = ""
        return {
            "renderer": renderer,
            "family_id": torch.tensor(FAMILY_TO_ID[row["family"]]),
            "profile_id": torch.tensor(PROFILE_TO_ID[row["profile"]]),
            "shared_text_id": torch.tensor(self.text_ids[shared]),
            "current_text_id": torch.tensor(self.text_ids[current]),
            "target_actions": torch.from_numpy(target),
            "policy_indices": policy_indices,
            "valid_mask": torch.ones(8, dtype=torch.bool),
            "sample_weight": torch.tensor(float(row.get("sample_weight", 1.))),
        }


def collate_inserted_policy(samples):
    renderer = collate_renderer([sample.pop("renderer") for sample in samples])
    result = default_collate(samples)
    result.update(rgb=renderer["rgb"], conditions=renderer["conditions"])
    return result
