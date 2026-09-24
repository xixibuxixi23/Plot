"""Inspect one cached TextAgent target-pointer prediction."""
import argparse
import json
from pathlib import Path
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data.state_policy_dataset import load_state_cache
from plot.models.state_policy_text_v2 import IndependentStatePolicyV4, TextStatePolicyV2Args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--label", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = IndependentStatePolicyV4(TextStatePolicyV2Args(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device).eval()
    inputs = dict(load_state_cache(args.cache)["inputs"])
    text_path = json.loads(args.summary.read_text())["text_cache"]
    text = load_file(text_path, device="cpu")
    for name in ("shared", "current"):
        index = int(inputs.pop(name + "_text_id"))
        inputs[name + "_text"] = text["encoder_hidden"][index].float()
        inputs[name + "_text_mask"] = text["attention_mask"][index].bool()
    inputs = {key: value[None].to(args.device) for key, value in inputs.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(inputs)
    probability = logits["target"][0].float().softmax(-1)
    entity_probability = logits["target_entity"][0].float().softmax(-1)
    order = probability.argsort(descending=True)
    label = args.label
    rank = int((order == label).nonzero()[0]) + 1
    top = []
    for index in order[:10].tolist():
        kind = ("voxel" if index < 2197 else
                "resident" if index < probability.numel() - 1 else "null")
        row = dict(index=index, kind=kind, probability=float(probability[index]))
        if kind == "voxel":
            row["local"] = list(torch.unravel_index(torch.tensor(index), (13, 13, 13)))
            row["local"] = [int(value) for value in row["local"]]
        top.append(row)
    actions = model.decode(logits)[0].float().cpu()
    entities = []
    resident_count = inputs["resident_state"].shape[2]
    for index, value in enumerate(entity_probability.tolist()):
        row = dict(index=index,
                   kind="resident" if index < resident_count else "null",
                   probability=value)
        if index < resident_count:
            row["relative_position"] = [float(x) for x in
                (inputs["resident_state"][0, -1, index, :3] * 24).tolist()]
            row["resident_type"] = int(inputs["resident_type"][0, -1, index])
        entities.append(row)
    print(json.dumps(dict(label=label, label_local=[int(value) for value in
        torch.unravel_index(torch.tensor(label), (13, 13, 13))],
        label_rank=rank, label_probability=float(probability[label]), top10=top,
        entity_pointer=entities,
        place_actions=int((actions[:, 9] > 0).sum())), indent=2))


if __name__ == "__main__":
    main()
