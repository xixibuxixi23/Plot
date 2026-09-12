"""Find M3 target windows that contain non-full target health."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--full-health", type=float, default=20.0)
    args = parser.parse_args()

    source = torch.load(args.window_index, map_location="cpu", weights_only=False)
    context = int(source["context_frames"])
    episodes = source["episodes"]
    windows = source["windows"]
    root = Path(args.dataset_root)
    focused = []
    cursor = 0
    while cursor < len(windows):
        episode_id = int(windows[cursor][0])
        end = cursor + 1
        while end < len(windows) and int(windows[end][0]) == episode_id:
            end += 1
        episode = episodes[episode_id]
        path = Path(episode["path"])
        if not path.is_absolute():
            path = root / path
        data_file = episode["manifest"].get("training_data_file", "data.npz")
        with np.load(path / data_file, allow_pickle=False) as data:
            health = data["player_health"]
            for _, start, target in windows[cursor:end]:
                start, target = int(start), int(target)
                if np.min(health[start:start + context, target]) < args.full_health - 1e-3:
                    focused.append((episode_id, start, target))
        cursor = end

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(
        {
            "context_frames": context,
            "split": source["split"],
            "full_health": args.full_health,
            "source_window_index": str(args.window_index),
            "rows": torch.tensor(focused, dtype=torch.int32).reshape(-1, 3),
            "source_rows": len(windows),
        },
        temporary,
    )
    temporary.replace(output)
    print(
        f"wrote {len(focused)}/{len(windows)} focused target windows "
        f"({100 * len(focused) / max(1, len(windows)):.2f}%) to {output}"
    )


if __name__ == "__main__":
    main()
