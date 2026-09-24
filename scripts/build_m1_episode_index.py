"""Build the compact M1 episode index from release split ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    payload: dict[str, object] = {
        "dataset_root": str(root),
        "splits": {},
    }
    splits: dict[str, list[dict[str, object]]] = {}
    for split in ("train", "val_id", "test_id"):
        records: list[dict[str, object]] = []
        ledger = root / f"{split}.jsonl"
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                manifest = json.loads(line)
                episode_id = manifest["episode_id"]
                records.append(
                    {
                        "path": f"{split}/{episode_id}",
                        "manifest": manifest,
                    }
                )
        splits[split] = records
    payload["splits"] = splits
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(json.dumps({name: len(rows) for name, rows in splits.items()}, sort_keys=True))


if __name__ == "__main__":
    main()
