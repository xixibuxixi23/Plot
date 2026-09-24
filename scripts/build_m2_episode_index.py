"""Build a lightweight train/val_id episode index for M2 player rollouts."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits = {}
    for split in ("train", "val_id"):
        records = []
        for manifest_path in sorted((args.dataset_root / split).glob("*/manifest.json")):
            episode = manifest_path.parent
            validation = episode / "validation.json"
            if not validation.exists() or not json.loads(validation.read_text()).get("usable"):
                continue
            records.append({"path": str(episode.relative_to(args.dataset_root))})
        splits[split] = records
    payload = {"schema": "m2-player-rollout-index-v1",
               "dataset_root": str(args.dataset_root), "splits": splits}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps({key: len(value) for key, value in splits.items()}))


if __name__ == "__main__":
    main()
