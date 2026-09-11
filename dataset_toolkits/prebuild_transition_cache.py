"""Materialize one rank of the deterministic M2 cache without feeding a trainer."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot.data.transition_stream import TransitionStream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--vocabulary', required=True)
    parser.add_argument('--cache-root', required=True)
    parser.add_argument('--legacy-cache')
    parser.add_argument('--split', choices=['train', 'val_id'], required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--progress-dir', required=True)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error('rank must be in [0, world-size)')

    stream = TransitionStream(
        args.index, args.vocabulary, args.cache_root, args.split,
        rank=args.rank, world_size=args.world_size, fixed_order=True,
        legacy_cache=args.legacy_cache, dataset_root=args.dataset_root,
    )
    progress_dir = Path(args.progress_dir)
    progress_dir.mkdir(parents=True, exist_ok=True)
    samples = 0
    for sample in stream:
        samples += 1
        if samples % 256 == 0:
            (progress_dir / f'{args.split}_{args.rank:03d}.json').write_text(json.dumps({
                'complete': False, 'samples': samples,
            }))
    (progress_dir / f'{args.split}_{args.rank:03d}.json').write_text(json.dumps({
        'complete': True, 'samples': samples,
    }))


if __name__ == '__main__':
    main()
