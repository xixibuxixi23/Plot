"""Opt-in shared-trunk M2: state dynamics + remove/place/attack interactions.

Does not launch automatically or convert old checkpoints. Use a fresh output
directory. --resume only accepts this architecture with strict weight loading.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader

from plot.data.transition_dataset import collate_transition
from plot.data.transition_stream import TransitionStream
from plot.models.transition import TransitionArgs, TransitionNetwork
from plot.training.transition_trainer import transition_loss


def move(batch, device):
    return {g: {k: v.to(device) for k, v in batch[g].items()} for g in ('inputs', 'targets')}


@torch.no_grad()
def evaluate(model, loader, device, max_batches):
    model.eval()
    rows = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        values = move(batch, device)
        _, metrics = transition_loss(model, model(values['inputs']), **values, objective='unified')
        rows.append(metrics)
    if not rows:
        raise ValueError('validation stream contains no windows')
    result = {k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}
    for key in ('event_labels','predicted_events','correct_events','block_labels','correct_edits',
                'attack_labels','correct_attacks','supervised_event_windows',
                'incomplete_event_windows','overflow_event_windows'):
        result[key] = sum(r[key] for r in rows)
    result['event_precision'] = result['correct_events']/max(1,result['predicted_events'])
    result['event_recall'] = result['correct_events']/max(1,result['event_labels'])
    result['edit_recall'] = result['correct_edits']/max(1,result['block_labels'])
    result['attack_recall'] = result['correct_attacks']/max(1,result['attack_labels'])
    model.train()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('dataset-root','index','vocabulary','items','cache-root','output-dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--steps', type=int, default=10000)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--depth', type=int, default=6)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--event-queries', type=int, default=8)
    p.add_argument('--event-context-frames', type=int, choices=(0,4), default=4)
    p.add_argument('--use-rgb', action='store_true')
    p.add_argument('--air-raw-id', type=int, default=126)
    p.add_argument('--device', default='cuda')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--eval-batches', type=int, default=32)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--resume', help='Unified checkpoint only; use a fresh output directory')
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.eval_batches) < 1:
        p.error('steps/batch-size/save-every/eval-batches must be positive')
    if args.workers < 0 or args.width % args.heads:
        p.error('workers must be nonnegative and width divisible by heads')
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError('output directory must be new or empty; existing runs are never overwritten')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    item_data = json.loads(Path(args.items).read_text())
    items = item_data.get('items', item_data)
    cache = Path(args.cache_root)
    stamp = cache/'unified_cache_manifest.json'
    cache_identity = dict(schema='m2-unified-cache-v1', items=items,
                          dataset_root=str(Path(args.dataset_root).resolve()),
                          index_sha256=hashlib.sha256(Path(args.index).read_bytes()).hexdigest(),
                          vocabulary_sha256=hashlib.sha256(Path(args.vocabulary).read_bytes()).hexdigest())
    if stamp.exists():
        if json.loads(stamp.read_text()) != cache_identity:
            raise ValueError('cache identity differs; use a fresh cache root')
    elif cache.exists() and any(cache.iterdir()):
        raise ValueError('legacy/nonempty cache has no unified identity; use a fresh cache root')
    else:
        cache.mkdir(parents=True,exist_ok=True)
        stamp.write_text(json.dumps(cache_identity,indent=2))
    streams = {split: TransitionStream(args.index, args.vocabulary, args.cache_root, split,
               args.seed, dataset_root=args.dataset_root, items=items,
               fixed_order=(split == 'val_id')) for split in ('train','val_id')}
    train, val = streams['train'], streams['val_id']
    train_seeds = {Path(r['path']).name.split('_seed')[-1] for r in train.records}
    val_seeds = {Path(r['path']).name.split('_seed')[-1] for r in val.records}
    if train_seeds & val_seeds:
        raise ValueError('train/validation scenario seeds overlap')
    classes = tuple(train.vocabulary.class_to_raw)
    if args.air_raw_id not in classes:
        raise ValueError('air raw ID is absent from block vocabulary')
    cfg = TransitionArgs(len(classes), len(items), width=args.width, depth=args.depth, heads=args.heads,
                         event_queries=args.event_queries, event_context_frames=args.event_context_frames,
                         unified_interactions=True, unified_use_rgb=args.use_rgb,
                         air_class=classes.index(args.air_raw_id))
    model = TransitionNetwork(cfg).to(args.device)
    # Legacy diagnostic heads are retained for API compatibility, not trained.
    for name in ('hp_aux','pointer_query','occurrence_head','edit_kind_head','edit_age_head','edit_progress_head'):
        for param in getattr(model,name).parameters():
            param.requires_grad_(False)
    if not args.use_rgb:
        for param in model.visual.parameters():
            param.requires_grad_(False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    first_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
        old = checkpoint['config']
        if old['model'] != asdict(cfg) or old['items'] != items or tuple(old['class_to_raw']) != classes:
            raise ValueError('resume requires identical unified architecture and vocabularies')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        first_step = int(checkpoint['step'])
        if first_step >= args.steps:
            raise ValueError('--steps must exceed the resumed checkpoint step')
    loaders = {split: DataLoader(stream, batch_size=args.batch_size, num_workers=args.workers,
               collate_fn=collate_transition, persistent_workers=args.workers>0)
               for split,stream in streams.items()}
    config = dict(schema='m2-unified-v1', training=vars(args), model=asdict(cfg),
                  objective='unified', items=items, class_to_raw=classes,
                  operations=['remove','place','attack'], parameters=sum(p.numel() for p in model.parameters()))
    out.mkdir(parents=True, exist_ok=True)
    (out/'config.json').write_text(json.dumps(config, indent=2))
    def log(step, split, metrics):
        row = dict(step=step, split=split, **metrics)
        with (out/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
    iterator = iter(loaders['train'])
    for step in range(first_step+1, args.steps+1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders['train'])
            try:
                batch = next(iterator)
            except StopIteration as error:
                raise ValueError('training stream contains no windows') from error
        values = move(batch, args.device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = transition_loss(model, model(values['inputs']), **values, objective='unified')
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite unified loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('non-finite unified gradient')
        optimizer.step()
        log(step, 'train', dict(metrics, gradient_norm=float(norm)))
        if step % args.save_every == 0 or step == args.steps:
            tmp = out/'checkpoint.tmp'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                            step=step, config=config), tmp)
            tmp.replace(out/'checkpoint.pt')
            log(step, 'val_subset', evaluate(model, loaders['val_id'], args.device, args.eval_batches))
    (out/'COMPLETED.json').write_text(json.dumps(dict(steps=args.steps, schema='m2-unified-v1')))


if __name__ == '__main__':
    main()
