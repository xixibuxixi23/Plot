"""Train one independent M4-State NPC; CPU/single GPU or torchrun DDP."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler, Subset
from plot.data.state_policy_dataset import CachedStatePolicyDataset, collate_state_policy
from plot.models.state_policy import StateInhabitantPolicy, StatePolicyArgs, STATE_PROFILES


def move(batch, device):
    return {k: move(v, device) if isinstance(v, dict) else v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device, amp, shuffle_inputs=False, text_mode=None):
    model.eval(); totals = torch.zeros(6, device=device); horizon = torch.zeros(8, device=device)
    place_counts = torch.zeros(3, device=device)
    changed_text = 0
    for batch in loader:
        batch = move(batch, device)
        if shuffle_inputs:
            batch['inputs'] = {key:value.roll(1, dims=0) for key,value in batch['inputs'].items()}
        if text_mode == 'none':
            for name in ('shared', 'current'): batch['inputs'][name+'_text_mask'].zero_()
        elif text_mode == 'shuffled':
            before = batch['inputs']['current_text']
            changed_text += int((before != before.roll(1, dims=0)).flatten(1).any(-1).sum())
            for name in ('shared_text','current_text','shared_text_mask','current_text_mask'):
                batch['inputs'][name] = batch['inputs'][name].roll(1, dims=0)
        with amp():
            logits = model(batch['inputs'])
            loss, parts = model.loss(logits, batch['target_actions'], batch['valid_mask'])
        n = batch['target_actions'].shape[0]
        prediction = model.decode(logits)
        keys = model.head.key_indices
        totals[0] += loss * n; totals[1] += n
        totals[2] += ((prediction[..., keys] > 0) == (batch['target_actions'][..., keys] > 0)).float().mean() * n
        attack, true = prediction[..., 8] > 0, batch['target_actions'][..., 8] > 0
        totals[3] += (attack & true).sum(); totals[4] += (attack & ~true).sum(); totals[5] += (~attack & true).sum()
        placed, true_placed = prediction[..., 9] > 0, batch['target_actions'][..., 9] > 0
        place_counts += torch.stack(((placed & true_placed).sum(), (placed & ~true_placed).sum(),
                                     (~placed & true_placed).sum()))
        horizon += sum(parts.values()).mean(0) * n
    n = totals[1].clamp_min(1)
    result = dict(loss=float(totals[0]/n), key_accuracy=float(totals[2]/n),
                  attack_tp=int(totals[3]), attack_fp=int(totals[4]), attack_fn=int(totals[5]),
                  attack_precision=float(totals[3]/(totals[3]+totals[4]).clamp_min(1)),
                  attack_recall=float(totals[3]/(totals[3]+totals[5]).clamp_min(1)),
                  attack_f1=float(2*totals[3]/(2*totals[3]+totals[4]+totals[5]).clamp_min(1)),
                  horizon_loss=(horizon/n).tolist())
    tp,fp,fn = place_counts
    result.update(place_tp=int(tp),place_fp=int(fp),place_fn=int(fn),
                  place_precision=float(tp/(tp+fp).clamp_min(1)),
                  place_recall=float(tp/(tp+fn).clamp_min(1)),
                  place_f1=float(2*tp/(2*tp+fp+fn).clamp_min(1)))
    if text_mode == 'shuffled': result['changed_current_text_fraction'] = changed_text / float(n)
    model.train()
    return result


def marginal_baseline(model, train, val):
    """Fit only training labels, report on the unchanged validation distribution."""
    train_actions = train.action_labels()
    val_actions = val.action_labels()
    # Head buffers may reside on CUDA; use a small independent CPU loss/decode head.
    from plot.models.structured_action import StructuredActionHead, constrain_peaceful_logits
    head = StructuredActionHead(1,8)
    targets = head.targets(train_actions); n = len(train)
    probability = (targets['keys'].sum(0)+.5)/(n+1)
    logits = {'keys':torch.logit(probability)[None].expand(len(val),-1,-1)}
    for key,count in (('hotbar',10),('mouse_x',17),('mouse_y',17)):
        probability = (torch.nn.functional.one_hot(targets[key],count).float().sum(0)+.5)/(n+count*.5)
        logits[key] = probability.log()[None].expand(len(val),-1,-1)
    if model.cfg.profile == 'villager_peaceful':
        logits = constrain_peaceful_logits(logits, torch.ones(len(val),dtype=torch.bool))
    loss,_ = head.loss(logits,val_actions)
    predicted = head.decode(logits)[...,8] > 0
    actual = val_actions[...,8] > 0
    return dict(loss=float(loss), attack_tp=int((predicted & actual).sum()),
                attack_fp=int((predicted & ~actual).sum()), attack_fn=int((~predicted & actual).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--profile', choices=STATE_PROFILES, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--validate-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--precision', choices=('fp32','bf16'), default='bf16')
    p.add_argument('--balance-attack', action='store_true',
                   help='sample attack-containing and other train windows with equal total mass')
    p.add_argument('--attack-positive-weight', type=float, default=1.)
    p.add_argument('--preload-cache', action='store_true')
    p.add_argument('--train-eval-limit', type=int, default=0,
                   help='fixed diagnostic training subset only; zero evaluates all; validation always full')
    args = p.parse_args()
    world = int(os.environ.get('WORLD_SIZE', 1)); rank = int(os.environ.get('RANK', 0))
    local = int(os.environ.get('LOCAL_RANK', 0))
    if world > 1:
        dist.init_process_group('nccl' if args.device.startswith('cuda') else 'gloo')
    device = torch.device(f'cuda:{local}' if world > 1 and args.device.startswith('cuda') else args.device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda:0')
    if device.type == 'cuda': torch.cuda.set_device(device)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    summary = json.loads((args.cache_dir/'summary.json').read_text())
    if summary['profile'] != args.profile: raise ValueError('profile/cache mismatch')
    cfg = StatePolicyArgs(summary['num_block_classes'], max(summary['item_vocabulary'].values())+1,
                          args.profile, hidden=args.hidden, text_hidden_size=summary.get('text_hidden_size',0))
    raw = StateInhabitantPolicy(cfg).to(device)
    model = DistributedDataParallel(raw, device_ids=[local] if device.type == 'cuda' else None) if world > 1 else raw
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    train = CachedStatePolicyDataset(args.cache_dir/'train.jsonl', args.profile, preload=args.preload_cache)
    val = CachedStatePolicyDataset(args.cache_dir/'val_id.jsonl', args.profile, preload=args.preload_cache)
    sampler = DistributedSampler(train, num_replicas=world, rank=rank) if world > 1 else None
    sampling_info = {'mode': 'uniform'}
    if args.balance_attack:
        if world > 1:
            raise ValueError('attack-balanced pilot sampler is single-device only')
        positive = train.action_labels()[:,:,8].gt(0).any(-1)
        counts = [int((~positive).sum()), int(positive.sum())]
        if not min(counts):
            raise ValueError('balanced sampling requires both attack and non-attack windows')
        sample_weights = torch.where(positive, 1./counts[1], 1./counts[0])
        sampler = WeightedRandomSampler(sample_weights, len(train), replacement=True)
        sampling_info = {'mode': 'attack-balanced', 'nonattack_windows': counts[0],
                         'attack_windows': counts[1]}
    loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=collate_state_policy)
    # Rank zero validates the full held-out set; all ranks wait at a barrier.
    val_loader = DataLoader(val, batch_size=args.batch_size, num_workers=args.workers,
                           persistent_workers=args.workers > 0, collate_fn=collate_state_policy)
    # Episode-contiguous validation batches often have identical instructions.
    # Mix episodes before the text-only permutation and report actual changes.
    text_probe_loader = (DataLoader(val, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed+97), num_workers=args.workers,
        collate_fn=collate_state_policy) if args.profile == 'language_builder' else None)
    diagnostic = (Subset(train, np.random.default_rng(args.seed).choice(len(train),
                  min(args.train_eval_limit,len(train)), replace=False).tolist()) if args.train_eval_limit else train)
    train_eval = DataLoader(diagnostic, batch_size=args.batch_size, num_workers=args.workers,
                           persistent_workers=args.workers > 0, collate_fn=collate_state_policy)
    use_amp = device.type == 'cuda' and args.precision == 'bf16'
    amp = lambda: torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else nullcontext()
    output = args.output_dir
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        paths = ['plot/models/state_policy.py', 'plot/data/state_policy_dataset.py',
                 'train_scripts/train_state_policy.py', 'dataset_toolkits/prepare_state_policy.py']
        root = Path(__file__).resolve().parents[1]
        hashes = {s: hashlib.sha256((root/s).read_bytes()).hexdigest() for s in paths}
        for source in paths:
            destination = output/'source'/source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root/source, destination)
        (output/'config.json').write_text(json.dumps(dict(model=asdict(cfg),
            training={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            dataset=summary, source_sha256=hashes, sampling=sampling_info,
            parameters=sum(p.numel() for p in raw.parameters())), indent=2))
        baseline = marginal_baseline(raw, train, val)
        (output/'baseline.json').write_text(json.dumps(baseline,indent=2))
        print(json.dumps({'baseline':baseline, 'train_windows':len(train), 'val_windows':len(val)}),flush=True)
    if world > 1: dist.barrier()
    weights = torch.arange(8,0,-1,device=device)
    best_loss, best_f1 = float('inf'), -1.
    def report(step):
        nonlocal best_loss, best_f1
        if rank == 0:
            record = dict(step=step, train_diagnostic_windows=len(diagnostic),
                          train=evaluate(raw, train_eval, device, amp),
                          val=evaluate(raw, val_loader, device, amp))
            if step == args.steps:
                record['val_shuffled_inputs'] = evaluate(raw, val_loader, device, amp, shuffle_inputs=True)
                if args.profile == 'language_builder':
                    record['val_no_text'] = evaluate(raw, val_loader, device, amp, text_mode='none')
                    record['val_shuffled_text'] = evaluate(raw, text_probe_loader, device, amp, text_mode='shuffled')
            with (output/'validation.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
            print(json.dumps(record), flush=True)
            checkpoint = dict(model=raw.state_dict(), config=asdict(cfg), step=step,
                              optimizer=optimizer.state_dict(), dataset=summary)
            temporary = output/'latest.tmp'; torch.save(checkpoint, temporary)
            temporary.replace(output/'latest.pt')
            if record['val']['loss'] < best_loss:
                best_loss = record['val']['loss']
                shutil.copy2(output/'latest.pt', output/'best-val-loss.pt')
            if record['val']['attack_f1'] > best_f1:
                best_f1 = record['val']['attack_f1']
                shutil.copy2(output/'latest.pt', output/'best-attack-f1.pt')
        if world > 1: dist.barrier()
    report(0)
    iterator = iter(loader); epoch = 0; started = time.monotonic()
    for step in range(1, args.steps+1):
        try: batch = next(iterator)
        except StopIteration:
            epoch += 1
            if isinstance(sampler, DistributedSampler): sampler.set_epoch(epoch)
            iterator = iter(loader); batch = next(iterator)
        batch = move(batch, device); optimizer.zero_grad(set_to_none=True)
        with amp():
            logits = model(batch['inputs'])
            loss, _ = raw.loss(logits, batch['target_actions'], batch['valid_mask'], horizon_weights=weights,
                               attack_positive_weight=args.attack_positive_weight)
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
        loss.backward(); grad = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if rank == 0 and (step == 1 or step % 10 == 0):
            record = dict(step=step, loss=float(loss), grad_norm=float(grad),
                          elapsed_seconds=time.monotonic()-started)
            print(json.dumps(record), flush=True)
            with (output/'train.jsonl').open('a') as stream: stream.write(json.dumps(record)+'\n')
        if step % args.validate_every == 0 or step == args.steps: report(step)
    if rank == 0: (output/'COMPLETE.json').write_text(json.dumps({'steps': args.steps, 'profile': args.profile,
        'best_val_loss':best_loss, 'best_attack_f1':best_f1,
        'gpu_peak_allocated_mb':torch.cuda.max_memory_allocated(device)/2**20 if device.type == 'cuda' else 0}))
    if world > 1: dist.destroy_process_group()


if __name__ == '__main__': main()
