"""Build bounded, state-only NPC pilot caches with episode-disjoint validation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from plot.data.fill_dataset import BlockVocabulary, TextAgentFillDataset
from plot.data.state_policy_dataset import read_state_sample
from plot.models.state_policy import STATE_PROFILES


def text_ids(manifest, target, catalog):
    canonical = lambda text: ' '.join(str(text or '').split())
    shared = canonical(manifest.get('task_text', ''))
    per_agent = manifest.get('agent_task_texts') or manifest.get('agent_subtasks') or {}
    current = canonical(per_agent.get(f'agent{target}', shared))
    if not shared or not current: raise ValueError('language sample lacks instruction text')
    return {name+'_text_id': torch.tensor(catalog[text], dtype=torch.int64)
            for name, text in (('shared', shared), ('current', current))}


def cache_episode(job):
    split, episode, start, windows, scenario, args, vocabulary, items = job
    manifest = json.loads((episode/'manifest.json').read_text())
    records = []; labels = []; activity = Counter()
    ordered = sorted(windows)
    if getattr(args, 'stop_when_summary_ready', False) and (args.output_dir/'summary.json').is_file():
        return records, labels, activity, scenario
    if getattr(args, 'resume_existing', False):
        names = [hashlib.sha256(f'{split}/{episode.name}/{anchor}/{target}'.encode()).hexdigest()[:24]+'.npz'
                 for anchor, target in ordered]
        if all((args.output_dir/name).is_file() for name in names):
            with np.load(episode/manifest.get('training_data_file', 'data.npz'), allow_pickle=False) as data:
                actions = data['action_continuous']
                for (anchor, target), name in zip(ordered, names):
                    label = actions[anchor:anchor+8, target].astype(np.float32)
                    labels.append(label)
                    attack_steps = int((label[:, 8] > 0).sum())
                    activity['attack_steps'] += attack_steps
                    activity['wait_steps'] += int((label == 0).all(-1).sum())
                    records.append(dict(profile=args.profile, episode_id=episode.name, anchor=anchor,
                                        agent_slot=target, cache=name, attack_steps=attack_steps,
                                        scenario_id=scenario))
            return records, labels, activity, scenario
    # One forward decompression pass per episode, bounded to one raw frame.
    with zipfile.ZipFile(episode/manifest.get('training_data_file', 'data.npz')) as archive:
        with archive.open('obs_voxel_mt.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            if fortran or dtype.hasobject: raise ValueError('unsupported voxel layout')
            stride = int(np.prod(shape[1:])) * dtype.itemsize
            previous = 0; last_anchor = -1; frame = None
            for anchor, target in ordered:
                if anchor != last_anchor:
                    stream.seek((anchor-previous)*stride, 1)
                    frame = np.frombuffer(stream.read(stride), dtype).reshape(shape[1:])
                    previous = anchor+1; last_anchor = anchor
                name = hashlib.sha256(f'{split}/{episode.name}/{anchor}/{target}'.encode()).hexdigest()[:24]+'.npz'
                sample = read_state_sample(episode, anchor, target, vocabulary, start, items,
                                           raw_observation=frame[target], include_m3_fields=getattr(args,'m3_state',False))
                if args.profile == 'language_builder':
                    sample['inputs'].update(text_ids(manifest, target, args.text_ids))
                arrays = {'inputs/'+key: value.numpy() for key,value in sample['inputs'].items()}
                arrays['inputs/voxel_classes'] = arrays['inputs/voxel_classes'].astype(np.uint16)
                arrays['target_actions'] = sample['target_actions'].numpy()
                arrays['valid_mask'] = sample['valid_mask'].numpy()
                destination = args.output_dir/name
                temporary = destination.with_suffix(f'.tmp.{os.getpid()}')
                with temporary.open('wb') as output:
                    np.savez_compressed(output, **arrays)
                temporary.replace(destination)
                actions = arrays['target_actions']; labels.append(actions)
                attack_steps = int((actions[:,8]>0).sum())
                activity['attack_steps'] += attack_steps
                activity['wait_steps'] += int((actions == 0).all(-1).sum())
                records.append(dict(profile=args.profile, episode_id=episode.name, anchor=anchor,
                                    agent_slot=target, cache=name, attack_steps=attack_steps, scenario_id=scenario))
    return records, labels, activity, scenario


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--vocabulary', type=Path, required=True)
    p.add_argument('--profile', choices=STATE_PROFILES, required=True)
    p.add_argument('--train-episodes', type=int, default=8)
    p.add_argument('--val-episodes', type=int, default=3)
    p.add_argument('--windows-per-episode', type=int, default=8)
    p.add_argument('--retain-attacks', action='store_true', help='retain all legal training attack windows')
    p.add_argument('--shuffle-episodes', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cache-workers', type=int, default=1)
    p.add_argument('--all-data', action='store_true', help='all eligible episodes and stride-8 windows; compact sequential cache')
    p.add_argument('--compact-cache', action='store_true')
    p.add_argument('--m3-state', action='store_true', help='include exact M3 raster camera and executed resident actions')
    p.add_argument('--text-catalog', type=Path)
    p.add_argument('--text-cache', type=Path)
    p.add_argument('--cache-shards', type=int, default=1,
                   help='number of node-level cache shards')
    p.add_argument('--cache-shard-index', type=int, default=0,
                   help='zero-based node-level cache shard')
    p.add_argument('--resume-existing', action='store_true',
                   help='reuse complete per-episode sample files already in output-dir')
    p.add_argument('--merge-wait-seconds', type=int, default=21600,
                   help='shard zero wait limit before merging all shard outputs')
    p.add_argument('--stop-when-summary-ready', action='store_true',
                   help='for speculative helpers: stop adding cache files after the primary merge')
    args = p.parse_args()
    if args.cache_shards < 1 or not 0 <= args.cache_shard_index < args.cache_shards:
        raise ValueError('invalid cache shard index/count')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir/'summary.json').exists():
        raise FileExistsError('cache already complete; choose a new output directory')
    vocabulary = BlockVocabulary.load(args.vocabulary)
    if args.profile == 'language_builder':
        if not args.text_catalog or not args.text_cache or not args.text_cache.is_file():
            raise ValueError('language policy requires an existing frozen text catalog/cache')
        catalog = json.loads(args.text_catalog.read_text())
        args.text_ids = {' '.join(row['text'].split()): row['text_id'] for row in catalog['texts']}
        from safetensors import safe_open
        with safe_open(args.text_cache, framework='pt', device='cpu') as cache:
            shape = cache.get_slice('encoder_hidden').get_shape()
            if shape[0] != len(catalog['texts']): raise ValueError('text cache/catalog count mismatch')
            text_hidden_size = shape[-1]
        args.compact_cache = True
    selected = {}; items = {'': 0}; seen_episodes = set()
    for split, limit in (('train', args.train_episodes), ('val_id', args.val_episodes)):
        episodes = []
        required_kind = {'language_builder':'human_like', 'zombie_melee':'npc_zombie', 'skeleton_swordsman':'npc_skeleton',
                         'villager_peaceful':'npc_villager', 'villager_defender':'npc_villager'}[args.profile]
        candidate_rows = [row for line in (args.dataset_root/f'{split}.jsonl').open()
                          if required_kind in (row := json.loads(line)).get('agent_kinds', {}).values()]
        if args.shuffle_episodes:
            np.random.default_rng(args.seed + (split != 'train')).shuffle(candidate_rows)
        for row in candidate_rows:
            episode = args.dataset_root/split/row['episode_id']
            manifest = json.loads((episode/'manifest.json').read_text())
            routes = manifest.get('behavior_routes', {})
            if not routes and row['scenario_id'] == 'S01':
                routes = {f'agent{i}': {'profile': 'language_builder'} for i in range(manifest['num_agents'])}
            targets = [int(k.removeprefix('agent')) for k, v in routes.items()
                       if v.get('profile') == args.profile]
            if not targets: continue
            if args.profile == 'language_builder':
                for target in targets: text_ids(manifest, target, args.text_ids)
            validation = episode/'validation.json'
            if validation.exists() and not json.loads(validation.read_text()).get('usable'): continue
            with np.load(episode/manifest.get('training_data_file', 'data.npz'), allow_pickle=False) as data:
                start = TextAgentFillDataset._model_start(data, manifest)
                actions = data['action_continuous']
                health, health_valid = data['player_health'], data['player_health_valid']
                termination = data['termination_flag'].reshape(len(actions), -1).any(-1)
                mask = data['language_policy_train_mask' if args.profile == 'language_builder' else 'npc_policy_train_mask']
                windows = []
                for target in targets:
                    candidates = []
                    for anchor in range(start, len(actions)-7, 8):
                        lo = max(start, anchor-7)
                        if not mask[anchor:anchor+8, target].all(): continue
                        if termination[lo:anchor+8].any(): continue
                        if not health_valid[lo:anchor+9, target].all(): continue
                        if (health[lo:anchor+9, target] <= 0).any(): continue
                        label = actions[anchor:anchor+8, target]
                        if args.profile == 'villager_peaceful' and label[:, [8,9,10,*range(12,21)]].any():
                            raise ValueError(f'peaceful constraint violation: {episode} {anchor}')
                        candidates.append((anchor, target))
                    if candidates:
                        # Spread pilot windows across each episode, retaining waits.
                        indices = (np.arange(len(candidates)) if args.all_data else
                                   np.linspace(0, len(candidates)-1,
                                               min(args.windows_per_episode, len(candidates))).round().astype(int))
                        selected_indices = set(indices.tolist())
                        if args.retain_attacks and split == 'train':
                            selected_indices.update(i for i,(anchor,_) in enumerate(candidates)
                                                    if (actions[anchor:anchor+8,target,8] > 0).any())
                        windows.extend(candidates[i] for i in sorted(selected_indices))
            if not windows: continue
            if row['episode_id'] in seen_episodes: raise ValueError('episode overlap across splits')
            seen_episodes.add(row['episode_id'])
            local_items = json.loads((episode/'training_metadata.json').read_text())['item_vocabulary']
            for name in sorted(local_items):
                if name not in items: items[name] = len(items)
            episodes.append((episode, start, windows, row['scenario_id']))
            print(json.dumps({'split': split, 'selected_episodes': len(episodes),
                              'profile': args.profile}), flush=True)
            if not args.all_data and len(episodes) >= limit: break
        if not episodes or (not args.all_data and len(episodes) < limit):
            raise ValueError(f'only {len(episodes)} valid {split} episodes')
        selected[split] = episodes
    if args.cache_shards == 1 or args.cache_shard_index == 0:
        (args.output_dir/'item_vocabulary.json').write_text(json.dumps(items, indent=2))
    summary = {'schema': 'm4-state-pilot-v1', 'profile': args.profile,
               'dataset_root': str(args.dataset_root.resolve()), 'num_block_classes': vocabulary.size,
               'class_to_raw': list(vocabulary.class_to_raw), 'item_vocabulary': items,
               'selection': {'seed': args.seed, 'shuffle_episodes': args.shuffle_episodes,
                             'retain_training_attacks': args.retain_attacks,
                             'windows_per_episode': None if args.all_data else args.windows_per_episode,
                             'all_data': args.all_data, 'anchor_stride': 8},
               'goal_available': False, 'splits': {},
               'm3_state': args.m3_state,
               'scope': 'all eligible episodes/windows; no behavior-success claim' if args.all_data else 'bounded pilot; no behavior-success claim'}
    if args.profile == 'language_builder':
        summary.update(text_cache=str(args.text_cache.resolve()), text_catalog=str(args.text_catalog.resolve()),
                       text_hidden_size=text_hidden_size,
                       text_catalog_sha256=hashlib.sha256(args.text_catalog.read_bytes()).hexdigest(),
                       text_encoder=catalog['encoder'])
    if args.all_data or args.compact_cache:
        shard_dir = args.output_dir/'.shards'
        if args.cache_shards > 1:
            shard_dir.mkdir(parents=True, exist_ok=True)
        shard_summaries = {}
        for split, episodes in selected.items():
            records = []; labels = []; activity = Counter(); scenarios = Counter()
            assigned = episodes[args.cache_shard_index::args.cache_shards]
            jobs = [(split, *episode, args, vocabulary, items) for episode in assigned]
            with ThreadPoolExecutor(max_workers=args.cache_workers) as pool:
                for episode_records, episode_labels, counts, scenario in pool.map(cache_episode, jobs):
                    records.extend(episode_records); labels.extend(episode_labels)
                    activity.update(counts); scenarios[scenario] += 1
                    print(json.dumps({'cached_episodes': sum(scenarios.values()), 'split': split,
                                      'windows': len(records), 'profile': args.profile}), flush=True)
            split_summary = dict(episodes=len(assigned), windows=len(records),
                                           scenarios=dict(scenarios), **activity)
            if args.cache_shards == 1:
                record_path = args.output_dir/f'{split}.jsonl'
                action_path = args.output_dir/f'{split}.actions.npy'
            else:
                stem = f'{split}.{args.cache_shard_index:03d}-of-{args.cache_shards:03d}'
                record_path = shard_dir/f'{stem}.jsonl'
                action_path = shard_dir/f'{stem}.actions.npy'
            record_path.write_text(''.join(json.dumps(r)+'\n' for r in records))
            action_array = np.stack(labels) if labels else np.empty((0, 8, 23), np.float32)
            np.save(action_path, action_array, allow_pickle=False)
            summary['splits'][split] = split_summary
            shard_summaries[split] = split_summary
        if args.cache_shards > 1:
            vocabulary_path = shard_dir/f'item_vocabulary.{args.cache_shard_index:03d}.json'
            vocabulary_path.write_text(json.dumps(items, indent=2))
            done_path = shard_dir/f'done.{args.cache_shard_index:03d}-of-{args.cache_shards:03d}.json'
            done_tmp = done_path.with_suffix('.tmp')
            done_tmp.write_text(json.dumps({'splits': shard_summaries}, indent=2))
            done_tmp.replace(done_path)
            if args.cache_shard_index != 0:
                print(json.dumps({'cache_shard_complete': args.cache_shard_index,
                                  'cache_shards': args.cache_shards}), flush=True)
                return
            deadline = time.time() + args.merge_wait_seconds
            done_paths = [shard_dir/f'done.{index:03d}-of-{args.cache_shards:03d}.json'
                          for index in range(args.cache_shards)]
            while not all(path.is_file() for path in done_paths):
                if time.time() >= deadline:
                    raise TimeoutError('timed out waiting for cache shards')
                time.sleep(10)
            vocabularies = [json.loads((shard_dir/f'item_vocabulary.{index:03d}.json').read_text())
                            for index in range(args.cache_shards)]
            if any(vocabulary != vocabularies[0] for vocabulary in vocabularies[1:]):
                raise ValueError('item vocabularies differ across cache shards')
            (args.output_dir/'item_vocabulary.json').write_text(json.dumps(vocabularies[0], indent=2))
            partials = [json.loads(path.read_text()) for path in done_paths]
            for split in selected:
                record_paths = [shard_dir/f'{split}.{index:03d}-of-{args.cache_shards:03d}.jsonl'
                                for index in range(args.cache_shards)]
                action_paths = [shard_dir/f'{split}.{index:03d}-of-{args.cache_shards:03d}.actions.npy'
                                for index in range(args.cache_shards)]
                with (args.output_dir/f'{split}.jsonl.tmp').open('wb') as destination:
                    for path in record_paths:
                        with path.open('rb') as source:
                            shutil.copyfileobj(source, destination, length=8 << 20)
                (args.output_dir/f'{split}.jsonl.tmp').replace(args.output_dir/f'{split}.jsonl')
                arrays = [np.load(path, mmap_mode='r') for path in action_paths]
                total = sum(len(array) for array in arrays)
                merged_path = args.output_dir/f'{split}.actions.tmp.npy'
                merged = np.lib.format.open_memmap(merged_path, mode='w+', dtype=np.float32,
                                                   shape=(total, 8, 23))
                offset = 0
                for array in arrays:
                    merged[offset:offset+len(array)] = array
                    offset += len(array)
                merged.flush(); del merged
                merged_path.replace(args.output_dir/f'{split}.actions.npy')
                combined = Counter()
                scenarios = Counter()
                for partial in partials:
                    values = partial['splits'][split]
                    combined.update({key: values.get(key, 0) for key in ('episodes', 'windows',
                                                                         'attack_steps', 'wait_steps')})
                    scenarios.update(values['scenarios'])
                summary['splits'][split] = dict(episodes=combined['episodes'], windows=combined['windows'],
                                                scenarios=dict(scenarios), attack_steps=combined['attack_steps'],
                                                wait_steps=combined['wait_steps'])
            summary_tmp = args.output_dir/'summary.tmp.json'
            summary_tmp.write_text(json.dumps(summary, indent=2))
            summary_tmp.replace(args.output_dir/'summary.json')
            print(json.dumps(summary), flush=True)
            return
        (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return
    for split, episodes in selected.items():
        records = []; activity = Counter()
        scenarios = Counter()
        for episode, start, windows, scenario in episodes:
            scenarios[scenario] += 1
            def make_sample(window):
                anchor, target = window
                name = hashlib.sha256(f'{split}/{episode.name}/{anchor}/{target}'.encode()).hexdigest()[:24]+'.pt'
                sample = read_state_sample(episode, anchor, target, vocabulary, start, items)
                torch.save(sample, args.output_dir/name)
                return name, sample
            with ThreadPoolExecutor(max_workers=args.cache_workers) as pool:
                cached = list(pool.map(make_sample, windows))
            for (anchor, target), (name, sample) in zip(windows, cached):
                actions = sample['target_actions']
                activity['attack_steps'] += int((actions[:, 8] > 0).sum())
                activity['wait_steps'] += int((actions == 0).all(-1).sum())
                records.append(dict(profile=args.profile, episode_id=episode.name,
                                    anchor=anchor, agent_slot=target, cache=name,
                                    attack_steps=int((actions[:,8]>0).sum()), scenario_id=scenario))
            print(json.dumps({'cached_episode': episode.name, 'windows': len(records)}), flush=True)
        (args.output_dir/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        summary['splits'][split] = dict(episodes=len(episodes), windows=len(records),
                                       scenarios=dict(scenarios), **activity)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
