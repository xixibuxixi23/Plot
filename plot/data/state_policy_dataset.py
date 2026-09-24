"""State-only adapters shared by offline data and committed-state inference."""
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def read_npz_slice(path, key, start, end):
    """Read contiguous frames without materializing a multi-GB voxel member.

    Stored NPZ members support direct seeking; compressed members may require
    decompression up to the selected frame, but memory stays bounded.
    """
    with zipfile.ZipFile(path) as archive, archive.open(key + '.npy') as stream:
        version = np.lib.format.read_magic(stream)
        shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
        if fortran or dtype.hasobject or not 0 <= start <= end <= shape[0]:
            raise ValueError(f'unsupported array/slice: {key} {shape}')
        stride = int(np.prod(shape[1:])) * dtype.itemsize
        stream.seek(start * stride, 1)
        payload = stream.read((end-start) * stride)
        return np.frombuffer(payload, dtype).reshape(end-start, *shape[1:]).copy()


def crop_observation(raw, source_center, anchor, vocabulary):
    """Resample the current observer crop in global integer coordinates."""
    out = np.zeros((48, 48, 48), np.int64)
    known = np.zeros(out.shape, bool)
    lower = np.asarray(anchor) - 24
    source_lower = np.asarray(source_center) - 24
    lo = np.maximum(lower, source_lower)
    hi = np.minimum(lower + 48, source_lower + 49)
    if (hi <= lo).any():
        return out, known
    src = tuple(slice(int(l), int(h)) for l, h in zip(lo-source_lower, hi-source_lower))
    dst = tuple(slice(int(l), int(h)) for l, h in zip(lo-lower, hi-lower))
    out[dst], known[dst] = vocabulary.encode(raw[src][..., 0])
    return out, known


def assemble_state_inputs(*, blocks, known, anchor, position, angles, hp,
                          camera_relative, camera_direction, event_cues,
                          held_item, resident_type, resident_valid, incoming_actions,
                          target, history_valid=None, goal_world=None, goal_radius=0.):
    """Eight completed states; no RGB, future state/action, or engine-only velocity.

    Local state privilege: residents outside +/-24 blocks at each observation
    are masked. Geometry is axis aligned, and its metric offset to the current
    target is encoded to preserve sub-voxel positioning.
    """
    position = np.asarray(position, np.float32)
    origin = position[-1, target]
    local = position - origin
    state = np.concatenate((local / 24, np.sin(angles), np.cos(angles),
                            np.asarray(hp)[..., None] / 20,
                            np.asarray(camera_relative) / 4,
                            camera_direction, np.asarray(event_cues) / 4), axis=-1).astype(np.float32)
    valid = np.asarray(resident_valid, bool).copy()
    local_at_time = position - position[:, target:target+1]
    valid &= (np.abs(local_at_time) < 24).all(-1)
    history_valid = (np.ones(8, bool) if history_valid is None else np.asarray(history_valid, bool))
    valid &= history_valid[:, None]
    goal = np.zeros(5, np.float32)
    if goal_world is not None:
        goal[:3] = (np.asarray(goal_world) - origin) / 24
        goal[3:] = [goal_radius / 24, 1.]
    result = dict(voxel_classes=np.asarray(blocks, np.int64), voxel_known=np.asarray(known, bool),
                  grid_offset=(np.asarray(anchor, np.float32)-.5-origin),
                  resident_state=np.where(valid[..., None], state, 0),
                  resident_valid=valid, held_item=np.where(valid, held_item, 0).astype(np.int64),
                  resident_type=np.where(valid, resident_type, 0).astype(np.int64),
                  incoming_actions=np.asarray(incoming_actions, np.float32),
                  history_valid=history_valid, target_agent=np.asarray(target, np.int64), goal=goal)
    return {key: torch.from_numpy(np.asarray(value).copy()) for key, value in result.items()}


def read_state_sample(episode, anchor, target, vocabulary, model_start, item_vocabulary, raw_observation=None,
                      include_m3_fields=False):
    episode = Path(episode)
    manifest = json.loads((episode/'manifest.json').read_text())
    path = episode/manifest.get('training_data_file', 'data.npz')
    start = max(model_start, anchor-7)
    ids = np.maximum(np.arange(anchor-7, anchor+1), model_start)
    take = ids-start
    with np.load(path, allow_pickle=False) as data:
        position = data['player_pos'][start:anchor+1][take].astype(np.float32)
        agents = position.shape[1]
        angles = np.deg2rad(np.stack((data['player_yaw'][start:anchor+1][take],
                                     data['player_pitch'][start:anchor+1][take]), -1)).astype(np.float32)
        hp = data['player_health'][start:anchor+1][take].astype(np.float32)
        cam = data['cam_pos'][start:anchor+1][take].astype(np.float32)
        direction = data['cam_dir'][start:anchor+1][take].astype(np.float32)
        # Matches the repository's transition-aligned held-item contract.
        raw_held = data['wielded_item_id'][ids, :agents]
        local_items = json.loads((episode/'training_metadata.json').read_text())['item_vocabulary']
        remap = np.zeros(max(local_items.values(), default=0)+1, np.int64)
        for name, index in local_items.items():
            remap[index] = item_vocabulary[name]
        held = remap[raw_held]
        entity_ids = list(data['entity_id'].astype(str))
        slots = [entity_ids.index(f'agent{i}') for i in range(agents)]
        valid = data['entity_valid'][ids][:, slots].astype(bool)
        actions = data['action_continuous']
        incoming = np.zeros((8, 23), np.float32)
        has_action = ids > model_start
        incoming[has_action] = actions[ids[has_action]-1, target]
        target_actions = actions[anchor:anchor+8, target].astype(np.float32)
        centers = data['obs_voxel_center'][anchor]
        health = data['player_health']
        cues = np.zeros((8, agents, 4), np.float32)
        cues[has_action, :, 3] = health[ids[has_action]] - health[ids[has_action]-1]
        if include_m3_fields:
            fov = data['fov_x'][anchor]
            fov = float(fov if np.ndim(fov) == 0 else fov[target])
            resident_actions = np.zeros((8, agents, 23), np.float32)
            resident_actions[has_action] = actions[ids[has_action]-1, :agents]
    # No instance masks or videos are opened. Only the anchor's voxel data is read.
    raw = (read_npz_slice(path, 'obs_voxel_mt', anchor, anchor+1)[0, target]
           if raw_observation is None else raw_observation)
    crop_anchor = np.rint(position[-1, target]).astype(np.int64)
    blocks, known = crop_observation(raw, centers[target], crop_anchor, vocabulary)
    event_path = episode/manifest.get('event_file', 'events.jsonl')
    if event_path.exists():
        for line in event_path.open():
            event = json.loads(line)
            frame = int(event.get('observation_frame', -1))
            if not start <= frame <= anchor or frame <= model_start:
                continue
            times = np.flatnonzero((ids == frame) & has_action)
            source = event.get('source', event.get('actor'))
            recipient = event.get('target')
            kind = event.get('event')
            for slot in range(agents):
                if source == f'agent{slot}' and kind in ('block_dug', 'block_placed'):
                    cues[times, slot, 0] += 1
                if kind == 'damage':
                    if source == f'agent{slot}': cues[times, slot, 1] += 1
                    if recipient == f'agent{slot}': cues[times, slot, 2] += 1
    kinds = {'human_like': 0, 'npc_villager': 1, 'npc_zombie': 2, 'npc_skeleton': 3}
    resident_type = np.broadcast_to([kinds[manifest['agent_kinds'][f'agent{i}']]
                                    for i in range(agents)], (8, agents))
    inputs = assemble_state_inputs(
        blocks=blocks, known=known, anchor=crop_anchor, position=position, angles=angles,
        hp=hp, camera_relative=cam-position, camera_direction=direction, event_cues=cues,
        held_item=held, resident_type=resident_type, resident_valid=valid,
        incoming_actions=incoming, target=target,
        history_valid=np.arange(anchor-7, anchor+1) >= model_start)
    if include_m3_fields:
        from .renderer_dataset import raster_camera
        inputs['raster_camera'] = torch.from_numpy(raster_camera(cam[-1,target],direction[-1,target],fov,crop_anchor))
        inputs['resident_actions'] = torch.from_numpy(resident_actions)
    # No verified guard-origin field exists in the current adapter. Explicitly
    # leave goal masked; do not infer a hidden teacher target from future labels.
    return dict(inputs=inputs, target_actions=torch.from_numpy(target_actions),
                valid_mask=torch.ones(8, dtype=torch.bool))


class CachedStatePolicyDataset(Dataset):
    def __init__(self, index, profile, preload=False):
        self.index = Path(index)
        self.rows = [r for line in self.index.open() if (r := json.loads(line))['profile'] == profile]
        if not self.rows: raise ValueError(f'no {profile} samples in {index}')
        self.text_cache = None
        if profile == 'language_builder':
            from safetensors.torch import load_file
            summary = json.loads((self.index.parent/'summary.json').read_text())
            self.text_cache = load_file(summary['text_cache'], device='cpu')
        self.samples = ([load_state_cache(self.index.parent/r['cache'])
                         for r in self.rows] if preload else None)
        target_path = self.index.with_suffix('.targets.npy')
        self.target_addresses = (np.load(target_path, mmap_mode='r', allow_pickle=False)
                                 if target_path.exists() else None)
        if self.target_addresses is not None and self.target_addresses.shape != (len(self),):
            raise ValueError(f'target/index mismatch: {target_path}')
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        sample = (self.samples[i] if self.samples is not None else
                  load_state_cache(self.index.parent/self.rows[i]['cache']))
        if self.text_cache is not None:
            inputs = dict(sample['inputs'])
            for name in ('shared', 'current'):
                index = int(inputs.pop(name+'_text_id'))
                inputs[name+'_text'] = self.text_cache['encoder_hidden'][index].float()
                inputs[name+'_text_mask'] = self.text_cache['attention_mask'][index].bool()
            sample = dict(sample, inputs=inputs)
        if self.target_addresses is not None:
            sample = dict(sample, target_address=torch.tensor(
                int(self.target_addresses[i]), dtype=torch.long))
        return sample

    def action_labels(self):
        path = self.index.with_suffix('.actions.npy')
        if path.exists():
            labels = torch.from_numpy(np.load(path, allow_pickle=False))
            if labels.shape != (len(self), 8, 23): raise ValueError('label/index mismatch')
            return labels
        return torch.stack([self[i]['target_actions'] for i in range(len(self))])


class RoleStatePolicyDataset(Dataset):
    """Attach a unified-policy role and synthesize masked text when absent."""

    def __init__(self, dataset, role, text_template):
        self.dataset = dataset
        self.role = int(role)
        self.text_template = text_template

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        inputs = dict(sample["inputs"])
        inputs["policy_role"] = torch.tensor(self.role, dtype=torch.long)
        for name in ("shared", "current"):
            text_key, mask_key = name + "_text", name + "_text_mask"
            if text_key not in inputs:
                inputs[text_key] = torch.zeros_like(self.text_template[text_key])
                inputs[mask_key] = torch.zeros_like(self.text_template[mask_key])
        return dict(sample, inputs=inputs)


class BalancedUnifiedStatePolicyDataset(Dataset):
    """Exactly balance zombie and builder windows while cycling the smaller set."""

    BUILDER_ROLE = 0
    ZOMBIE_ROLE = 1

    def __init__(self, zombie_index, text_index):
        zombie = CachedStatePolicyDataset(zombie_index, "zombie_melee")
        text = CachedStatePolicyDataset(text_index, "language_builder")
        prototype = text[0]["inputs"]
        template = {key: prototype[key] for key in (
            "shared_text", "shared_text_mask", "current_text", "current_text_mask")}
        self.zombie = RoleStatePolicyDataset(zombie, self.ZOMBIE_ROLE, template)
        self.text = RoleStatePolicyDataset(text, self.BUILDER_ROLE, template)
        self.per_role = max(len(self.zombie), len(self.text))

    def __len__(self):
        return 2 * self.per_role

    def __getitem__(self, index):
        role = index & 1
        offset = index // 2
        dataset = self.text if role == 0 else self.zombie
        return dataset[offset % len(dataset)]


def load_state_cache(path):
    if Path(path).suffix == '.pt':
        return torch.load(path, weights_only=True)
    with np.load(path, allow_pickle=False) as data:
        inputs = {k[7:]: torch.from_numpy(data[k].copy()) for k in data.files if k.startswith('inputs/')}
        inputs['voxel_classes'] = inputs['voxel_classes'].long()
        return dict(inputs=inputs, target_actions=torch.from_numpy(data['target_actions'].copy()),
                    valid_mask=torch.from_numpy(data['valid_mask'].copy()))


def collate_state_policy(samples):
    max_agents = max(s['inputs']['resident_state'].shape[1] for s in samples)
    per_agent = {'resident_state', 'resident_valid', 'held_item', 'resident_type', 'resident_actions'}
    inputs = {}
    for key in samples[0]['inputs']:
        values = []
        for sample in samples:
            value = sample['inputs'][key]
            if key in per_agent and value.shape[1] < max_agents:
                shape = list(value.shape); shape[1] = max_agents-value.shape[1]
                value = torch.cat((value, value.new_zeros(shape)), 1)
            values.append(value)
        inputs[key] = torch.stack(values)
    result = dict(inputs=inputs,
                  target_actions=torch.stack([s['target_actions'] for s in samples]),
                  valid_mask=torch.stack([s['valid_mask'] for s in samples]))
    if 'target_address' in samples[0]:
        result['target_address'] = torch.stack([s['target_address'] for s in samples])
    return result
