"""Capture committed state before M3, then run independent NPC inference.

The caller can dispatch inference and M3 rendering concurrently after capture.
This adapter does not remove the existing M1/M2 dependencies on recent RGB.
"""
from collections import deque
import json
from pathlib import Path

import numpy as np
import torch

from plot.data.state_policy_dataset import assemble_state_inputs
from plot.models.state_policy import StateInhabitantPolicy, StatePolicyArgs


class CachedInstructionEncoder:
    """Frozen T5 features for catalog instructions; unknown text fails explicitly.

    New instructions require the same T5 encoder/tokenization as the training
    catalog. Text IDs are lookup keys only, never trainable task-ID embeddings.
    """
    def __init__(self, catalog, cache):
        from safetensors.torch import load_file
        metadata = json.loads(Path(catalog).read_text())
        self.ids = {' '.join(row['text'].split()): row['text_id'] for row in metadata['texts']}
        self.cache = load_file(str(cache), device='cpu')
        if len(self.cache['encoder_hidden']) != len(self.ids): raise ValueError('catalog/cache mismatch')

    def __call__(self, shared, current=None):
        result = {}
        for name, text in (('shared',shared), ('current',shared if current is None else current)):
            canonical = ' '.join(text.split())
            if canonical not in self.ids:
                raise ValueError('instruction not cached; encode it with the matching frozen T5 first')
            index = self.ids[canonical]
            result[name+'_text'] = self.cache['encoder_hidden'][index].float().clone()
            result[name+'_text_mask'] = self.cache['attention_mask'][index].bool().clone()
        return result


class StatePolicySession:
    """Per-resident history; same-profile sessions may share a model instance."""
    def __init__(self, model, resident_ids, target, item_remap):
        self.model = model.eval()
        self.resident_ids = tuple(resident_ids)
        self.target = self.resident_ids.index(target)
        self.item_remap = dict(item_remap)
        self.frames = deque(maxlen=8)

    @classmethod
    def from_checkpoint(cls, path, *, resident_ids, target, deployment_items,
                        deployment_blocks, device='cpu'):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        metadata = checkpoint['dataset']
        if list(deployment_blocks) != metadata['class_to_raw']:
            raise ValueError('deployment block vocabulary does not match checkpoint')
        training_items = metadata['item_vocabulary']
        remap = {index: training_items[name] for name, index in deployment_items.items()
                 if name in training_items}
        model = StateInhabitantPolicy(StatePolicyArgs(**checkpoint['config']))
        model.load_state_dict(checkpoint['model'], strict=True)
        return cls(model.to(device), resident_ids, target, remap)

    def append(self, chars, *, incoming_actions=None, events=()):
        """Call once per completed observation (including initial state).

        incoming_actions is [A,23], already executed. Write events must have
        been committed; never append predictions before conflict resolution.
        """
        rows = [chars[name] for name in self.resident_ids]
        a = len(rows)
        hp = np.asarray([r.hp for r in rows], np.float32)
        cues = np.zeros((a, 4), np.float32)
        if self.frames: cues[:, 3] = hp-self.frames[-1]['hp']
        for event in events:
            if event.source in self.resident_ids:
                cues[self.resident_ids.index(event.source), 0 if event.target_kind == 'voxel' else 1] += 1
            if event.target_kind != 'voxel' and event.target in self.resident_ids:
                cues[self.resident_ids.index(event.target), 2] += 1
        action = np.zeros((a,23), np.float32) if incoming_actions is None else np.asarray(incoming_actions, np.float32)
        if action.shape != (a,23): raise ValueError('incoming actions must be [A,23]')
        self.frames.append(dict(
            position=np.asarray([r.position_xyz for r in rows], np.float32),
            angles=np.asarray([[r.yaw,r.pitch] for r in rows], np.float32), hp=hp,
            camera_relative=np.asarray([r.camera_relative for r in rows], np.float32),
            camera_direction=np.asarray([r.camera_direction for r in rows], np.float32),
            event_cues=cues, held_item=np.asarray([self.item_remap[r.held_item] for r in rows]),
            resident_type=np.asarray([r.resident_type for r in rows]),
            resident_valid=np.ones(a,bool), incoming_actions=action[self.target].copy()))

    def capture(self, memory, *, text_condition=None):
        """Freeze geometry/history before either M3 or next M2 can advance."""
        if not self.frames: raise ValueError('append initial/completed state first')
        frames = [self.frames[0]]*(8-len(self.frames)) + list(self.frames)
        values = {key: np.stack([frame[key] for frame in frames]) for key in frames[0]}
        valid = np.arange(8) >= 8-len(self.frames)
        values['incoming_actions'][~valid] = 0
        anchor = np.rint(values['position'][-1,self.target]).astype(np.int64)
        blocks, known = memory.read_tile(anchor)
        result = assemble_state_inputs(blocks=blocks.copy(), known=known.copy(), anchor=anchor,
                                       target=self.target, history_valid=valid, **values)
        if self.model.cfg.profile == 'language_builder':
            if text_condition is None: raise ValueError('language policy needs an instruction')
            for key in ('shared_text','current_text','shared_text_mask','current_text_mask'):
                result[key] = text_condition[key].detach().clone()
        return result

    @torch.no_grad()
    def act(self, frozen_inputs):
        device = next(self.model.parameters()).device
        inputs = {key:value[None].to(device) for key,value in frozen_inputs.items()}
        return self.model.decode(self.model(inputs))[0]
