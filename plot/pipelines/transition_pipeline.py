"""Commit parallel M2 predictions without running any engine physics."""
from copy import deepcopy

import numpy as np

from plot.transition_state import WriteEvent, held_item_trajectory

__all__ = ["TransitionCommitter", "held_item_trajectory"]


class TransitionCommitter:
    def __init__(self, memory, chars):
        self.memory=memory
        self.chars={row.resident_id:deepcopy(row) for row in chars}
        if len(self.chars)!=len(chars) or len({r.slot_order for r in chars})!=len(chars):
            raise ValueError('resident identities and stable slot orders must be unique')
        self.ledger=[]
        self.next_transition=0

    def commit(self, *, resident_ids, anchors, pose, address, block_payload, hp_payload,
               held_item, camera_relative, camera_direction, velocity=None, after_step=None):
        """Inputs are unbatched numpy arrays; callback captures memory before next step.

        This interface requires the fixed-hotbar adapter to have been validated.
        Auxiliary HP predictions are deliberately absent from the commit inputs.
        """
        a=len(resident_ids);anchors=np.asarray(anchors)
        if len(set(resident_ids))!=a or set(resident_ids)!=set(self.chars):
            raise ValueError('resident IDs must match the current state exactly')
        if anchors.shape!=(a,3) or pose.shape!=(8,a,5) or address.shape!=(8,a):
            raise ValueError('expected eight steps, all residents, and fixed integer anchors')
        if not np.equal(anchors,np.floor(anchors)).all():raise ValueError('anchors must be integers')
        if np.any(address<0) or np.any(address>2197+a):raise ValueError('invalid write address')
        if velocity is None:
            previous = np.asarray([self.chars[name].position_xyz for name in resident_ids], np.float32)
            velocity = np.diff(np.concatenate((previous[None], pose[..., :3]), axis=0), axis=0)
        velocity = np.asarray(velocity)
        for array in (pose,velocity,hp_payload,camera_relative,camera_direction):
            if not np.isfinite(array).all():raise ValueError('non-finite prediction')
        if hp_payload.shape!=(8,a) or block_payload.shape!=(8,a) or held_item.shape!=(8,a):
            raise ValueError('payload arrays must be [8,A]')
        if camera_relative.shape!=(8,a,3) or camera_direction.shape!=(8,a,3):
            raise ValueError('camera arrays must be [8,A,3]')
        if velocity.shape!=(8,a,3):raise ValueError('velocity must be [8,A,3]')
        if np.any(np.asarray(block_payload)<0):raise ValueError('negative block class')
        for i in range(a):
            if np.any(address[:,i]==2197+i):raise ValueError('self-address is not a resident interaction')
            for target in address[:,i]:
                if int(target)<2197:
                    xyz=anchors[i]-6+np.asarray(np.unravel_index(int(target),(13,13,13)))
                    if not self.memory.read_region(xyz,(1,1,1))[1].all():
                        raise ValueError('M1 must establish voxel targets before M2 commits')
        order=sorted(range(a),key=lambda i:self.chars[resident_ids[i]].slot_order)
        snapshots=[]
        for k in range(8):
            hp_changes={name:0. for name in resident_ids};events=[]
            for i in order:
                target=int(address[k,i]);source=resident_ids[i]
                if target==2197+a:continue
                if target<2197:
                    xyz=tuple(int(v) for v in (anchors[i]-6+np.asarray(np.unravel_index(target,(13,13,13)))))
                    value=int(block_payload[k,i])
                    self.memory.commit_points(np.asarray([xyz]),np.asarray([value]),allow_overwrite=True)
                    event=WriteEvent(self.next_transition,len(events),source,'voxel',xyz,value)
                else:
                    victim=resident_ids[target-2197];delta=float(hp_payload[k,i]);hp_changes[victim]+=delta
                    event=WriteEvent(self.next_transition,len(events),source,'character',victim,delta)
                self.ledger.append(event);events.append(event)
            for i,name in enumerate(resident_ids):
                row=self.chars[name]
                row.position_xyz=tuple(float(v) for v in pose[k,i,:3])
                row.velocity_xyz=tuple(float(v) for v in velocity[k,i])
                row.yaw,row.pitch=(float(v) for v in pose[k,i,3:])
                row.hp+=hp_changes[name]
                row.held_item=int(held_item[k,i])
                row.camera_relative=tuple(float(v) for v in camera_relative[k,i])
                row.camera_direction=tuple(float(v) for v in camera_direction[k,i])
            snapshots.append(deepcopy(self.chars))
            if after_step is not None:after_step(self.next_transition,self.memory,deepcopy(self.chars),tuple(events))
            self.next_transition+=1
        return snapshots
