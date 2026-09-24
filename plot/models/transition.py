"""M2: bidirectional joint-action transformer and typed sparse write heads."""
from dataclasses import dataclass, field

import torch
from torch import nn

from plot.kinematics import KinematicsConfig, propose_trajectory, wrap_angle
from plot.transition_state import held_item_trajectory


@dataclass(frozen=True)
class TransitionArgs:
    num_block_classes: int
    num_items: int
    width: int = 256
    depth: int = 6
    heads: int = 8
    event_queries: int = 4
    event_decoder_layers: int = 2
    event_context_frames: int = 0
    player_branch: bool = False
    player_depth: int = 3
    player_velocity: bool = False
    player_voxel: bool = False
    attack_branch: bool = False
    attack_queries: int = 2
    attack_decoder_layers: int = 2
    attack_null_logit_bias: float = 0.
    kinematics: KinematicsConfig = field(default_factory=KinematicsConfig)


class TransitionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.time = nn.MultiheadAttention(width,heads,batch_first=True)
        self.agents = nn.MultiheadAttention(width,heads,batch_first=True)
        self.memory = nn.MultiheadAttention(width,heads,batch_first=True)
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(4)])
        self.ffn = nn.Sequential(nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))

    def forward(self, q, memory, memory_padding, active):
        b,t,a,d = q.shape
        x = self.norms[0](q).transpose(1,2).reshape(b*a,t,d)
        q = q + self.time(x,x,x,need_weights=False)[0].reshape(b,a,t,d).transpose(1,2)
        x = self.norms[1](q).reshape(b*t,a,d)
        padding = ~active[:,None].expand(b,t,a).reshape(b*t,a)
        q = q + self.agents(x,x,x,key_padding_mask=padding,need_weights=False)[0].reshape(b,t,a,d)
        x = self.norms[2](q).transpose(1,2).reshape(b*a,t,d)
        q = q + self.memory(x,memory,memory,key_padding_mask=memory_padding,
                            need_weights=False)[0].reshape(b,a,t,d).transpose(1,2)
        q = q + self.ffn(self.norms[3](q))
        return q * active[:,None,:,None]


class PlayerBlock(nn.Module):
    """Temporal and cross-resident dynamics without renderer/voxel features."""
    def __init__(self,width,heads):
        super().__init__()
        self.time=nn.MultiheadAttention(width,heads,batch_first=True)
        self.agents=nn.MultiheadAttention(width,heads,batch_first=True)
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.ffn=nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))

    def forward(self,q,active):
        b,t,a,d=q.shape
        x=self.norms[0](q).transpose(1,2).reshape(b*a,t,d)
        q=q+self.time(x,x,x,need_weights=False)[0].reshape(b,a,t,d).transpose(1,2)
        x=self.norms[1](q).reshape(b*t,a,d)
        padding=~active[:,None].expand(b,t,a).reshape(b*t,a)
        q=q+self.agents(x,x,x,key_padding_mask=padding,need_weights=False)[0].reshape(b,t,a,d)
        q=q+self.ffn(self.norms[2](q))
        return q*active[:,None,:,None]


class TransitionNetwork(nn.Module):
    def __init__(self, cfg: TransitionArgs):
        super().__init__()
        self.cfg = cfg
        d = cfg.width
        self.block_embedding = nn.Embedding(cfg.num_block_classes+1,d)
        self.voxel_cnn = nn.Sequential(nn.Conv3d(d,d,3,padding=1,groups=d),nn.GELU(),nn.Conv3d(d,d,1))
        self.coordinate = nn.Sequential(nn.Linear(3,d),nn.SiLU(),nn.Linear(d,d))
        self.item_embedding = nn.Embedding(cfg.num_items,32)
        self.type_embedding = nn.Embedding(4,16)
        self.state = nn.Sequential(nn.Linear(1+4+32+16+3+3,d),nn.SiLU(),nn.Linear(d,d))
        # Pilot visual branch retains spatial tokens from the last completed RGB.
        self.visual = nn.Sequential(nn.Conv2d(3,32,5,stride=2,padding=2),nn.GELU(),
                                    nn.Conv2d(32,d,3,stride=2,padding=1),nn.GELU(),
                                    nn.AdaptiveAvgPool2d((4,4)))
        self.held_condition = nn.Linear(32,d)
        nn.init.zeros_(self.held_condition.weight);nn.init.zeros_(self.held_condition.bias)
        self.action = nn.Linear(23,d)
        self.proposal = nn.Linear(7,d)
        self.time_embedding = nn.Parameter(torch.randn(8,d)*.02)
        self.blocks = nn.ModuleList([TransitionBlock(d,cfg.heads) for _ in range(cfg.depth)])
        # Player-only finetuning uses this residual branch.  Its zero-initialized
        # output preserves the event representation of older checkpoints.
        self.player_adapter = nn.Sequential(
            nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        nn.init.zeros_(self.player_adapter[-1].weight)
        nn.init.zeros_(self.player_adapter[-1].bias)
        self.pose_head = nn.Linear(d,5)
        self.camera_head = nn.Linear(d,6)
        self.hp_aux = nn.Linear(d,1)
        self.held_head = nn.Linear(d,cfg.num_items)
        self.pointer_query = nn.Linear(d,d)
        self.pointer_key = nn.Linear(d,d)
        self.occurrence_head = nn.Linear(d,1)
        nn.init.zeros_(self.occurrence_head.weight)
        nn.init.constant_(self.occurrence_head.bias,-4.)
        self.edit_kind_head = nn.Linear(d,3)
        self.edit_age_head = nn.Linear(d,1)
        self.edit_progress_head = nn.Linear(d,1)
        # Window-level ordered events use a single categorical count followed by
        # ordered slots. This is the production architecture.
        event_layer = nn.TransformerDecoderLayer(
            d_model=d,nhead=cfg.heads,dim_feedforward=4*d,dropout=.1,
            activation='gelu',batch_first=True,norm_first=True)
        self.event_decoder = nn.TransformerDecoder(
            event_layer,cfg.event_decoder_layers,norm=nn.LayerNorm(d))
        self.event_query = nn.Embedding(cfg.event_queries,d)
        self.event_count_head = nn.Sequential(
            nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),
            nn.Linear(d,cfg.event_queries+1))
        self.event_time_head = nn.Linear(d,8)
        self.event_operation_head = nn.Linear(d,2)
        self.event_pointer_query = nn.Linear(d,d)
        if cfg.event_context_frames:
            self.event_context_time_embedding=nn.Parameter(
                torch.randn(cfg.event_context_frames,d)*.02)
            self.event_context_norm=nn.LayerNorm(d)
        if cfg.player_branch:
            self.player_item_embedding=nn.Embedding(cfg.num_items,32)
            self.player_type_embedding=nn.Embedding(4,16)
            self.player_state=nn.Sequential(nn.Linear(1+4+32+16+3+3,d),nn.SiLU(),nn.Linear(d,d))
            self.player_action=nn.Linear(23,d)
            self.player_proposal=nn.Linear(7,d)
            self.player_held_condition=nn.Linear(32,d)
            if cfg.player_velocity:
                self.player_velocity=nn.Linear(3,d)
                nn.init.zeros_(self.player_velocity.weight)
                nn.init.zeros_(self.player_velocity.bias)
            if cfg.player_voxel:
                self.player_voxel_norm=nn.ModuleList([nn.LayerNorm(d) for _ in range(cfg.player_depth)])
                self.player_voxel_attn=nn.ModuleList([
                    nn.MultiheadAttention(d,cfg.heads,batch_first=True)
                    for _ in range(cfg.player_depth)])
                self.player_voxel_null=nn.Parameter(torch.randn(d)*.02)
                for attention in self.player_voxel_attn:
                    nn.init.zeros_(attention.out_proj.weight)
                    nn.init.zeros_(attention.out_proj.bias)
            self.player_time_embedding=nn.Parameter(torch.randn(8,d)*.02)
            self.player_blocks=nn.ModuleList([PlayerBlock(d,cfg.heads) for _ in range(cfg.player_depth)])
        if cfg.attack_branch:
            if not cfg.player_branch:raise ValueError('attack branch requires player branch')
            attack_layer=nn.TransformerDecoderLayer(
                d_model=d,nhead=cfg.heads,dim_feedforward=4*d,dropout=.1,
                activation='gelu',batch_first=True,norm_first=True)
            self.attack_decoder=nn.TransformerDecoder(
                attack_layer,cfg.attack_decoder_layers,norm=nn.LayerNorm(d))
            self.attack_query=nn.Embedding(cfg.attack_queries,d)
            self.attack_count_head=nn.Sequential(
                nn.LayerNorm(2*d),nn.Linear(2*d,d),nn.GELU(),
                nn.Linear(d,cfg.attack_queries+1))
            self.attack_time_head=nn.Linear(d,8)
            self.attack_target_query=nn.Linear(d,d)
            self.attack_target_key=nn.Linear(d,d)
            self.attack_damage_head=nn.Linear(d,1)
        self.null_token = nn.Parameter(torch.randn(d)*.02)
        self.target_type = nn.Embedding(3,d)
        self.payload = nn.Sequential(nn.Linear(2*d,d),nn.SiLU())
        self.block_head = nn.Linear(d,cfg.num_block_classes)
        self.damage_head = nn.Linear(d,1)
        nn.init.zeros_(self.pose_head.weight); nn.init.zeros_(self.pose_head.bias)
        nn.init.zeros_(self.camera_head.weight); nn.init.zeros_(self.camera_head.bias)

    def forward(self, inputs):
        active = inputs['active'].bool()
        b,a = active.shape
        if not active.any(1).all():
            raise ValueError("each sample needs at least one real resident")
        d = self.cfg.width
        known = inputs['voxel_known'].bool()
        ids = torch.where(known,inputs['voxels'].long(),self.cfg.num_block_classes)
        v = self.block_embedding(ids).reshape(b*a,13,13,13,d).permute(0,4,1,2,3)
        v = (v+self.voxel_cnn(v)).flatten(2).transpose(1,2).reshape(b,a,2197,d)
        v = v + self.coordinate(inputs['voxel_relative_xyz']/6.)
        pose = inputs['initial_pose']
        state = torch.cat((inputs['initial_hp'][...,None]/20., pose[...,3:].sin(),pose[...,3:].cos(),
                           self.item_embedding(inputs['held_item']),self.type_embedding(inputs['resident_type']),
                           inputs['camera_relative'],inputs['camera_direction']),-1)
        state = self.state(state) * active[...,None]
        relative = (pose[:,None,:,:3]-pose[:,:,None,:3])/6.
        residents = state[:,None].expand(-1,a,-1,-1) + self.coordinate(relative)
        pixels = self.visual(inputs['previous_rgb'].reshape(b*a,3,*inputs['previous_rgb'].shape[-2:]))
        pixels = pixels.flatten(2).transpose(1,2).reshape(b,a,16,d)
        memory = torch.cat((v,residents,pixels),2).reshape(b*a,2197+a+16,d)
        # Unknown is an explicit input token; only padded resident keys are excluded.
        padding = torch.cat((torch.zeros(b,a,2197,device=v.device,dtype=torch.bool),
                             ~active[:,None].expand(b,a,a),
                             torch.zeros(b,a,16,device=v.device,dtype=torch.bool)),2).reshape(b*a,-1)
        proposal = propose_trajectory(pose,inputs['actions'],self.cfg.kinematics)
        proposal_features = torch.cat((proposal[...,:3]/6.,proposal[...,3:].sin(),proposal[...,3:].cos()),-1)
        q = self.action(inputs['actions']) + self.proposal(proposal_features)
        held,held_valid=held_item_trajectory(inputs)
        held=torch.where(held_valid,held,inputs['held_item'][:,None])
        q = q + self.held_condition(self.item_embedding(held))
        q = q + state[:,None] + pixels.mean(2)[:,None] + self.time_embedding[None,:,None]
        for block in self.blocks:
            q = block(q,memory,padding,active)
        player_q = q + self.player_adapter(q)
        candidates = torch.cat((v+self.target_type.weight[0],residents+self.target_type.weight[1],
                                 (self.null_token+self.target_type.weight[2]).expand(b,a,1,d)),2)
        logits = torch.einsum('btad,band->btan',self.pointer_query(q),self.pointer_key(candidates))/d**.5
        candidate_valid = torch.cat((known.flatten(2),active[:,None].expand(b,a,a),
                                     torch.ones(b,a,1,dtype=torch.bool,device=q.device)),2)
        diagonal = torch.eye(a,dtype=torch.bool,device=q.device)[None]
        candidate_valid[:,:,2197:2197+a] &= ~diagonal
        logits = logits.masked_fill(~candidate_valid[:,None],-torch.inf)
        residual = self.pose_head(player_q)
        predicted = torch.cat((pose[:,None,:,:3]+proposal[...,:3]+residual[...,:3],
                                wrap_angle(proposal[...,3:]+residual[...,3:])), -1)
        velocity = torch.cat((predicted[:, :1, :, :3] - pose[:, None, :, :3],
                              predicted[:, 1:, :, :3] - predicted[:, :-1, :, :3]), 1)
        camera = self.camera_head(player_q)
        direction = nn.functional.normalize(inputs['camera_direction'][:,None]+camera[...,3:],dim=-1)
        event_memory=q.transpose(1,2).reshape(b*a,8,d)
        event_memory_valid=torch.ones(b*a,8,dtype=torch.bool,device=q.device)
        if self.cfg.event_context_frames:
            frames=self.cfg.event_context_frames
            context_pose=inputs['event_context_pose']
            if context_pose.shape[1]!=frames:
                raise ValueError('event context length does not match model configuration')
            context_held=inputs['event_context_held_item'].long()
            context_state=torch.cat((
                inputs['event_context_hp'][...,None]/20.,
                context_pose[...,3:].sin(),context_pose[...,3:].cos(),
                self.item_embedding(context_held),
                self.type_embedding(inputs['resident_type'])[:,None].expand(-1,frames,-1,-1),
                inputs['event_context_camera_relative'],
                inputs['event_context_camera_direction']),-1)
            context_valid=(inputs['event_context_valid'].bool()
                           &active[:,None])
            context=(self.action(inputs['event_context_actions'])
                     +self.state(context_state)
                     +self.held_condition(self.item_embedding(context_held))
                     +self.event_context_time_embedding[None,:,None])
            context=self.event_context_norm(context)*context_valid[...,None]
            context=context.transpose(1,2).reshape(b*a,frames,d)
            context_valid=context_valid.transpose(1,2).reshape(b*a,frames)
            event_memory=torch.cat((context,event_memory),1)
            event_memory_valid=torch.cat((context_valid,event_memory_valid),1)
        event_query=self.event_query.weight[None].expand(b*a,-1,-1)
        event=self.event_decoder(
            event_query,event_memory,memory_key_padding_mask=~event_memory_valid
        ).reshape(b,a,self.cfg.event_queries,d)
        summary=(event_memory*event_memory_valid[...,None]).sum(1)
        summary=summary/event_memory_valid.sum(1,keepdim=True).clamp_min(1)
        event_summary=summary.reshape(b,a,d)
        event_count=self.event_count_head(torch.cat((event_summary,event.mean(2)),-1))
        event_candidates=candidates[:,:,:-1]
        event_valid=candidate_valid[:,:,:-1]
        event_address=torch.einsum(
            'baqd,band->baqn',self.event_pointer_query(event),self.pointer_key(event_candidates)
        )/d**.5
        event_address=event_address.masked_fill(~event_valid[:,:,None],-torch.inf)
        player=(self.forward_player(inputs) if self.cfg.player_branch else {
            'held_logits':self.held_head(player_q),'pose':predicted,'velocity':velocity,
            'proposal':proposal,'pose_residual':residual,
            'camera_relative':inputs['camera_relative'][:,None]+camera[...,:3],
            'camera_direction':direction,'player_hidden':player_q})
        return {**player,'occurrence_logits':self.occurrence_head(q).squeeze(-1),
                'edit_kind_logits':self.edit_kind_head(q),
                'edit_age':self.edit_age_head(q).squeeze(-1),
                'edit_progress':self.edit_progress_head(q).squeeze(-1).sigmoid(),
                'event_hidden':event,
                'event_count_logits':event_count,
                'event_time_logits':self.event_time_head(event),
                'event_operation_logits':self.event_operation_head(event),
                'event_address_logits':event_address,'address_logits':logits,
                'hp_aux':inputs['initial_hp'][:,None]+self.hp_aux(q).squeeze(-1),
                'hidden':q,'candidates':candidates}

    def forward_player(self,inputs):
        """Predict resident state without materializing visual or voxel memory."""
        if not self.cfg.player_branch:raise RuntimeError('player branch is disabled')
        active=inputs['active'].bool();pose=inputs['initial_pose'];b,a=active.shape
        embedding=self.player_item_embedding
        state=torch.cat((inputs['initial_hp'][...,None]/20.,pose[...,3:].sin(),pose[...,3:].cos(),
                         embedding(inputs['held_item']),self.player_type_embedding(inputs['resident_type']),
                         inputs['camera_relative'],inputs['camera_direction']),-1)
        state=self.player_state(state)*active[...,None]
        proposal=propose_trajectory(pose,inputs['actions'],self.cfg.kinematics)
        proposal_features=torch.cat((proposal[...,:3]/6.,proposal[...,3:].sin(),proposal[...,3:].cos()),-1)
        held,held_valid=held_item_trajectory(inputs)
        held=torch.where(held_valid,held,inputs['held_item'][:,None])
        q=(self.player_action(inputs['actions'])+self.player_proposal(proposal_features)+state[:,None]
           +self.player_held_condition(embedding(held))+self.player_time_embedding[None,:,None])
        if self.cfg.player_velocity:
            velocity=inputs.get('initial_velocity')
            if velocity is None:
                velocity=torch.zeros_like(pose[...,:3])
            velocity=velocity/self.cfg.kinematics.distance_per_step
            q=q+self.player_velocity(velocity)[:,None]
        geometry=geometry_padding=None
        if self.cfg.player_voxel:
            if 'player_voxels' in inputs:
                player_voxels=inputs['player_voxels']
                player_known=inputs['player_voxel_known']
                player_xyz=inputs['player_voxel_relative_xyz']
            else:
                center=slice(3,10)
                player_voxels=inputs['voxels'][:,:,center,center,center]
                player_known=inputs['voxel_known'][:,:,center,center,center]
                player_xyz=(inputs['voxel_relative_xyz'].reshape(b,a,13,13,13,3)
                            [:,:,center,center,center].reshape(b,a,7**3,3))
            ids=torch.where(player_known.bool(),player_voxels.long(),
                            self.cfg.num_block_classes)
            side=ids.shape[-1]
            geometry=self.block_embedding(ids).reshape(b*a,side,side,side,self.cfg.width).permute(0,4,1,2,3)
            geometry=(geometry+self.voxel_cnn(geometry)).flatten(2).transpose(1,2)
            xyz=player_xyz.reshape(b*a,side**3,3)
            geometry=geometry+self.coordinate(xyz/6.)
            geometry=torch.cat((geometry,self.player_voxel_null[None,None].expand(b*a,1,-1)),1)
            known=player_known.reshape(b*a,side**3).bool()
            geometry_padding=torch.cat((~known,torch.zeros(b*a,1,dtype=torch.bool,device=known.device)),1)
        for index,block in enumerate(self.player_blocks):
            q=block(q,active)
            if geometry is not None:
                query=self.player_voxel_norm[index](q).transpose(1,2).reshape(b*a,8,self.cfg.width)
                update=self.player_voxel_attn[index](query,geometry,geometry,
                                                     key_padding_mask=geometry_padding,
                                                     need_weights=False)[0]
                q=q+update.reshape(b,a,8,self.cfg.width).transpose(1,2)
        residual=self.pose_head(q);camera=self.camera_head(q)
        predicted=torch.cat((pose[:,None,:,:3]+proposal[...,:3]+residual[...,:3],
                             wrap_angle(proposal[...,3:]+residual[...,3:])), -1)
        velocity=torch.cat((predicted[:,:1,:,:3]-pose[:,None,:,:3],
                            predicted[:,1:,:,:3]-predicted[:,:-1,:,:3]),1)
        direction=nn.functional.normalize(inputs['camera_direction'][:,None]+camera[...,3:],dim=-1)
        output={'held_logits':self.held_head(q),'pose':predicted,'velocity':velocity,'proposal':proposal,
                'pose_residual':residual,'camera_relative':inputs['camera_relative'][:,None]+camera[...,:3],
                'camera_direction':direction,'player_hidden':q}
        if self.cfg.attack_branch:
            memory=q.transpose(1,2).reshape(b*a,8,self.cfg.width)
            query=self.attack_query.weight[None].expand(b*a,-1,-1)
            event=self.attack_decoder(query,memory).reshape(b,a,self.cfg.attack_queries,self.cfg.width)
            resident=q.mean(1)
            target_logits=torch.einsum(
                'baqd,bnd->baqn',self.attack_target_query(event),self.attack_target_key(resident)
            )/self.cfg.width**.5
            target_valid=active[:,None,None,:].expand_as(target_logits).clone()
            target_valid&=~torch.eye(a,dtype=torch.bool,device=q.device)[None,:,None]
            output.update(
                attack_hidden=event,
                attack_count_logits=self.attack_count_head(torch.cat((q.mean(1),event.mean(2)),-1)),
                attack_time_logits=self.attack_time_head(event),
                attack_target_logits=target_logits.masked_fill(~target_valid,-torch.inf),
                attack_damage=self.attack_damage_head(event).squeeze(-1))
        return output

    def payloads(self, output, addresses):
        """Teacher-address payload training is separate from all state predictions."""
        q = output['hidden']; b,t,a,d = q.shape
        candidates = output['candidates'][:,None].expand(-1,t,-1,-1,-1)
        selected = candidates.gather(3,addresses[...,None,None].expand(b,t,a,1,d)).squeeze(3)
        h = self.payload(torch.cat((q,selected),-1))
        return self.block_head(h), self.damage_head(h).squeeze(-1)

    def event_payloads(self, output, addresses):
        """Decode ordered-slot payloads at slot-selected candidate addresses."""
        event=output['event_hidden'];b,a,q,d=event.shape
        candidates=output['candidates'][:,:,:-1]
        selected=candidates[:,:,None].expand(-1,-1,q,-1,-1).gather(
            3,addresses[...,None,None].expand(b,a,q,1,d)).squeeze(3)
        h=self.payload(torch.cat((event,selected),-1))
        return self.block_head(h),self.damage_head(h).squeeze(-1)

    def decode_attacks(self,output):
        """Return calibrated ordered attacks; each slot has exactly one time."""
        logits=output['attack_count_logits'].clone()
        logits[...,0]+=self.cfg.attack_null_logit_bias
        count=logits.argmax(-1);queries=output['attack_time_logits'].shape[-2]
        valid=torch.arange(queries,device=count.device)<count[...,None]
        return {'count':count,'valid':valid,
                'time':output['attack_time_logits'].argmax(-1),
                'target':output['attack_target_logits'].argmax(-1),
                'damage':output['attack_damage']}
