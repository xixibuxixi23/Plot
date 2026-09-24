"""Large state-only M4: M3 depth-stack geometry and deep actor/text cross-attention."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .structured_action import StructuredActionHead
from .renderer_backbone.embeddings import DepthPatchEmbedder, PixelPatchEmbedder
from .renderer_backbone.camera_util import camera_params_to_matrices


@dataclass(frozen=True)
class LargeStatePolicyArgs:
    num_block_classes: int
    item_vocab_size: int
    profile: str
    hidden: int = 768
    heads: int = 12
    depth: int = 12
    text_hidden_size: int = 768
    gradient_checkpointing: bool = True
    image_h: int = 36
    image_w: int = 64
    voxel_channels: int = 32


class M3StateGeometry(nn.Module):
    """Same M3 class embedding, camera conversion, 192-layer raster and depth stack.

    Reuses M3 implementations, not M3 weights. No image latents or renderer DiT.
    The final 2x2 patch projection consumes geometry channels only.
    """
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.voxel_embedder = nn.Embedding(cfg.num_block_classes+1,cfg.voxel_channels)
        self.depth_embedder = DepthPatchEmbedder(num_layers=192,patch_size=6,stride=4,
            in_chans=cfg.voxel_channels,out_chans=16,use_depth_pos_enc=True,num_freqs=8)
        self.patch = PixelPatchEmbedder(cfg.image_h,cfg.image_w,2,47*16,cfg.hidden,flatten=True)
        self.position = nn.Parameter(torch.randn(1,(cfg.image_h//2)*(cfg.image_w//2),cfg.hidden)*.02)
        self._rasterizer = None

    def raster(self, inputs):
        if not inputs['voxel_classes'].is_cuda: raise ValueError('M3 rasterizer requires CUDA')
        if self._rasterizer is None:
            from .renderer_backbone.voxel_rasterizer import VoxelMeshRasterizer
            self._rasterizer = VoxelMeshRasterizer(dim=48,height=self.cfg.image_h,
                width=self.cfg.image_w,max_layers=192,device=inputs['voxel_classes'].device)
        with torch.autocast('cuda',enabled=False):
            ids = torch.where(inputs['voxel_known'].bool(),inputs['voxel_classes'].long(),self.cfg.num_block_classes)
            features = self.voxel_embedder(ids).flatten(1,3).float()
            view,proj = camera_params_to_matrices(inputs['raster_camera'].float()[:,None],
                image_width=self.cfg.image_w,image_height=self.cfg.image_h)
            raster,depth = self._rasterizer.rasterize(features,view[:,0],proj[:,0],features.new_zeros(self.cfg.voxel_channels))
        return raster.permute(1,2,3,0,4), depth.permute(1,2,3,0)

    def forward(self, inputs):
        features,depth = self.raster(inputs)
        projected = self.depth_embedder(features,depth)
        b,h,w,l,c = projected.shape
        return self.patch(projected.permute(0,3,4,1,2).reshape(b,l*c,h,w)) + self.position


class Attention(nn.Module):
    def __init__(self,d,heads):
        super().__init__(); self.heads=heads; self.hdim=d//heads
        self.q=nn.Linear(d,d); self.kv=nn.Linear(d,d*2); self.out=nn.Linear(d,d)
    def forward(self,query,memory,valid):
        def split(x): return x.reshape(x.shape[0],x.shape[1],self.heads,self.hdim).transpose(1,2)
        q=split(self.q(query)); k,v=[split(value) for value in self.kv(memory).chunk(2,-1)]
        q=F.rms_norm(q,(self.hdim,)); k=F.rms_norm(k,(self.hdim,))
        value=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:],dropout_p=0.)
        return self.out(value.transpose(1,2).flatten(2))


class PolicyBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.hidden
        self.self_norm=nn.LayerNorm(d); self.self_attn=Attention(d,cfg.heads)
        self.actor_norm=nn.LayerNorm(d); self.actor_attn=Attention(d,cfg.heads)
        self.has_text=cfg.profile=='language_builder'
        if self.has_text:
            self.text_norm=nn.LayerNorm(d); self.text_attn=Attention(d,cfg.heads)
        self.ff_norm=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d))
    def forward(self,x,valid,actors,actor_valid,text,text_valid):
        value=self.self_norm(x); x=x+self.self_attn(value,value,valid)
        x=x+self.actor_attn(self.actor_norm(x),actors,actor_valid)
        if self.has_text: x=x+self.text_attn(self.text_norm(x),text,text_valid)
        return x+self.ff(self.ff_norm(x))


class LargeStatePolicy(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg=cfg; d=cfg.hidden
        if cfg.profile not in ('zombie_melee','skeleton_swordsman','language_builder'):
            raise ValueError('unsupported large policy profile')
        self.geometry=M3StateGeometry(cfg)
        self.item=nn.Embedding(cfg.item_vocab_size,64); self.kind=nn.Embedding(4,16)
        self.resident=nn.Sequential(nn.Linear(18+64+16+23,d),nn.GELU(),nn.Linear(d,d),nn.LayerNorm(d))
        self.history_position=nn.Parameter(torch.randn(1,8,1,d)*.02)
        self.self_role=nn.Parameter(torch.randn(d)*.02)
        self.actor_null=nn.Parameter(torch.zeros(1,1,d))
        self.actor_memory_norm=nn.LayerNorm(d)
        self.query=nn.Parameter(torch.randn(1,8,d)*.02)
        if cfg.profile=='language_builder':
            self.text_projection=nn.Linear(cfg.text_hidden_size,d)
            self.text_role=nn.Parameter(torch.randn(2,1,d)*.02)
            self.text_null=nn.Parameter(torch.zeros(1,1,d))
            self.text_memory_norm=nn.LayerNorm(d)
        self.blocks=nn.ModuleList([PolicyBlock(cfg) for _ in range(cfg.depth)])
        self.norm=nn.LayerNorm(d)
        self.head=StructuredActionHead(d,1)
        self.loss_head=StructuredActionHead(d,8)
        for name in ('keys','hotbar','mouse_x','mouse_y'): delattr(self.loss_head,name)

    def forward(self,inputs):
        b,t,a,_=inputs['resident_state'].shape
        if t!=8 or not inputs['history_valid'][:,-1].all(): raise ValueError('invalid history')
        target=inputs['target_agent'].long(); device=target.device
        valid=inputs['resident_valid'].bool() & inputs['history_valid'].bool()[:,:,None]
        if not valid[torch.arange(b,device=device),-1,target].all(): raise ValueError('missing target')
        residents=self.resident(torch.cat((inputs['resident_state'],self.item(inputs['held_item'].long()),
            self.kind(inputs['resident_type'].long()),inputs['resident_actions']),-1))+self.history_position
        own=residents[torch.arange(b,device=device),:,target]+self.self_role
        others=valid & ~F.one_hot(target,a).bool()[:,None,:]
        actors=torch.cat((self.actor_null.expand(b,-1,-1),residents.flatten(1,2)),1)
        actors=self.actor_memory_norm(actors)
        actor_valid=torch.cat((torch.ones(b,1,device=device,dtype=torch.bool),others.flatten(1,2)),1)
        geometry=self.geometry(inputs)
        queries=self.query.expand(b,-1,-1)+own[:,-1:]
        x=torch.cat((geometry,own,queries),1)
        xvalid=torch.cat((torch.ones(b,geometry.shape[1],device=device,dtype=torch.bool),
            inputs['history_valid'].bool(),torch.ones(b,8,device=device,dtype=torch.bool)),1)
        text=x.new_zeros(b,1,self.cfg.hidden); text_valid=torch.ones(b,1,device=device,dtype=torch.bool)
        if self.cfg.profile=='language_builder':
            texts=[self.text_null.expand(b,-1,-1)]; masks=[text_valid]
            for role,name in enumerate(('shared','current')):
                texts.append(self.text_projection(inputs[name+'_text'])+self.text_role[role])
                masks.append(inputs[name+'_text_mask'].bool())
            text=self.text_memory_norm(torch.cat(texts,1)); text_valid=torch.cat(masks,1)
        for block in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x=checkpoint(block,x,xvalid,actors,actor_valid,text,text_valid,use_reentrant=False)
            else: x=block(x,xvalid,actors,actor_valid,text,text_valid)
        return {key:value.squeeze(-2) for key,value in self.head(self.norm(x[:,-8:])).items()}

    def loss(self,logits,actions,valid_mask=None,positive_weight=1.,key_weight=1.,horizon_weights=None):
        _,parts=self.loss_head.loss(logits,actions,valid_mask)
        if positive_weight!=1:
            weights=logits['keys'].new_ones(10); weights[7]=positive_weight
            if self.cfg.profile=='language_builder': weights[8]=positive_weight
            targets=self.loss_head.targets(actions)
            parts['keys']=F.binary_cross_entropy_with_logits(logits['keys'],targets['keys'],
                pos_weight=weights,reduction='none').mean(-1)
        combined=key_weight*parts['keys']+parts['hotbar']+parts['mouse_x']+parts['mouse_y']
        weights=torch.ones_like(combined) if valid_mask is None else valid_mask.to(combined)
        if horizon_weights is not None: weights=weights*horizon_weights.to(combined)[None]
        per_sample=(combined*weights).sum(-1)/weights.sum(-1).clamp_min(1)
        return per_sample.mean(),parts

    def decode(self,logits,key_threshold=.5): return self.loss_head.decode(logits,key_threshold)
