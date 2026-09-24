import torch
from torch import nn
from plot.models.state_policy_large import LargeStatePolicy,LargeStatePolicyArgs


def inputs(profile='zombie_melee',batch=1):
    result=dict(resident_state=torch.randn(batch,8,3,18),resident_actions=torch.zeros(batch,8,3,23),
        resident_valid=torch.ones(batch,8,3,dtype=torch.bool),held_item=torch.zeros(batch,8,3,dtype=torch.long),
        resident_type=torch.zeros(batch,8,3,dtype=torch.long),history_valid=torch.ones(batch,8,dtype=torch.bool),
        target_agent=torch.zeros(batch,dtype=torch.long),voxel_classes=torch.zeros(batch,48,48,48,dtype=torch.long),
        voxel_known=torch.ones(batch,48,48,48,dtype=torch.bool),
        raster_camera=torch.tensor([[0.,1.,0.,0.,0.,-1.,0.,0.,0.,1.2]]).repeat(batch,1))
    if profile=='language_builder':
        for name in ('shared','current'):
            result[name+'_text']=torch.randn(batch,5,12)
            result[name+'_text_mask']=torch.tensor([[True,True,False,False,False]]).repeat(batch,1)
    return result


class StubGeometry(nn.Module):
    def __init__(self):super().__init__();self.value=nn.Parameter(torch.randn(1,4,32))
    def forward(self,inputs):return self.value.expand(len(inputs['target_agent']),-1,-1)


def tiny(profile):
    model=LargeStatePolicy(LargeStatePolicyArgs(3,4,profile,hidden=32,heads=4,depth=2,
        text_hidden_size=12,gradient_checkpointing=True,image_h=4,image_w=4))
    model.geometry=StubGeometry()
    return model


def test_large_permutation_and_masked_history():
    torch.manual_seed(12);model=tiny('zombie_melee').eval();data=inputs()
    data['history_valid'][:,:3]=False;data['resident_valid'][:,:3]=False
    first=model(data);changed={k:v.clone() for k,v in data.items()}
    permutation=torch.tensor([2,0,1])
    for key in ('resident_state','resident_actions','resident_valid','held_item','resident_type'):
        changed[key]=changed[key][:,:,permutation]
    changed['target_agent'][:]=1
    changed['resident_state'][:,:3]=100
    changed['resident_actions'][:,:3]=100
    for key,value in first.items():torch.testing.assert_close(value,model(changed)[key],atol=3e-6,rtol=3e-5)


def test_large_text_gradient_padding_and_roundtrip(tmp_path):
    torch.manual_seed(13);model=tiny('language_builder');data=inputs('language_builder')
    first=model(data);loss,_=model.loss(first,torch.zeros(1,8,23),positive_weight=4,key_weight=2)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.text_projection.weight.grad.abs().sum()>0
    assert model.decode(first).shape==(1,8,23)
    model.eval();first=model(data);changed={k:v.clone() for k,v in data.items()}
    changed['current_text'][:,2:]=100
    for key,value in first.items():torch.testing.assert_close(value,model(changed)[key])
    changed['current_text'][:,:2]*=-1
    assert any(not torch.allclose(value,model(changed)[key]) for key,value in first.items())
    for name in ('shared','current'):changed[name+'_text_mask'].zero_()
    assert all(torch.isfinite(value).all() for value in model(changed).values())
    torch.save(model.state_dict(),tmp_path/'weights.pt')
    restored=tiny('language_builder').eval();restored.load_state_dict(torch.load(tmp_path/'weights.pt',weights_only=True))
    for key,value in first.items():torch.testing.assert_close(value,restored(data)[key])
