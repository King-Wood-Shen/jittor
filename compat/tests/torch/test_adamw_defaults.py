import copy
import pytest
import torch

@pytest.mark.parametrize("args,kwargs,expected", [
 ((), {}, (.001,(.9,.999),1e-8,.01)),
 ((), {"lr":.02}, (.02,(.9,.999),1e-8,.01)),
 ((), {"weight_decay":0.}, (.001,(.9,.999),1e-8,0.)),
 ((), {"weight_decay":.01}, (.001,(.9,.999),1e-8,.01)),
 ((.02,), {}, (.02,(.9,.999),1e-8,.01)),
 ((.02,(.8,.95)), {}, (.02,(.8,.95),1e-8,.01)),
 ((.02,(.8,.95),1e-7), {}, (.02,(.8,.95),1e-7,.01)),
 ((.02,(.8,.95),1e-7,0.), {}, (.02,(.8,.95),1e-7,0.)),
])
def test_adamw_constructor_defaults_and_positions(args,kwargs,expected):
 p=torch.nn.Parameter(torch.tensor([1.]))
 o=torch.optim.AdamW([p],*args,foreach=False,**kwargs)
 lr,betas,eps,wd=expected
 for group in [o.defaults,o.param_groups[0]]:
  assert group["lr"]==lr and group["betas"]==betas
  assert group["eps"]==eps and group["weight_decay"]==wd
 p.grad=torch.tensor([.3]);o.step()
 assert abs(p.item()-(1*(1-lr*wd)-lr*.3/(.3+eps)))<2e-7

def test_adamw_group_override_add_load_defaults():
 params=[torch.nn.Parameter(torch.tensor([1.])) for _ in range(4)]
 o=torch.optim.AdamW([{"params":[params[0]],"weight_decay":0.},{"params":[params[1]]}],weight_decay=.04,foreach=False)
 o.add_param_group({"params":[params[2]]})
 o.add_param_group({"params":[params[3]],"weight_decay":.02})
 assert [g["weight_decay"] for g in o.param_groups]==[0.,.04,.04,.02]
 for p in params:p.grad=torch.ones_like(p)
 o.step()
 saved=copy.deepcopy(o.state_dict())
 other=[torch.nn.Parameter(p.detach().clone()) for p in params]
 n=torch.optim.AdamW([{"params":[p]} for p in other],weight_decay=.07,foreach=False)
 n.load_state_dict(saved)
 assert n.defaults["weight_decay"]==.07
 assert [g["weight_decay"] for g in n.param_groups]==[0.,.04,.04,.02]
 for p in other:p.grad=torch.ones_like(p)
 o.step();n.step()
 for p,q in zip(params,other):assert abs(p.item()-q.item())<2e-7


def test_adamw_duplicate_position_keyword_rejected():
 p=torch.nn.Parameter(torch.tensor([1.]))
 with pytest.raises(TypeError):
  torch.optim.AdamW([p],.02,lr=.03)
 with pytest.raises(TypeError):
  torch.optim.AdamW([p],.02,(.8,.95),betas=(.9,.99))

def test_adamw_sixth_position_is_amsgrad_false():
 p=torch.nn.Parameter(torch.tensor([1.]))
 o=torch.optim.AdamW([p],.02,(.8,.95),1e-7,0.,False)
 assert o.defaults["weight_decay"]==0. and o.defaults["amsgrad"] is False
