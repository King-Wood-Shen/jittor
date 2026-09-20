"""AdamW noncapturable step counter ownership across state boundaries."""
import copy
import pytest
import torch


def resident(t, key):
    expected = "cpu" if key.endswith("/step") else "cuda"
    assert t.device.type == expected
    if hasattr(t, "location"):
        t.sync()
        assert t.location() == ("cpu" if expected == "cpu" else "device")


@pytest.mark.parametrize("default", [torch.float32, torch.float64])
@pytest.mark.parametrize("param_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_adamw_step_state_ownership(default, param_dtype, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    old = torch.get_default_dtype()
    r = {}
    try:
        torch.set_default_dtype(default)
        p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=param_dtype,device=device))
        o=torch.optim.AdamW([p],lr=.01,weight_decay=.01,foreach=False)
        assert len(o.state)==0 and o.state_dict()['state']=={}
        p.grad=torch.ones_like(p);o.step()
        s=o.state[p]['step'];d=o.state_dict()['state'][0]['step']
        r.update(step_type=type(s).__name__,step_dtype=str(getattr(s,'dtype',None)),step_device=str(getattr(s,'device',None)),export_same=s is d)
        assert torch.is_tensor(s) and s.dtype==default and s.device.type=='cpu' and s.ndim==0
        assert s is d and s is o.state[p]['step']
        if device=='cuda':
         resident(s,'public/step');resident(p,'parameter')
         resident(o.state[p]['exp_avg'],'moment')
        torch.set_default_dtype(torch.float64 if default==torch.float32 else torch.float32)
        o.step();assert s.item()==2 and s.dtype==default
        p.grad=None;o.step();assert s.item()==2
        saved=copy.deepcopy(o.state_dict());ss=saved['state'][0]['step']
        q=torch.nn.Parameter(p.detach().clone());n=torch.optim.AdamW([q],lr=.01,weight_decay=.01,foreach=False)
        n.load_state_dict(saved);t=n.state[q]['step']
        assert t is ss and t is n.state_dict()['state'][0]['step']
        assert t.dtype==default and t.device.type=='cpu'
        q.grad=torch.ones_like(q);n.step()
        assert t.item()==3 and ss.item()==3
        if device=='cuda':
         resident(t,'loaded/step');resident(q,'loaded_parameter')
         resident(n.state[q]['exp_avg'],'loaded_moment')
        torch.set_default_dtype(default)
        u=torch.nn.Parameter(torch.tensor([.7,-.4],dtype=param_dtype,device=device))
        v=torch.nn.Parameter(torch.tensor([.2,.9],dtype=param_dtype,device=device))
        z=torch.optim.AdamW([{'params':[u],'lr':.02},{'params':[v],'lr':.03}],weight_decay=.01,foreach=False)
        z.step();assert len(z.state)==0
        u.grad=torch.ones_like(u);z.step()
        first=z.state[u]['step'];assert first.dtype==default and v not in z.state
        torch.set_default_dtype(torch.float64 if default==torch.float32 else torch.float32)
        v.grad=torch.ones_like(v);z.step()
        assert first.item()==2 and first.dtype==default
        assert z.state[v]['step'].dtype==torch.get_default_dtype()
        first.fill_(5);z.step();assert first.item()==6
        replacement=torch.tensor(10.,dtype=default,device='cpu')
        z.state[u]['step']=replacement
        assert z.state[u]['step'] is replacement
        z.step();assert replacement.item()==11 and first.item()==6
        r['mutation_parameters']=[u.detach().cpu().numpy().tolist(),v.detach().cpu().numpy().tolist()]
        if device=='cuda':
         resident(u,'mutation_parameter');resident(v,'mutation_parameter2')
         resident(replacement,'mutation/step')
    finally:
        torch.set_default_dtype(old)
