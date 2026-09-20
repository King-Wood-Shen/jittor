"""The existing scalar optimizers retain explicit foreach=None/False metadata."""
import copy
import pytest
import torch

@pytest.mark.parametrize('name',['SGD','AdamW'])
@pytest.mark.parametrize('default',[None,False])
@pytest.mark.parametrize('override',[None,False])
def test_foreach_defaults_groups_add_and_restore(name,default,override):
    cls=getattr(torch.optim,name)
    params=[torch.nn.Parameter(torch.tensor([float(i+1)])) for i in range(4)]
    opt=cls([{'params':[params[0]]},{'params':[params[1]],'foreach':override}],lr=.03,foreach=default)
    assert opt.defaults['foreach'] is default
    assert [g['foreach'] for g in opt.param_groups]==[default,override]
    opt.add_param_group({'params':[params[2]]})
    opt.add_param_group({'params':[params[3]],'foreach':override})
    expected=[default,override,default,override]
    assert [g['foreach'] for g in opt.param_groups]==expected
    for p in params:p.grad=torch.ones_like(p)
    opt.step()
    saved=copy.deepcopy(opt.state_dict())
    assert [g['foreach'] for g in saved['param_groups']]==expected
    new_default=False if default is None else None
    restored=cls([{'params':[torch.nn.Parameter(p.detach().clone())]} for p in params],lr=.4,foreach=new_default)
    restored.load_state_dict(saved)
    assert [g['foreach'] for g in restored.param_groups]==expected
    assert [g['foreach'] for g in restored.state_dict()['param_groups']]==expected
    assert [g['foreach'] for g in saved['param_groups']]==expected
    assert restored.defaults['foreach'] is new_default
    restored.add_param_group({'params':[torch.nn.Parameter(torch.tensor([5.]))]})
    assert restored.param_groups[-1]['foreach'] is new_default

@pytest.mark.parametrize('name',['SGD','AdamW'])
def test_omitted_foreach_default_is_none(name):
    p=torch.nn.Parameter(torch.tensor([1.]))
    opt=getattr(torch.optim,name)([p],lr=.03)
    assert opt.defaults['foreach'] is None
    assert opt.param_groups[0]['foreach'] is None
