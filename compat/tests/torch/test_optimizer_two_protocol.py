"""RAdam/NAdam scalar eager contracts against the fixed torch oracle."""
import pytest
import torch
import copy

@pytest.mark.parametrize('name', ['RAdam','NAdam'])
def test_defaults_lazy_state_and_none_zero_grad(name):
    p=torch.nn.Parameter(torch.tensor([1.,-2.]))
    opt=getattr(torch.optim,name)([p])
    expected=dict(lr=.001 if name=='RAdam' else .002,betas=(.9,.999),eps=1e-8,weight_decay=0,
                  decoupled_weight_decay=False,foreach=None,maximize=False,capturable=False,differentiable=False)
    if name=='NAdam':expected['momentum_decay']=.004
    assert opt.defaults==expected
    assert not opt.state
    p.grad=None;opt.step();assert not opt.state
    p.grad=torch.zeros_like(p);opt.step()
    assert opt.param_groups[0]['params'][0] is p
    expected_keys={'step','exp_avg','exp_avg_sq'}|({'mu_product'} if name=='NAdam' else set())
    assert set(opt.state[p])==expected_keys
    assert opt.state[p]['step'].item()==1
    for key in {'step'}|({'mu_product'} if name=='NAdam' else set()):
        assert opt.state[p][key].dtype==torch.float32
        assert opt.state[p][key].device.type=='cpu'
    assert p.requires_grad

@pytest.mark.parametrize('name', ['RAdam','NAdam'])
@pytest.mark.parametrize('kwargs',[{'lr':-.1},{'eps':-.1},{'weight_decay':-.1},{'betas':(-.1,.9)},{'betas':(.9,1.)}])
def test_invalid_options(name,kwargs):
    with pytest.raises(ValueError):getattr(torch.optim,name)([torch.nn.Parameter(torch.tensor([1.]))],**kwargs)

def test_invalid_momentum_decay():
    with pytest.raises(ValueError):torch.optim.NAdam([torch.nn.Parameter(torch.tensor([1.]))],momentum_decay=-.1)

@pytest.mark.parametrize('name', ['RAdam','NAdam'])
def test_closure_grad_mode_return_and_parameter_identity(name):
    p=torch.nn.Parameter(torch.tensor([1.,-2.]));opt=getattr(torch.optim,name)([p]);calls=[]
    def closure():
        assert torch.is_grad_enabled()
        opt.zero_grad();loss=(p*p).sum();loss.backward();calls.append(loss);return loss
    with torch.no_grad():result=opt.step(closure)
    assert len(calls)==1 and result is calls[0]
    assert p.requires_grad and opt.param_groups[0]['params'][0] is p
    assert p.grad.tolist()==[2.,-4.]
    assert opt.state[p]['step'].item()==1

@pytest.mark.parametrize('name', ['RAdam','NAdam'])
def test_scalar_state_follows_default_dtype(name):
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        p=torch.nn.Parameter(torch.tensor([1.]));opt=getattr(torch.optim,name)([p])
        p.grad=torch.ones_like(p);opt.step()
        assert opt.state[p]['step'].dtype==torch.float64
        if name=='NAdam':assert opt.state[p]['mu_product'].dtype==torch.float64
    finally:torch.set_default_dtype(old)

def test_nadam_double_restore_casts_mu_product_without_changing_saved_state():
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        p=torch.nn.Parameter(torch.tensor([.123456789012345],dtype=torch.float64))
        opt=torch.optim.NAdam([p],foreach=False,momentum_decay=.0125)
        for _ in range(3):
            p.grad=torch.tensor([.314159265358979],dtype=torch.float64);opt.step()
        saved=copy.deepcopy(opt.state_dict())
        product=saved['state'][0]['mu_product'].item()
        restored=torch.nn.Parameter(torch.from_numpy(p.detach().cpu().numpy().copy()))
        other=torch.optim.NAdam([restored],foreach=False,momentum_decay=.0125)
        other.load_state_dict(saved)
        assert torch.get_default_dtype()==torch.float32
        assert torch.equal(p,restored)
        assert saved['state'][0]['mu_product'].dtype==torch.float32
        assert opt.state[p]['mu_product'].dtype==torch.float32
        assert other.state[restored]['mu_product'].dtype==torch.float64
        assert saved['state'][0]['mu_product'].item()==product
        assert opt.state[p]['mu_product'].item()==product
        assert other.state[restored]['mu_product'].item()==product
    finally:torch.set_default_dtype(old)
