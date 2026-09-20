"""Constructor, state timing and closure contracts pinned to torch 2.5.1."""
import pytest
import torch


@pytest.mark.parametrize('name', ['Adagrad', 'Adadelta', 'Adamax'])
def test_default_options_and_initial_state(name):
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    opt = getattr(torch.optim, name)([p])
    common = {'weight_decay': 0, 'foreach': None, 'maximize': False, 'differentiable': False}
    extra = {
        'Adagrad': {'lr': .01, 'lr_decay': 0, 'eps': 1e-10, 'initial_accumulator_value': 0, 'fused': None},
        'Adadelta': {'lr': 1., 'rho': .9, 'eps': 1e-6, 'capturable': False},
        'Adamax': {'lr': .002, 'betas': (.9, .999), 'eps': 1e-8, 'capturable': False},
    }
    assert opt.defaults == dict(common, **extra[name])
    assert opt.param_groups[0]['params'][0] is p
    assert len(opt.state) == (1 if name == 'Adagrad' else 0)
    if name == 'Adagrad':
        assert opt.state[p]['step'].item() == 0
        assert opt.state[p]['step'].dtype == torch.float32
        assert opt.state[p]['step'].device.type == 'cpu'
        assert torch.equal(opt.state[p]['sum'], torch.zeros_like(p))
    p.grad = None
    opt.step()
    assert len(opt.state) == (1 if name == 'Adagrad' else 0)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert opt.state[p]['step'].item() == 1
    assert opt.state[p]['step'].dtype == torch.float32
    assert opt.param_groups[0]['params'][0] is p


@pytest.mark.parametrize('name', ['Adagrad', 'Adadelta', 'Adamax'])
@pytest.mark.parametrize('keyword', ['lr', 'eps', 'weight_decay'])
def test_negative_options_raise(name, keyword):
    with pytest.raises(ValueError):
        getattr(torch.optim, name)([torch.nn.Parameter(torch.tensor([1.]))], **{keyword: -.1})


@pytest.mark.parametrize('name,options', [
    ('Adagrad', {'lr_decay': -.1}), ('Adagrad', {'initial_accumulator_value': -.1}),
    ('Adadelta', {'rho': -.1}), ('Adadelta', {'rho': 1.1}),
    ('Adamax', {'betas': (-.1, .9)}), ('Adamax', {'betas': (.9, 1.)}),
])
def test_algorithm_options_raise(name, options):
    with pytest.raises(ValueError):
        getattr(torch.optim, name)([torch.nn.Parameter(torch.tensor([1.]))], **options)


@pytest.mark.parametrize('name', ['Adagrad', 'Adadelta', 'Adamax'])
def test_closure_enabled_grad_and_returns_same_loss(name):
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    opt = getattr(torch.optim, name)([p])
    calls = []
    def closure():
        assert torch.is_grad_enabled()
        opt.zero_grad()
        loss = (p * p).sum()
        loss.backward()
        calls.append(loss)
        return loss
    with torch.no_grad():
        loss = opt.step(closure)
    assert len(calls) == 1 and loss is calls[0]
    assert p.requires_grad
    assert opt.state[p]['step'].item() == 1
    assert p.grad.tolist() == [2., -4.]


@pytest.mark.parametrize('name', ['Adagrad', 'Adadelta', 'Adamax'])
def test_step_uses_float64_scalar_when_default_dtype_is_float64(name):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        p = torch.nn.Parameter(torch.tensor([1.]))
        opt = getattr(torch.optim, name)([p])
        p.grad = torch.ones_like(p)
        opt.step()
        assert opt.state[p]['step'].dtype == torch.float64
    finally:
        torch.set_default_dtype(old)
