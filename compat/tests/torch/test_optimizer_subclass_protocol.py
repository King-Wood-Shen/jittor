"""CUDA contracts for the shared Torch optimizer subclass protocol."""
import copy
import numpy as np
import jittor as jt
import pytest
import torch
from _helpers.checkpoint_cuda import accelerator, snapshot


class _MomentumOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters, defaults=None, *, keyword=False):
        defaults = dict(lr=.1, momentum=.5) if defaults is None else defaults
        if keyword:
            super().__init__(parameters, defaults=defaults)
        else:
            super().__init__(parameters, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state['velocity'] = torch.zeros_like(parameter)
                    state['step'] = 0
                    state['metadata'] = {'algorithm': 'custom'}
                state['velocity'].mul_(group['momentum']).add_(parameter.grad)
                state['step'] += 1
                parameter.add_(state['velocity'], alpha=-group['lr'])


@pytest.mark.parametrize('keyword', [False, True])
def test_custom_defaults_scheduler_and_real_momentum_updates(keyword):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
    defaults = dict(lr=.1, momentum=.5)
    optimizer = _MomentumOptimizer((p for p in [parameter]), defaults, keyword=keyword)
    assert optimizer.defaults is defaults
    assert optimizer.param_groups[0]['lr'] == .1
    assert optimizer.param_groups[0]['momentum'] == .5
    assert optimizer.state is optimizer.state
    assert optimizer.state.get(parameter) is None
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1. / (step + 1))
    (parameter.square().sum()).backward()
    optimizer.step()
    np.testing.assert_allclose(snapshot(parameter), [.8, 1.6], rtol=1e-6)
    scheduler.step()
    optimizer.zero_grad()
    (parameter.square().sum()).backward()
    optimizer.step()
    np.testing.assert_allclose(snapshot(parameter), [.67, 1.34], rtol=1e-6)
    np.testing.assert_allclose(snapshot(optimizer.state[parameter]['velocity']), [2.6, 5.2], rtol=1e-6)
    assert optimizer.state[parameter]['step'] == 2
    assert optimizer.state[parameter]['metadata'] == {'algorithm': 'custom'}


def test_custom_group_overrides_and_late_parameter_group():
    first = torch.nn.Parameter(torch.tensor([1.], device='cuda'))
    second = torch.nn.Parameter(torch.tensor([2.], device='cuda'))
    optimizer = _MomentumOptimizer([{'params': first, 'lr': .2}])
    optimizer.add_param_group({'params': (p for p in [second]), 'momentum': .25})
    assert optimizer.param_groups[0]['lr'] == .2
    assert optimizer.param_groups[0]['momentum'] == .5
    assert optimizer.param_groups[1]['lr'] == .1
    assert optimizer.param_groups[1]['momentum'] == .25
    (first.square().sum() + second.square().sum()).backward()
    optimizer.step()
    np.testing.assert_allclose(snapshot(first), [.6], rtol=1e-6)
    np.testing.assert_allclose(snapshot(second), [1.6], rtol=1e-6)
    with pytest.raises(ValueError, match='more than one'):
        optimizer.add_param_group({'params': [first]})


def test_foreign_adamw_named_class_retains_arbitrary_state_and_resume():
    class AdamW(_MomentumOptimizer):
        pass
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
    optimizer = AdamW([parameter])
    parameter.square().sum().backward()
    optimizer.step()
    optimizer.state['global_metadata'] = {'version': 7}
    saved = copy.deepcopy(optimizer.state_dict())
    assert 'velocity' in saved['state'][0]
    assert 'exp_avg' not in saved['state'][0]
    assert 'betas' not in saved['param_groups'][0]
    assert 'grads' not in saved['param_groups'][0]
    restored_parameter = torch.nn.Parameter(torch.tensor(snapshot(parameter), device='cuda'))
    restored = AdamW([restored_parameter], dict(lr=.7, momentum=.9))
    restored.load_state_dict(saved)
    assert restored.defaults == dict(lr=.7, momentum=.9)
    assert restored.param_groups[0]['lr'] == .1
    assert restored.state['global_metadata'] == {'version': 7}
    assert restored.state[restored_parameter]['metadata'] == {'algorithm': 'custom'}
    restored_parameter.square().sum().backward()
    restored.step()
    np.testing.assert_allclose(snapshot(restored_parameter), [.54, 1.08], rtol=1e-6)
    assert restored.state[restored_parameter]['step'] == 2
    np.testing.assert_allclose(snapshot(saved['state'][0]['velocity']), [2., 4.], rtol=1e-6)
    before_groups, before_state = restored.param_groups, restored.state
    invalid = copy.deepcopy(saved)
    invalid['param_groups'][0]['params'].append(9)
    with pytest.raises(ValueError, match='size'):
        restored.load_state_dict(invalid)
    assert restored.param_groups is before_groups and restored.state is before_state


@pytest.mark.parametrize('name', ['SGD', 'AdamW'])
def test_native_algorithm_numeric_initializer_and_state_remain_valid(name):
    parameter = torch.nn.Parameter(torch.tensor([1., 2.], device='cuda'))
    optimizer = getattr(torch.optim, name)([parameter], lr=.1)
    parameter.square().sum().backward()
    optimizer.step()
    if name == 'SGD':
        expected = [.8, 1.6]
    else:
        assert optimizer.defaults['weight_decay'] == .01
        expected = np.array([1., 2.]) * (1 - .1 * .01) - .1
    np.testing.assert_allclose(snapshot(parameter), expected, rtol=1e-6)
    assert isinstance(optimizer.param_groups[0]['lr'], float)
    if name == 'AdamW':
        assert {'exp_avg', 'exp_avg_sq', 'step'} <= set(optimizer.state[parameter])
