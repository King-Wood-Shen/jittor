import importlib
import pytest
import torch


def test_optimizer_mapping_defaults_are_not_a_learning_rate():
    opt = torch.optim.Optimizer([torch.nn.Parameter(torch.tensor([1.]))], {'lr': .2, 'rate': 7})
    assert opt.defaults['lr'] == .2
    assert opt.param_groups[0]['lr'] == .2
    assert opt.param_groups[0]['rate'] == 7


def test_required_import_repr_and_identity():
    from torch.optim.optimizer import required
    assert repr(required) == '<required parameter>'
    assert required is importlib.import_module('torch.optim.optimizer').required


def test_required_missing_at_construction():
    from torch.optim.optimizer import required
    p = torch.nn.Parameter(torch.tensor([1.]))
    with pytest.raises(ValueError, match='required optimization parameter rate'):
        torch.optim.Optimizer([p], {'rate': required})


def test_required_present_and_defaults_retained():
    from torch.optim.optimizer import required
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = torch.optim.Optimizer([{'params': [p], 'rate': .2}], {'rate': required, 'other': 7})
    assert opt.defaults['rate'] is required
    assert opt.param_groups[0]['rate'] == .2
    assert opt.param_groups[0]['other'] == 7


def test_required_missing_new_group_does_not_append():
    from torch.optim.optimizer import required
    p, q = [torch.nn.Parameter(torch.tensor([float(x)])) for x in [1, 2]]
    opt = torch.optim.Optimizer([{'params': [p], 'rate': .2}], {'rate': required})
    with pytest.raises(ValueError, match='required optimization parameter rate'):
        opt.add_param_group({'params': [q]})
    assert len(opt.param_groups) == 1


def test_required_supplied_new_group():
    from torch.optim.optimizer import required
    p, q = [torch.nn.Parameter(torch.tensor([float(x)])) for x in [1, 2]]
    opt = torch.optim.Optimizer([{'params': [p], 'rate': .2}], {'rate': required, 'other': 7})
    opt.add_param_group({'params': [q], 'rate': .3})
    assert len(opt.param_groups) == 2
    assert opt.param_groups[1]['rate'] == .3
    assert opt.param_groups[1]['other'] == 7


def test_required_check_uses_identity():
    from torch.optim.optimizer import required
    other = type(required)()
    assert other is not required and repr(other) == repr(required)
    opt = torch.optim.Optimizer([torch.nn.Parameter(torch.tensor([1.]))], {'rate': other})
    assert opt.param_groups[0]['rate'] is other


def test_defaults_and_groups_keep_reference_and_overrides():
    p, q = [torch.nn.Parameter(torch.tensor([float(x)])) for x in [1, 2]]
    shared = [1]
    defaults = {'rate': .2, 'options': shared}
    group = {'params': [p], 'rate': .3}
    opt = torch.optim.Optimizer([group], defaults)
    assert opt.defaults is defaults
    assert opt.param_groups[0] is group
    assert defaults == {'rate': .2, 'options': [1]}
    assert set(group) == {'params', 'rate', 'options'}
    assert group['options'] is shared
    added = {'params': [q]}
    opt.add_param_group(added)
    assert opt.param_groups[1] is added and added['options'] is shared
    assert added['rate'] == .2
    shared.append(2)
    assert group['options'] == added['options'] == [1, 2]


def test_failed_group_keeps_oracle_partial_default_mutation():
    from torch.optim.optimizer import required
    p, q = [torch.nn.Parameter(torch.tensor([float(x)])) for x in [1, 2]]
    defaults = {'optional': 7, 'rate': required}
    opt = torch.optim.Optimizer([{'params': [p], 'rate': .2}], defaults)
    group = {'params': (q,)}
    with pytest.raises(ValueError, match='required optimization parameter rate'):
        opt.add_param_group(group)
    assert group['params'] == [q]
    assert group['optional'] == 7 and 'rate' not in group
    assert len(opt.param_groups) == 1
    assert defaults['rate'] is required


def test_generic_state_dict_does_not_inject_learning_rate():
    p, q = [torch.nn.Parameter(torch.tensor([float(x)])) for x in [1, 2]]
    opt = torch.optim.Optimizer([{'params': [p]}, {'params': [q], 'lr': .3}], {'rate': 7})
    assert opt.state_dict() == {'state': {}, 'param_groups': [
        {'params': [0], 'rate': 7}, {'params': [1], 'rate': 7, 'lr': .3}]}


def test_native_optimizer_isolation():
    if not hasattr(torch, '_torch_compat_install_context'):
        pytest.skip('native isolation applies to shim')
    import jittor as jt
    assert jt.optim.Optimizer is not torch.optim.Optimizer
    opt = jt.optim.Optimizer([jt.array([1.])], .2)
    assert opt.lr == .2
    assert 'rate' not in opt.defaults
