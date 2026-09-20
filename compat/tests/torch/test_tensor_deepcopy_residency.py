"""Tensor deepcopy preserves source placement and public copy protocols."""
import copy
import numpy as np
import pytest
import torch

@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and (not torch.cuda.is_available()):
        pytest.skip('CUDA unavailable')
    return request.param

def resident(t, device):
    assert t.device.type == device
    if hasattr(t, 'location'):
        t.sync()
        assert t.location() == ('device' if device == 'cuda' else 'cpu')
        import jittor as jt
        backend, index = jt.core.dispatch_context([t])
        assert backend == device
        if device == 'cuda':
            assert index == 0

@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('kind', ['tensor', 'parameter'])
def test_deepcopy_preserves_source_and_protocol(device, dtype, requires_grad, kind):
    expected = np.array([[0.25, -0.5], [1.25, 2.0]])
    t = torch.tensor(expected, dtype=dtype, device=device, requires_grad=requires_grad)
    if kind == 'parameter':
        t = torch.nn.Parameter(t, requires_grad=requires_grad)
    else:
        t.custom = {'nested': [1, 2]}
    resident(t, device)
    copied = copy.deepcopy({'first': t, 'again': t})
    result = copied['first']
    assert result is copied['again'] and result is not t
    assert type(result) is type(t) and result.dtype == t.dtype
    assert result.requires_grad == requires_grad and result.is_leaf
    resident(t, device)
    resident(result, device)
    np.testing.assert_array_equal(result.detach().cpu().numpy(), expected.astype('float32' if dtype == torch.float32 else 'float64'))
    if kind == 'tensor':
        assert result.custom == t.custom and result.custom is not t.custom
        result.custom['nested'].append(3)
        assert t.custom['nested'] == [1, 2]
    with torch.no_grad():
        result.fill_(7)
    np.testing.assert_array_equal(t.detach().cpu().numpy(), expected.astype('float32' if dtype == torch.float32 else 'float64'))
    resident(t, device)
    resident(result, device)

@pytest.mark.parametrize('dtype,values', [(torch.int64, [-3, 20000000001]), (torch.bool, [True, False])])
def test_deepcopy_integer_boolean(device, dtype, values):
    t = torch.tensor(values, dtype=dtype, device=device)
    resident(t, device)
    result = copy.deepcopy(t)
    resident(t, device)
    resident(result, device)
    assert result.dtype == dtype
    assert result.tolist() == values

def test_deepcopy_state_preserves_actual_module_residency(device):
    m = torch.nn.Linear(3, 2).to(device)
    for p in m.parameters():
        resident(p, device)
    copied = copy.deepcopy(m.state_dict())
    for n, p in m.named_parameters():
        resident(p, device)
        resident(copied[n], device)
        assert copied[n] is not p
        np.testing.assert_array_equal(copied[n].detach().cpu().numpy(), p.detach().cpu().numpy())
