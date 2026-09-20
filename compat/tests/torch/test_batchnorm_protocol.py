"""Torch BatchNorm state, mode and legacy-checkpoint behavior."""
from collections import OrderedDict
from contextlib import ExitStack
import io
import sys

import numpy as np
import pytest
import torch


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    name = request.param
    if name == 'cuda' and not torch.cuda.is_available():
        pytest.skip('accelerator prerequisite: CUDA unavailable')
    with ExitStack() as stack:
        native = sys.modules.get('jittor')
        if native is not None:
            from jittor._runtime.fallback import forbid_backend_fallbacks
            stack.enter_context(native.runtime.scope(backend_fallback='error'))
            stack.enter_context(forbid_backend_fallbacks())
            before = native.core.backend_fallback_count()
        yield name
        if native is not None:
            assert native.core.backend_fallback_count() == before


def _array(value):
    return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize('kind,valid,invalid', [
    ('BatchNorm1d', (2, 3), (1, 4)),
    ('BatchNorm2d', (4,), (2, 3, 5)),
    ('BatchNorm3d', (5,), (3, 4, 6)),
])
def test_public_dimensions_and_counter(kind, valid, invalid, device):
    layer_type = getattr(torch.nn, kind)
    layer = layer_type(4, device=device)
    assert type(layer).__name__ == kind
    assert layer.num_batches_tracked.dtype == torch.int64
    assert tuple(layer.num_batches_tracked.shape) == ()
    assert layer.num_batches_tracked.device.type == device
    assert 'num_batches_tracked' in layer.state_dict()
    for ndim in valid:
        layer(torch.ones((2, 4) + (2,) * (ndim - 2), device=device))
    assert layer.num_batches_tracked.item() == len(valid)
    for ndim in invalid:
        shape = (4,) if ndim == 1 else (2, 4) + (2,) * (ndim - 2)
        with pytest.raises(ValueError):
            layer(torch.ones(shape, device=device))


@pytest.mark.parametrize('momentum', [.1, None])
def test_training_statistics_and_eval_counter(momentum, device):
    layer = torch.nn.BatchNorm2d(4, momentum=momentum, device=device)
    mean = np.zeros(4)
    variance = np.ones(4)
    for step in range(3):
        values = np.random.RandomState(10 + step).randn(2, 4, 3, 3).astype('float32')
        x = torch.tensor(values, device=device, requires_grad=True)
        y = layer(x)
        y.square().mean().backward()
        assert x.grad is not None and layer.weight.grad is not None and layer.bias.grad is not None
        factor = .1 if momentum is not None else 1. / (step + 1)
        batch_mean = values.astype('float64').mean(axis=(0, 2, 3))
        batch_var = values.astype('float64').var(axis=(0, 2, 3), ddof=1)
        mean += factor * (batch_mean - mean)
        variance += factor * (batch_var - variance)
        np.testing.assert_allclose(_array(layer.running_mean), mean, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(_array(layer.running_var), variance, rtol=2e-5, atol=2e-6)
        assert layer.num_batches_tracked.item() == step + 1
    before = {n: _array(v) for n, v in layer.named_buffers()}
    layer.eval()
    layer(torch.ones((2, 4, 3, 3), device=device))
    for name, value in layer.named_buffers():
        np.testing.assert_array_equal(_array(value), before[name])
    layer.reset_running_stats()
    assert layer.num_batches_tracked.item() == 0
    np.testing.assert_array_equal(_array(layer.running_mean), np.zeros(4))
    np.testing.assert_array_equal(_array(layer.running_var), np.ones(4))
    with torch.no_grad():
        layer.weight.fill_(2)
        layer.bias.fill_(3)
    layer.reset_parameters()
    np.testing.assert_array_equal(_array(layer.weight), np.ones(4))
    np.testing.assert_array_equal(_array(layer.bias), np.zeros(4))


@pytest.mark.parametrize('tracking', [True, False])
def test_affine_false_dtype_and_untracked_eval(tracking, device):
    layer = torch.nn.BatchNorm2d(4, affine=False, track_running_stats=tracking,
                                 device=device, dtype=torch.float64)
    assert layer.weight is None and layer.bias is None
    assert list(layer.parameters()) == []
    if tracking:
        assert layer.running_mean.dtype == torch.float64
        assert layer.running_var.dtype == torch.float64
        assert layer.num_batches_tracked.dtype == torch.int64
    else:
        assert layer.running_mean is None and layer.running_var is None
        assert layer.num_batches_tracked is None
        assert not layer.state_dict()
    x = torch.tensor(np.arange(72).reshape(2, 4, 3, 3) / 72,
                     device=device, dtype=torch.float64)
    layer.train()
    trained = _array(layer(x))
    if not tracking:
        layer.eval()
        np.testing.assert_allclose(_array(layer(x)), trained, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('version', [None, 1, 2])
@pytest.mark.parametrize('strict', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_legacy_counter_loading_is_versioned(version, strict, nested, device):
    source = torch.nn.BatchNorm2d(4, device=device)
    target = torch.nn.BatchNorm2d(4, device=device)
    target.num_batches_tracked.fill_(7)
    if nested:
        source, target = torch.nn.Sequential(source), torch.nn.Sequential(target)
    key = ('0.' if nested else '') + 'num_batches_tracked'
    state = OrderedDict(source.state_dict())
    state.pop(key)
    if version is not None:
        state._metadata = {'': {'version': 1 if nested else version}}
        if nested:
            state._metadata['0'] = {'version': version}
    original_keys = tuple(state)
    original_metadata = repr(getattr(state, '_metadata', None))
    if version == 2 and strict:
        with pytest.raises(RuntimeError, match='num_batches_tracked'):
            target.load_state_dict(state, strict=True)
    else:
        result = target.load_state_dict(state, strict=strict)
        assert result.missing_keys == ([key] if version == 2 else [])
        assert result.unexpected_keys == []
    assert tuple(state) == original_keys
    assert repr(getattr(state, '_metadata', None)) == original_metadata
    counter = target[0].num_batches_tracked if nested else target.num_batches_tracked
    assert counter.item() == 7


@pytest.mark.parametrize('name', ['num_batches_tracked', 'ordinary_num_batches_tracked'])
def test_counter_suffix_on_an_ordinary_module_is_not_exempt(name, device):
    layer = torch.nn.Module()
    layer.register_buffer(name, torch.tensor(0, device=device, dtype=torch.int64))
    with pytest.raises(RuntimeError, match=name):
        layer.load_state_dict({}, strict=True)
    assert layer.load_state_dict({}, strict=False).missing_keys == [name]


def test_modern_metadata_survives_local_checkpoint(device):
    layer = torch.nn.Sequential(torch.nn.BatchNorm2d(4, device=device))
    layer[0].num_batches_tracked.fill_(2**33)
    state = layer.state_dict()
    assert isinstance(state, OrderedDict)
    assert state._metadata['0']['version'] == 2
    assert not state['0.weight'].requires_grad
    assert layer.state_dict(keep_vars=True)['0.weight'].requires_grad
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location=device, weights_only=True)
    assert isinstance(loaded, OrderedDict)
    assert loaded._metadata == state._metadata
    restored = torch.nn.Sequential(torch.nn.BatchNorm2d(4, device=device))
    restored.load_state_dict(loaded, strict=True)
    assert restored[0].num_batches_tracked.item() == 2**33
    del loaded['0.num_batches_tracked']
    with pytest.raises(RuntimeError, match='num_batches_tracked'):
        restored.load_state_dict(loaded, strict=True)

@pytest.mark.parametrize("nested", [False, True])
def test_copy_loading_preserves_destination_dtype_device_and_identity(nested, device):
    source = torch.nn.BatchNorm2d(4, dtype=torch.float64, device="cpu")
    target = torch.nn.BatchNorm2d(4, device=device)
    source.running_mean.fill_(3)
    source.num_batches_tracked.fill_(2**33)
    source.register_buffer("optional", None)
    target.register_buffer("optional", None)
    source.register_buffer("temporary", torch.tensor(8), persistent=False)
    target.register_buffer("temporary", torch.tensor(9, device=device), persistent=False)
    old_parameter, old_buffer = target.weight, target.running_mean
    optimizer = torch.optim.SGD(target.parameters(), lr=.1)
    if nested:
        source, target = torch.nn.Sequential(source), torch.nn.Sequential(target)
    result = target.load_state_dict(source.state_dict(), assign=False)
    layer = target[0] if nested else target
    assert not result.missing_keys and not result.unexpected_keys
    assert layer.weight is old_parameter and layer.running_mean is old_buffer
    assert optimizer.param_groups[0]["params"][0] is old_parameter
    assert layer.weight.dtype == torch.float32
    assert layer.weight.requires_grad
    assert layer.weight.device.type == device
    assert layer.running_mean.device.type == device
    assert layer.num_batches_tracked.device.type == device
    assert layer.num_batches_tracked.item() == 2**33
    assert layer.temporary.item() == 9
    np.testing.assert_array_equal(_array(layer.running_mean), np.full(4, 3))
