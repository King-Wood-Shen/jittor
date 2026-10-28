"""Pointwise Tensor contracts required by custom ms-swift optimizers."""
import os
import numpy as np
import jittor as jt
import pytest
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks


def _device():
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
    assert device in ("cpu", "cuda")
    return device


@pytest.fixture(autouse=True)
def pointwise_device():
    device = _device()
    if device == "cuda":
        assert torch.cuda.is_available()
    with jt.flag_scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


def snapshot(value):
    assert jt.core.dispatch_context([value])[0] == _device()
    return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize('operation', ['addcmul_', 'addcdiv_'])
def test_scaled_inplace_broadcast_storage_alias_and_version(operation):
    target = torch.ones((2, 3), device=_device())
    alias = target.view(-1)
    left = torch.tensor([[1., 2., 3.]], device=_device())
    right = torch.tensor([[2.], [4.]], device=_device())
    target.sync()
    pointer, version = target.data_ptr(), target._autograd_version()
    result = getattr(target, operation)(left, right, value=.25)
    expected = 1 + .25 * (np.array([[1., 2., 3.]]) * np.array([[2.], [4.]])
                           if operation == 'addcmul_' else np.array([[1., 2., 3.]]) / np.array([[2.], [4.]]))
    assert result is target
    np.testing.assert_allclose(snapshot(target), expected, rtol=1e-6)
    np.testing.assert_allclose(snapshot(alias), expected.reshape(-1), rtol=1e-6)
    assert target.data_ptr() == pointer
    assert target._autograd_version() > version


@pytest.mark.parametrize('operation', ['addcmul_', 'addcdiv_'])
def test_scaled_inplace_preserves_connected_nonleaf_gradients(operation):
    x = torch.tensor([1., 2.], device=_device(), requires_grad=True)
    left = torch.tensor([3., 4.], device=_device(), requires_grad=True)
    right = torch.tensor([2., 4.], device=_device(), requires_grad=True)
    target = x * 2
    getattr(target, operation)(left, right, value=.5)
    dx, dl, dr = jt.grad(target.sum(), [x, left, right])
    np.testing.assert_allclose(snapshot(dx), [2., 2.], rtol=1e-6)
    expected_l = np.array([1., 2.]) if operation == 'addcmul_' else np.array([.25, .125])
    expected_r = np.array([1.5, 2.]) if operation == 'addcmul_' else np.array([-.375, -.125])
    np.testing.assert_allclose(snapshot(dl), expected_l, rtol=1e-6)
    np.testing.assert_allclose(snapshot(dr), expected_r, rtol=1e-6)


def test_scaled_inplace_keeps_destination_float_dtype():
    target = torch.tensor([1., 2.], device=_device(), dtype=torch.float16)
    left = torch.tensor([2., 4.], device=_device(), dtype=torch.float32)
    right = torch.tensor([.5, .25], device=_device(), dtype=torch.float32)
    target.addcmul_(left, right, value=.5)
    assert target.dtype == torch.float16
    np.testing.assert_array_equal(snapshot(target), np.array([1.5, 2.5], dtype=np.float16))


def test_scaled_inplace_rejects_shape_cast_and_leaf_writes():
    left = torch.tensor([1., 2.], device=_device())
    right = torch.tensor([2., 4.], device=_device())
    small = torch.zeros(1, device=_device())
    with pytest.raises(RuntimeError, match='broadcast shape'):
        small.addcmul_(left, right)
    integral = torch.ones(2, device=_device(), dtype=torch.int64)
    with pytest.raises(RuntimeError, match='cast'):
        integral.addcmul_(left, right)
    np.testing.assert_array_equal(snapshot(integral), [1, 1])
    parameter = torch.nn.Parameter(torch.ones(2, device=_device()))
    with pytest.raises(RuntimeError, match='leaf'):
        parameter.addcmul_(left, right)


@pytest.mark.parametrize("weight", [.1, 1.5])
def test_lerp_scalar_broadcast_alias_storage_and_version(weight):
    target = torch.tensor([[1., 2.], [3., 4.]], device=_device())
    alias = target.view(-1)
    end = torch.tensor([5., 6.], device=_device())
    target.sync()
    pointer, version = target.data_ptr(), target._autograd_version()
    result = target.lerp_(end, weight)
    expected = np.array([[1., 2.], [3., 4.]]) * (1 - weight) + np.array([5., 6.]) * weight
    assert result is target
    np.testing.assert_allclose(snapshot(target), expected, rtol=1e-6)
    np.testing.assert_allclose(snapshot(alias), expected.reshape(-1), rtol=1e-6)
    assert target.data_ptr() == pointer
    assert target._autograd_version() > version


def test_lerp_tensor_weight_broadcast_connected_gradients():
    x = torch.tensor([[1., 2.], [3., 4.]], device=_device(), requires_grad=True)
    end = torch.tensor([10., 20.], device=_device(), requires_grad=True)
    weight = torch.tensor([.25, .5], device=_device(), requires_grad=True)
    target = x * 2
    target.lerp_(end, weight)
    dx, de, dw = jt.grad(target.sum(), [x, end, weight])
    np.testing.assert_allclose(snapshot(dx), [[1.5, 1.], [1.5, 1.]], rtol=1e-6)
    np.testing.assert_allclose(snapshot(de), [.5, 1.], rtol=1e-6)
    np.testing.assert_allclose(snapshot(dw), [12., 28.], rtol=1e-6)


def test_lerp_no_grad_repeated_optimizer_state():
    buffer = torch.zeros(2, device=_device())
    gradient = torch.tensor([2., 4.], device=_device(), requires_grad=True)
    with torch.no_grad():
        buffer.lerp_(gradient, .1)
        buffer.lerp_(gradient, .1)
    assert not buffer.requires_grad
    np.testing.assert_allclose(snapshot(buffer), [.38, .76], rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lerp_preserves_floating_dtype(dtype):
    target = torch.zeros(2, device=_device(), dtype=dtype)
    target.lerp_(torch.ones_like(target), .5)
    assert target.dtype == dtype
    np.testing.assert_array_equal(snapshot(target), [.5, .5])


def test_lerp_rejects_invalid_shape_dtype_and_leaf_writes():
    target = torch.zeros(1, device=_device())
    with pytest.raises(RuntimeError, match="broadcast shape"):
        target.lerp_(torch.ones(2, device=_device()), .5)
    with pytest.raises(RuntimeError, match="same dtype"):
        target.lerp_(torch.ones(1, device=_device(), dtype=torch.float64), .5)
    with pytest.raises(RuntimeError, match="same dtype"):
        target.lerp_(torch.ones_like(target), torch.ones(1, device=_device(), dtype=torch.float64))
    integral = torch.ones(1, device=_device(), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="floating point"):
        integral.lerp_(torch.ones_like(integral), .5)
    parameter = torch.nn.Parameter(torch.ones(1, device=_device()))
    with pytest.raises(RuntimeError, match="leaf"):
        parameter.lerp_(torch.ones_like(parameter), .5)
    np.testing.assert_array_equal(snapshot(target), [0.])


@pytest.mark.parametrize("alpha", [.95, -2.0])
def test_add_alpha_broadcast_and_gradients(alpha):
    x = torch.tensor([[1., 2.], [3., 4.]], device=_device(), requires_grad=True)
    y = torch.tensor([2., 3.], device=_device(), requires_grad=True)
    result = x.add(y, alpha=alpha)
    expected = np.array([[1., 2.], [3., 4.]]) + alpha * np.array([2., 3.])
    np.testing.assert_allclose(snapshot(result), expected, rtol=1e-6)
    dx, dy = jt.grad(result.sum(), [x, y])
    np.testing.assert_array_equal(snapshot(dx), np.ones((2, 2)))
    np.testing.assert_allclose(snapshot(dy), np.full(2, 2 * alpha), rtol=1e-6)
    np.testing.assert_array_equal(snapshot(x), [[1., 2.], [3., 4.]])


def test_add_scalar_default_functional_and_dunder_controls():
    x = torch.tensor([1., 2.], device=_device())
    np.testing.assert_array_equal(snapshot(x.add(3, alpha=2)), [7., 8.])
    np.testing.assert_array_equal(snapshot(x.add(3)), [4., 5.])
    np.testing.assert_array_equal(snapshot(torch.add(x, 3, alpha=2)), [7., 8.])
    np.testing.assert_array_equal(snapshot(x + 3), [4., 5.])
