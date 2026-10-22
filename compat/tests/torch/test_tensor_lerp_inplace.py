"""CUDA in-place interpolation through the shared pointwise graph."""
import numpy as np
import jittor as jt
import pytest
import torch
from _helpers.checkpoint_cuda import accelerator, snapshot


@pytest.mark.parametrize("weight", [.1, 1.5])
def test_lerp_scalar_broadcast_alias_storage_and_version(weight):
    target = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
    alias = target.view(-1)
    end = torch.tensor([5., 6.], device="cuda")
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
    x = torch.tensor([[1., 2.], [3., 4.]], device="cuda", requires_grad=True)
    end = torch.tensor([10., 20.], device="cuda", requires_grad=True)
    weight = torch.tensor([.25, .5], device="cuda", requires_grad=True)
    target = x * 2
    target.lerp_(end, weight)
    dx, de, dw = jt.grad(target.sum(), [x, end, weight])
    np.testing.assert_allclose(snapshot(dx), [[1.5, 1.], [1.5, 1.]], rtol=1e-6)
    np.testing.assert_allclose(snapshot(de), [.5, 1.], rtol=1e-6)
    np.testing.assert_allclose(snapshot(dw), [12., 28.], rtol=1e-6)


def test_lerp_no_grad_repeated_optimizer_state():
    buffer = torch.zeros(2, device="cuda")
    gradient = torch.tensor([2., 4.], device="cuda", requires_grad=True)
    with torch.no_grad():
        buffer.lerp_(gradient, .1)
        buffer.lerp_(gradient, .1)
    assert not buffer.requires_grad
    np.testing.assert_allclose(snapshot(buffer), [.38, .76], rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lerp_preserves_floating_dtype(dtype):
    target = torch.zeros(2, device="cuda", dtype=dtype)
    target.lerp_(torch.ones_like(target), .5)
    assert target.dtype == dtype
    np.testing.assert_array_equal(snapshot(target), [.5, .5])


def test_lerp_rejects_invalid_shape_dtype_and_leaf_writes():
    target = torch.zeros(1, device="cuda")
    with pytest.raises(RuntimeError, match="broadcast shape"):
        target.lerp_(torch.ones(2, device="cuda"), .5)
    with pytest.raises(RuntimeError, match="same dtype"):
        target.lerp_(torch.ones(1, device="cuda", dtype=torch.float64), .5)
    with pytest.raises(RuntimeError, match="same dtype"):
        target.lerp_(torch.ones_like(target), torch.ones(1, device="cuda", dtype=torch.float64))
    integral = torch.ones(1, device="cuda", dtype=torch.int64)
    with pytest.raises(RuntimeError, match="floating point"):
        integral.lerp_(torch.ones_like(integral), .5)
    parameter = torch.nn.Parameter(torch.ones(1, device="cuda"))
    with pytest.raises(RuntimeError, match="leaf"):
        parameter.lerp_(torch.ones_like(parameter), .5)
    np.testing.assert_array_equal(snapshot(target), [0.])
