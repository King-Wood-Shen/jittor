"""CUDA scaled in-place pointwise arithmetic and storage contracts."""
import numpy as np
import jittor as jt
import pytest
import torch
from _helpers.checkpoint_cuda import accelerator, snapshot


@pytest.mark.parametrize('operation', ['addcmul_', 'addcdiv_'])
def test_scaled_inplace_broadcast_storage_alias_and_version(operation):
    target = torch.ones((2, 3), device='cuda')
    alias = target.view(-1)
    left = torch.tensor([[1., 2., 3.]], device='cuda')
    right = torch.tensor([[2.], [4.]], device='cuda')
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
    x = torch.tensor([1., 2.], device='cuda', requires_grad=True)
    left = torch.tensor([3., 4.], device='cuda', requires_grad=True)
    right = torch.tensor([2., 4.], device='cuda', requires_grad=True)
    target = x * 2
    getattr(target, operation)(left, right, value=.5)
    dx, dl, dr = jt.grad(target.sum(), [x, left, right])
    np.testing.assert_allclose(snapshot(dx), [2., 2.], rtol=1e-6)
    expected_l = np.array([1., 2.]) if operation == 'addcmul_' else np.array([.25, .125])
    expected_r = np.array([1.5, 2.]) if operation == 'addcmul_' else np.array([-.375, -.125])
    np.testing.assert_allclose(snapshot(dl), expected_l, rtol=1e-6)
    np.testing.assert_allclose(snapshot(dr), expected_r, rtol=1e-6)


def test_scaled_inplace_keeps_destination_float_dtype():
    target = torch.tensor([1., 2.], device='cuda', dtype=torch.float16)
    left = torch.tensor([2., 4.], device='cuda', dtype=torch.float32)
    right = torch.tensor([.5, .25], device='cuda', dtype=torch.float32)
    target.addcmul_(left, right, value=.5)
    assert target.dtype == torch.float16
    np.testing.assert_array_equal(snapshot(target), np.array([1.5, 2.5], dtype=np.float16))


def test_scaled_inplace_rejects_shape_cast_and_leaf_writes():
    left = torch.tensor([1., 2.], device='cuda')
    right = torch.tensor([2., 4.], device='cuda')
    small = torch.zeros(1, device='cuda')
    with pytest.raises(RuntimeError, match='broadcast shape'):
        small.addcmul_(left, right)
    integral = torch.ones(2, device='cuda', dtype=torch.int64)
    with pytest.raises(RuntimeError, match='cast'):
        integral.addcmul_(left, right)
    np.testing.assert_array_equal(snapshot(integral), [1, 1])
    parameter = torch.nn.Parameter(torch.ones(2, device='cuda'))
    with pytest.raises(RuntimeError, match='leaf'):
        parameter.addcmul_(left, right)
