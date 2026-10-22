"""CUDA Tensor.add shares the functional alpha-aware frontend."""
import numpy as np
import jittor as jt
import pytest
import torch
from _helpers.checkpoint_cuda import accelerator, snapshot

@pytest.mark.parametrize("alpha", [.95, -2.0])
def test_add_alpha_broadcast_and_gradients(alpha):
    x = torch.tensor([[1., 2.], [3., 4.]], device="cuda", requires_grad=True)
    y = torch.tensor([2., 3.], device="cuda", requires_grad=True)
    result = x.add(y, alpha=alpha)
    expected = np.array([[1., 2.], [3., 4.]]) + alpha * np.array([2., 3.])
    np.testing.assert_allclose(snapshot(result), expected, rtol=1e-6)
    dx, dy = jt.grad(result.sum(), [x, y])
    np.testing.assert_array_equal(snapshot(dx), np.ones((2, 2)))
    np.testing.assert_allclose(snapshot(dy), np.full(2, 2 * alpha), rtol=1e-6)
    np.testing.assert_array_equal(snapshot(x), [[1., 2.], [3., 4.]])

def test_add_scalar_default_functional_and_dunder_controls():
    x = torch.tensor([1., 2.], device="cuda")
    np.testing.assert_array_equal(snapshot(x.add(3, alpha=2)), [7., 8.])
    np.testing.assert_array_equal(snapshot(x.add(3)), [4., 5.])
    np.testing.assert_array_equal(snapshot(torch.add(x, 3, alpha=2)), [7., 8.])
    np.testing.assert_array_equal(snapshot(x + 3), [4., 5.])
