"""Torch dimensional extrema route gradients to their returned index on ACL."""

import os

import numpy as np
import pytest
import torch
import jittor as jt


pytestmark = pytest.mark.skipif(
    os.environ.get("JITTOR_TEST_DEVICES") != "npu",
    reason="requires a selected real ACL device")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("kind", ["max", "min"])
def test_dimensional_ties_use_the_returned_index(dtype, keepdim, kind):
    assert jt.compiler.has_acl and jt.flags.backend_fallback == "error"
    if kind == "max":
        data = [[1., 3., -2., 0.], [4., -1., 2., 4.]]
        expected_values, expected_indices = [3., 4.], [1, 0]
        expected_grad = [[0., 1., 0., 0.], [1., 0., 0., 0.]]
    else:
        data = [[1., -2., -2., 3.], [4., -1., 2., -1.]]
        expected_values, expected_indices = [-2., -1.], [1, 1]
        expected_grad = [[0., 1., 0., 0.], [0., 1., 0., 0.]]
    x = torch.tensor(data, device="npu", dtype=dtype, requires_grad=True)
    values, indices = getattr(torch, kind)(x, dim=-1, keepdim=keepdim)
    assert values.dtype == dtype and indices.dtype == torch.int64
    assert values.device.type == indices.device.type == "npu"
    assert list(values.shape) == ([2, 1] if keepdim else [2])
    np.testing.assert_array_equal(values.float().cpu().numpy().reshape(-1), expected_values)
    np.testing.assert_array_equal(indices.cpu().numpy().reshape(-1), expected_indices)
    values.float().sum().backward()
    assert x.grad.device.type == "npu" and x.grad.dtype == dtype
    np.testing.assert_array_equal(x.grad.float().cpu().numpy(), expected_grad)
    assert jt.core.backend_fallback_count() == 0


@pytest.mark.parametrize("kind", ["max", "min"])
def test_vector_dimensional_extrema_keep_scalar_shape(kind):
    data = [3., 3., -1.] if kind == "max" else [-1., -1., 3.]
    x = torch.tensor(data, device="npu", requires_grad=True)
    values, indices = getattr(torch, kind)(x, dim=0)
    assert values.ndim == indices.ndim == 0
    assert indices.item() == 0
    values.backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), [1., 0., 0.])
    assert jt.core.backend_fallback_count() == 0
