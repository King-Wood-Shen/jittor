"""BF16 extrema use real ACL kernels for values, indices and gradients."""

import os

import numpy as np
import pytest
import torch
import jittor as jt


pytestmark = pytest.mark.skipif(
    os.environ.get("JITTOR_TEST_DEVICES") != "npu",
    reason="requires a selected real ACL device")


def _input(requires_grad=False, tied=True):
    assert jt.compiler.has_acl and jt.flags.backend_fallback == "error"
    return torch.tensor([[1., 3., -2., 0.], [4., -1., 2., 4. if tied else 0.]],
                        device="npu", dtype=torch.bfloat16,
                        requires_grad=requires_grad)


@pytest.mark.parametrize("kind", ["max", "min"])
def test_bfloat16_extrema_values_indices_and_keepdim(kind):
    x = _input()
    expected_values = [3., 4.] if kind == "max" else [-2., -1.]
    expected_indices = [1, 0] if kind == "max" else [2, 1]
    operation = getattr(torch, kind)
    for keepdim in (False, True):
        values, indices = operation(x, dim=-1, keepdim=keepdim)
        assert values.dtype == torch.bfloat16
        assert values.device.type == indices.device.type == "npu"
        assert list(values.shape) == ([2, 1] if keepdim else [2])
        np.testing.assert_array_equal(values.float().cpu().numpy().reshape(-1),
                                      expected_values)
        np.testing.assert_array_equal(indices.cpu().numpy().reshape(-1),
                                      expected_indices)
    direct = getattr(torch, "arg" + kind)(x, dim=-1)
    np.testing.assert_array_equal(direct.cpu().numpy(), expected_indices)
    assert jt.core.backend_fallback_count() == 0


@pytest.mark.parametrize("kind", ["max", "min"])
def test_bfloat16_extrema_backward_stays_on_npu(kind):
    x = _input(requires_grad=True, tied=False)
    values, _ = getattr(torch, kind)(x, dim=-1)
    values.float().sum().backward()
    expected = ([[0., 1., 0., 0.], [1., 0., 0., 0.]] if kind == "max"
                else [[0., 0., 1., 0.], [0., 1., 0., 0.]])
    assert x.grad.device.type == "npu" and x.grad.dtype == torch.bfloat16
    np.testing.assert_array_equal(x.grad.float().cpu().numpy(), expected)
    assert jt.core.backend_fallback_count() == 0
