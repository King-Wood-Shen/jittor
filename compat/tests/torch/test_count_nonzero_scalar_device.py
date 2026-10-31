"""Scalar comparison and count semantics on CPU and non-default CUDA."""

import numpy as np
import pytest


def _check_counts(torch, values, device):
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    mask = tensor != 0
    mask.sync()
    np.testing.assert_array_equal(mask.numpy(), np.not_equal(values, 0))
    reverse = torch.ne(0, tensor)
    reverse.sync()
    np.testing.assert_array_equal(reverse.numpy(), np.not_equal(0, values))
    numpy_scalar = tensor != np.float32(0)
    numpy_scalar.sync()
    np.testing.assert_array_equal(numpy_scalar.numpy(), np.not_equal(values, 0))
    for dim, expected in ((None, 3), (0, [0, 1, 2]), (1, [2, 1])):
        counted = torch.count_nonzero(tensor, dim=dim)
        counted.sync()
        assert counted.dtype == torch.int64
        assert counted.device == tensor.device
        np.testing.assert_array_equal(counted.numpy(), expected)

    empty = torch.empty((0, 3), dtype=torch.float32, device=device)
    for dim, expected in ((None, 0), (0, [0, 0, 0]), (1, [])):
        counted = torch.count_nonzero(empty, dim=dim)
        counted.sync()
        assert counted.dtype == torch.int64
        assert counted.device == empty.device
        np.testing.assert_array_equal(counted.numpy(), expected)


def test_count_nonzero_scalar_cpu_matches_native_torch():
    import jittor as jt
    import torch

    values = np.array([[0.0, np.nan, 2.0], [0.0, 0.0, -1.0]], dtype="float32")
    with jt.flag_scope(use_cuda=0):
        _check_counts(torch, values, "cpu")


def test_count_nonzero_scalar_cuda1_with_ambient_cuda0():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if not jt.compiler.has_cuda or torch.cuda.device_count() < 2:
        pytest.skip("two CUDA devices required")
    values = np.array([[0.0, np.nan, 2.0], [0.0, 0.0, -1.0]], dtype="float32")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_cuda=1), torch.cuda.device(0):
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            _check_counts(torch, values, "cuda:0")
            _check_counts(torch, values, "cuda:1")
            jt.sync_all(True)
    assert jt.core.backend_fallback_count() == before
