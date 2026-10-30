"""The clipping cap follows the coefficient, including off-ambient CUDA."""

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_clip_cap_cpu_dtype_values_and_nan(dtype):
    import jittor as jt
    from jittor.compat.torch import _clip_grad_norm_device

    with jt.flag_scope(use_cuda=0):
        grad = jt.array(np.array([3.0, 4.0], dtype=dtype)).cast(dtype)
        total = _clip_grad_norm_device([grad], 1.0)
        total_np, grad_np = jt.fetch_sync([total, grad])
        assert str(total.dtype) == dtype
        assert grad.device == "cpu"
        np.testing.assert_allclose(total_np, 5.0, rtol=1e-6)
        np.testing.assert_allclose(grad_np, [0.6, 0.8], rtol=2e-6)

        nan_grad = jt.array(np.array([1.0, np.nan], dtype=dtype)).cast(dtype)
        _clip_grad_norm_device([nan_grad], 1.0, error_if_nonfinite=False)
        assert np.isnan(nan_grad.numpy()).all()


def test_clip_cap_follows_cuda3_with_ambient_cuda0():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks
    from jittor.compat.torch import _clip_grad_norm_device

    if not jt.compiler.has_cuda or torch.cuda.device_count() < 4:
        pytest.skip("four CUDA devices required")
    with jt.flag_scope(use_cuda=1):
        torch.cuda.set_device(0)
        grads = [jt.array(np.array(values, dtype="float32")).to_device(3)
                 for values in ([3.0, 4.0], [0.0, -3.0])]
        before = jt.core.backend_fallback_count()
        with jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            total = _clip_grad_norm_device(grads, 1.0)
            jt.sync_all(True)
            assert total.device_id == 3
            assert all(g.device_id == 3 and g.device == "cuda:3" for g in grads)
            total_np, *grad_np = jt.fetch_sync([total] + grads)
        np.testing.assert_allclose(total_np, np.sqrt(34.0), rtol=1e-5)
        np.testing.assert_allclose(
            np.concatenate(grad_np),
            np.array([3.0, 4.0, 0.0, -3.0]) / np.sqrt(34.0), rtol=2e-5)
        assert jt.core.backend_fallback_count() == before
