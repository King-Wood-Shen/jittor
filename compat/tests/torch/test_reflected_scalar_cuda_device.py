"""Python scalar reflection follows the tensor on a non-current CUDA device."""
import numpy as np
import pytest


def test_reflected_power_on_cuda1_uses_cuda1_scalar():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if not jt.compiler.has_cuda or torch.cuda.device_count() < 2:
        pytest.skip('two CUDA devices required')

    torch.cuda.set_device(0)
    before = jt.core.backend_fallback_count()
    with torch.no_grad(), forbid_backend_fallbacks():
        for _ in range(8):
            exponent = torch.arange(0, 64, 2, dtype=torch.float32, device='cuda:1') / 64
            result = 10000 ** exponent
            result.sync()
            assert result.device.type == 'cuda'
            assert result.device.index == 1
            np.testing.assert_allclose(result.cpu().numpy(), 10000 ** (np.arange(0, 64, 2) / 64), rtol=2e-4)
    assert jt.core.backend_fallback_count() == before
