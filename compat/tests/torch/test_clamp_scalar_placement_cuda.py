"""Scalar clamp bounds must follow the input tensor's placement."""
import numpy as np
import pytest


def test_cpu_scalar_clamp_stays_cpu():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    value = torch.arange(32, dtype=torch.float32)
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        ramp = torch.clamp((value - 10) / 13, 0, 1)
        integers = torch.arange(8, dtype=torch.int32).clamp(2, 5)
        for tensor in (ramp, integers):
            tensor.sync()
            assert tensor.placement_backend == 0
            assert tensor.device.type == 'cpu'
        np.testing.assert_allclose(ramp.numpy(), np.clip((np.arange(32, dtype=np.float32) - 10) / 13,
                                                       0, 1), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(integers.numpy(), [2, 2, 2, 3, 4, 5, 5, 5])
    assert jt.core.backend_fallback_count() == before


def test_scalar_clamp_inside_cuda_module_follows_input():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    if not jt.compiler.has_cuda or not torch.cuda.is_available():
        pytest.skip('CUDA backend unavailable')

    class ClampProbe(torch.nn.Module):
        def forward(self, value_states):
            linear = (torch.arange(32, dtype=torch.float32) - 10) / 13
            ramp = torch.clamp(linear, 0, 1)
            integers = torch.arange(8, dtype=torch.int32).clamp(2, 5)
            following = torch.ones(4, dtype=torch.float32, device=value_states.device)
            return linear, ramp, integers, following

    jt.flags.use_cuda = 1
    torch.cuda.set_device(0)
    value_states = torch.ones((1, 16, 8, 128), dtype=torch.bfloat16, device='cuda:0')
    module = ClampProbe().to('cuda:0').eval()
    before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        linear, ramp, integers, following = module(value_states)
        torch.cuda.synchronize()
        for tensor in (linear, ramp, integers):
            tensor.sync()
            assert tensor.placement_backend == 0
            assert tensor.device.type == 'cpu'
        following.sync()
        assert following.placement_backend == 1
        assert following.device.type == 'cuda'
        np.testing.assert_allclose(ramp.numpy(), np.clip((np.arange(32, dtype=np.float32) - 10) / 13,
                                                       0, 1), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(integers.numpy(), [2, 2, 2, 3, 4, 5, 5, 5])
    assert jt.core.backend_fallback_count() == before
