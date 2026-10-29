"""Scalar clamp bounds follow the input during a module's device scope."""

import numpy as np


def test_cpu_clamp_bounds_inside_npu_module():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    assert jt.compiler.has_acl and torch.npu.is_available()

    class ClampProbe(torch.nn.Module):
        def forward(self, value_states):
            positions = torch.arange(0, 64, 2, dtype=torch.float32,
                                     device=value_states.device)
            frequency = 1.0 / (10000 ** (positions / 64))
            linear = (torch.arange(32, dtype=torch.float32) - 10) / 13
            ramp = torch.clamp(linear, 0, 1)
            integers = torch.arange(8, dtype=torch.int32).clamp(2, 5)
            following = torch.ones(4, dtype=torch.float32,
                                   device=value_states.device)
            return frequency, linear, ramp, integers, following

    jt.flags.use_acl = 1
    jt.flags.use_cuda = 1
    torch.npu.set_device(0)
    value_states = torch.ones((1, 16, 8, 128), dtype=torch.bfloat16,
                              device="npu:0")
    module = ClampProbe().to("npu:0").eval()
    fallback_before = jt.core.backend_fallback_count()
    with forbid_backend_fallbacks():
        frequency, linear, ramp, integers, following = module(value_states)
        torch.npu.synchronize()
        for tensor in (linear, ramp, integers):
            tensor.sync()
            assert tensor.placement_backend == 0
            assert tensor.device.type == "cpu"
            assert tensor.location() == "cpu"
        for tensor in (frequency, following):
            tensor.sync()
            assert tensor.placement_backend == 2
            assert tensor.device.type == "npu"
            assert tensor.location() == "device"
        np.testing.assert_allclose(
            ramp.numpy(), np.clip((np.arange(32, dtype=np.float32) - 10) / 13,
                                  0, 1), atol=1e-6, rtol=1e-6)
        np.testing.assert_array_equal(integers.numpy(), [2, 2, 2, 3, 4, 5, 5, 5])
    assert jt.core.backend_fallback_count() == fallback_before
