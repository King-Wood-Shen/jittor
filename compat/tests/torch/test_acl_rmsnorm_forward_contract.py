"""An explicit Torch RMSNorm forward keeps its BF16 ACL numerical contract."""

import numpy as np


def test_bfloat16_acl_module_call_preserves_explicit_forward():
    import jittor as jt
    import torch
    from jittor._runtime.fallback import forbid_backend_fallbacks

    assert jt.compiler.has_acl and torch.npu.is_available()

    class AffineRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2048, dtype=torch.bfloat16))
            self.variance_epsilon = 1e-6

        def forward(self, hidden):
            value = hidden.float()
            variance = value.pow(2).mean(-1, keepdim=True)
            normalized = self.weight * (
                value * torch.rsqrt(variance + self.variance_epsilon)).to(hidden.dtype)
            return normalized + torch.ones_like(normalized)

    jt.flags.use_acl = 1
    jt.flags.use_cuda = 1
    torch.npu.set_device(0)
    module = AffineRMSNorm().to("npu:0").eval()
    values = (np.arange(2 * 2048, dtype=np.float32).reshape(2, 2048) - 2048) / 4096
    hidden = torch.tensor(values, dtype=torch.bfloat16, device="npu:0")
    before = jt.core.backend_fallback_count()
    with torch.no_grad(), forbid_backend_fallbacks():
        expected = module.forward(hidden)
        actual = module(hidden)
        for tensor in (expected, actual):
            tensor.sync()
            assert tensor.device.type == "npu"
            assert tensor.location() == "device"
        np.testing.assert_array_equal(actual.float().cpu().numpy(),
                                      expected.float().cpu().numpy())
    assert jt.core.backend_fallback_count() == before
