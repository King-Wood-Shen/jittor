"""ACL slice gradient fill tensors follow the explicitly placed source."""
import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


def test_split_bf16_acl_source_with_ambient_default_cpu():
    if getattr(getattr(jt.compiler, "build_config", None), "backend", None) != "acl":
        pytest.skip("requires ACL build")
    with jt.runtime.scope(use_acl=1, use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        x = torch.arange(8, device="npu", dtype=torch.float32).to(torch.bfloat16)
        x = x.reshape(1, 1, 8).requires_grad_(True)
        assert x.placement_backend == 2
        assert jt.flags.device_id == -1
        with torch.autocast("npu", dtype=torch.bfloat16):
            left, right = torch.split(x, [3, 5], dim=-1)
            assert left.placement_backend == right.placement_backend == 2
            loss = left.float().sum() + right.float().sum() * 2
        loss.backward()
        np.testing.assert_array_equal(left.float().cpu().numpy(), [[[0, 1, 2]]])
        np.testing.assert_array_equal(right.float().cpu().numpy(), [[[3, 4, 5, 6, 7]]])
        np.testing.assert_array_equal(x.grad.float().cpu().numpy(), [[[1, 1, 1, 2, 2, 2, 2, 2]]])
