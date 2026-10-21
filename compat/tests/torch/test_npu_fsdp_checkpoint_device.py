"""FULL FSDP checkpoint values use their shard's public NPU device."""
import os

import jittor as jt
import numpy as np
import pytest
import torch

from jittor.compat.fsdp2._state_dict import _checkpoint_tensor


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected real ACL device")
@pytest.mark.parametrize("preserve_dtype", [False, True])
def test_checkpoint_tensor_restores_to_the_reference_npu(preserve_dtype):
    assert jt.compiler.has_acl and jt.flags.backend_fallback == "error"
    reference = torch.zeros(3, dtype=torch.bfloat16, device="npu")
    # Serialized model/optimizer data starts on the host; restoration must
    # execute its conversion on the shard's actual accelerator.
    saved = torch.tensor(np.array([.125, -.5, 1.25], dtype=np.float32), device="cpu")
    restored = _checkpoint_tensor(saved, reference, preserve_dtype=preserve_dtype)
    dtype = torch.float32 if preserve_dtype else torch.bfloat16
    torch.testing.assert_close(
        restored, torch.tensor([.125, -.5, 1.25], dtype=dtype, device="npu"))
    torch.npu.synchronize()
    assert restored.device == reference.device and restored.device.type == "npu"
    assert restored.dtype == dtype and not restored.requires_grad
    assert jt.core.dispatch_context([restored])[0] == "acl"
    assert jt.core.backend_fallback_count() == 0
