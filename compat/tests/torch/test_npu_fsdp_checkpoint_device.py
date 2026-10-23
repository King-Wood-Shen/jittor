"""FULL FSDP checkpoint values use their shard's public NPU device."""
import os

import jittor as jt
import numpy as np
import pytest
import torch

from jittor.compat.fsdp2._state_dict import _checkpoint_tensor
from jittor.compat.torch.checkpoint_dcp import _assign_tree, _check_tree
from jittor.serialization.sharded import TensorShard


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


@pytest.mark.skipif(
    os.environ.get("JITTOR_TEST_DEVICES") != "npu", reason="requires a selected real ACL device"
)
def test_dcp_local_shard_restores_host_payload_to_npu_with_padding():
    assert jt.compiler.has_acl and jt.flags.backend_fallback == "error"
    jt.core._device_memory_window_start(0)
    target = torch.zeros(8, dtype=torch.bfloat16, device="npu")
    host = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.bfloat16, device="cpu")
    saved = TensorShard(host, (15,), 8)

    _check_tree(target, saved)
    _assign_tree(target, saved)

    torch.testing.assert_close(
        target, torch.tensor([1, 2, 3, 4, 5, 6, 7, 0], dtype=torch.bfloat16, device="npu")
    )
    torch.npu.synchronize()
    assert jt.core.dispatch_context([target])[0] == "acl"
    assert jt.core.backend_fallback_count() == 0
    assert jt.core._device_memory_window_peak(0) > 0
