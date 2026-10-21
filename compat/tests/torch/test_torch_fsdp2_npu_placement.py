"""Implicit FSDP meshes retain the actual ACL device type."""

import os

import pytest
import torch
import jittor as jt

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected real ACL device")
def test_default_fsdp_mesh_uses_npu_and_preserves_forward_device():
    assert jt.compiler.has_acl and jt.flags.use_cuda
    assert jt.flags.backend_fallback == "error"
    before = int(jt.core.backend_fallback_count())
    model = torch.nn.Linear(4, 4, bias=False).to("npu")
    fully_shard(model)
    assert model._fsdp_state.mesh.device_type == "npu"
    assert DeviceMesh(mesh=[0]).device_type == "npu"
    assert DeviceMesh("cpu", [0]).device_type == "cpu"

    output = model(torch.ones((1, 4), device="npu"))
    jt.sync_all(True)
    assert model.weight.device.type == output.device.type == "npu"
    assert torch.isfinite(output).all().item()
    assert int(jt.core.backend_fallback_count()) == before == 0
