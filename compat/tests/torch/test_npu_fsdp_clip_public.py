"""Public split gradient clipping preserves local and sharded norms on ACL."""
import os

import jittor as jt
import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("JITTOR_TEST_DEVICES") != "npu",
    reason="requires selected real ACL devices")


def test_public_split_clip_plain_gradients():
    import torch.nn.utils.clip_grad as clip

    assert clip.get_total_norm is torch.nn.utils.get_total_norm
    assert clip.clip_grads_with_norm_ is torch.nn.utils.clip_grads_with_norm_
    parameter = torch.nn.Parameter(torch.zeros(2, device="npu"))
    parameter.grad = torch.tensor([3., 4.], device="npu")
    total = torch.nn.utils.get_total_norm([parameter.grad])
    assert total.item() == pytest.approx(5.)
    torch.nn.utils.clip_grads_with_norm_([parameter], 1., total)
    torch.testing.assert_close(parameter.grad, torch.tensor([.6, .8], device="npu"))
    assert total.device.type == parameter.grad.device.type == "npu"
    assert jt.core.backend_fallback_count() == 0


def test_public_split_clip_combines_shards_and_plain_gradients():
    from jittor import distributed as dist
    from torch.distributed.device_mesh import DeviceMesh

    if dist.get_world_size() != 2:
        pytest.skip("requires two real NPU ranks")
    mesh = DeviceMesh("npu", [0, 1])
    shard = torch.nn.Parameter(torch.zeros(1, device="npu"))
    shard.grad = torch.tensor([3. if dist.get_rank() == 0 else 4.], device="npu")
    object.__setattr__(shard.grad, "_fsdp_norm_group", mesh.get_group())
    plain = torch.nn.Parameter(torch.zeros(1, device="npu"))
    plain.grad = torch.tensor([12.], device="npu")
    shard_norm = torch.nn.utils.get_total_norm([shard.grad])
    plain_norm = torch.nn.utils.get_total_norm([plain.grad])
    total = torch.linalg.vector_norm(torch.stack([shard_norm, plain_norm]))
    assert shard_norm.item() == pytest.approx(5.)
    assert plain_norm.item() == pytest.approx(12.)
    assert total.item() == pytest.approx(13.)
    torch.nn.utils.clip_grads_with_norm_([shard, plain], 1., total)
    assert shard.grad.item() == pytest.approx((3. if dist.get_rank() == 0 else 4.) / 13.)
    assert plain.grad.item() == pytest.approx(12. / 13.)
    assert total.device.type == "npu"
    assert jt.core.backend_fallback_count() == 0
