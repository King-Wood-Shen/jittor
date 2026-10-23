"""Optimizer-owned gradients retain higher-order edges on ACL."""

import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
@pytest.mark.parametrize("accumulate", [False, True])
def test_optimizer_backward_create_graph_preserves_grad_and_hessian(accumulate):
    assert torch.npu.is_available()
    parameter = torch.tensor([2.0], device="npu", requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    assert optimizer.param_groups[0]["params"][0] is parameter

    (parameter ** 3).sum().backward(create_graph=True)
    if accumulate:
        (parameter ** 2).sum().backward(create_graph=True)
    first = parameter.grad
    assert first is not None and first.requires_grad
    assert first.device.type == "npu"
    assert first.item() == pytest.approx(16.0 if accumulate else 12.0)

    second = torch.autograd.grad(first.sum(), parameter)[0]
    assert second.device.type == "npu"
    assert second.item() == pytest.approx(14.0 if accumulate else 12.0)
    parameter.grad = None


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_optimizer_first_order_accumulation_keeps_grad_storage_and_updates():
    parameter = torch.tensor([2.0], device="npu", requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)

    parameter.square().sum().backward()
    stored = parameter.grad
    assert stored is not None and not stored.requires_grad
    assert stored.item() == pytest.approx(4.0)

    parameter.square().sum().backward()
    assert parameter.grad is stored
    assert stored.item() == pytest.approx(8.0)
    optimizer.step()
    assert parameter.item() == pytest.approx(1.2)
