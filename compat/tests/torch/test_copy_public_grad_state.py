"""Public requires_grad must not be inferred from native graph liveness."""
from contextlib import nullcontext
import json
import sys
import numpy as np
import pytest
import torch

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("source_inside_no_grad", [False, True])
@pytest.mark.parametrize("target_kind", ["buffer", "parameter"])
def test_copy_stopped_source_preserves_public_target_flag(device, source_inside_no_grad, target_kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("accelerator prerequisite: CUDA unavailable")
    module = torch.nn.Module()
    value = torch.zeros((2, 3), device=device, requires_grad=False)
    if target_kind == "parameter":
        module.register_parameter("value", torch.nn.Parameter(value))
    else:
        module.register_buffer("value", value)
    target = module.value
    expected_grad = target_kind == "parameter"
    assert target.requires_grad is expected_grad
    if "jittor" in sys.modules and not expected_grad:
        # Publicly frozen is distinct from a graph node permanently stopped.
        assert not target.is_stop_grad()
    with torch.no_grad() if source_inside_no_grad else nullcontext():
        source = torch.full((2, 3), 3., device=device, requires_grad=False)
    if "jittor" in sys.modules:
        assert source.is_stop_grad() is source_inside_no_grad
    source_before = source.requires_grad
    with torch.no_grad():
        result = target.copy_(source)
    print("COPY_SOURCE_FLAGS " + json.dumps({
        "device": device, "target_kind": target_kind,
        "source_inside_no_grad": source_inside_no_grad,
        "before": source_before, "after": source.requires_grad}))
    assert result is target and module.value is target
    assert target.requires_grad is expected_grad
    assert target.device.type == device
    np.testing.assert_array_equal(target.detach().cpu().numpy(), np.full((2, 3), 3.))
    if expected_grad:
        target.sum().backward()
        np.testing.assert_array_equal(target.grad.detach().cpu().numpy(), np.ones((2, 3)))
