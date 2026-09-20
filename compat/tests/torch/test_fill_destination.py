"""Exact fill values and destination semantics, shared with the Torch oracle."""
import pytest
import torch

@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("shape", [(), (2, 3)])
@pytest.mark.parametrize("dtype,value", [("int64", 2**33), ("int64", -(2**33)-1), ("float32", 1.25)])
def test_fill_exact_value_and_destination(device, shape, dtype, value):
    x = torch.zeros(shape, dtype=getattr(torch, dtype), device=device)
    original = x
    result = x.fill_(value)
    assert result is original
    assert x.dtype == getattr(torch, dtype)
    assert tuple(x.shape) == shape
    assert x.device.type == device
    assert (x.cpu().numpy() == value).all()

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fill_view_updates_base(device):
    base = torch.zeros((2, 3), device=device)
    view = base[0]
    assert view.fill_(1.25) is view
    assert (base.cpu().numpy()[0] == 1.25).all()
    assert (base.cpu().numpy()[1] == 0).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fill_rejects_non_scalar_tensor(device):
    value = torch.tensor([1.25], device=device)
    with pytest.raises(RuntimeError):
        torch.zeros((2, 3), device=device).fill_(value)
