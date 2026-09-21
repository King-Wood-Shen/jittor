"""Public storage-alias contracts across module conversion and data rebinding."""
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


class AliasModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1., 2.], dtype=torch.float32, device=device)
        )
        self.tied = self.weight
        self.register_buffer(
            "counter", torch.tensor([3], dtype=torch.int64, device=device)
        )


def check(tensor, expected):
    torch.testing.assert_close(
        tensor, torch.tensor(expected, dtype=tensor.dtype, device=tensor.device)
    )


def test_module_to_noop_preserves_exported_aliases(device):
    model = AliasModule(device)
    parameter = model.weight
    exported = model.state_dict()
    assert model.to(device=device, dtype=torch.float32) is model
    assert model.weight is parameter
    assert model.tied is parameter
    with torch.no_grad():
        exported["weight"].add_(2)
        check(model.weight, [3., 4.])
        check(exported["tied"], [3., 4.])
        model.weight.mul_(2)
        check(exported["weight"], [6., 8.])
        exported["counter"].add_(1)
        check(model.counter, [4])


def test_module_double_preserves_parameter_identity_but_rebinds_storage(device):
    model = AliasModule(device)
    (model.weight * 2).sum().backward()
    parameter = model.weight
    exported = model.state_dict()
    assert model.double() is model
    assert model.weight is parameter
    assert model.tied is parameter
    assert parameter.dtype == torch.float64
    assert parameter.grad.dtype == torch.float64
    check(parameter.grad, [2., 2.])
    assert model.counter.dtype == torch.int64
    assert exported["weight"].dtype == torch.float32
    assert exported["tied"].dtype == torch.float32
    with torch.no_grad():
        model.weight.add_(10)
        check(exported["weight"], [1., 2.])
        exported["tied"].fill_(7)
        check(exported["weight"], [7., 7.])
        check(model.weight, [11., 12.])
        exported["counter"].add_(1)
        check(model.counter, [4])


def test_module_device_migration_keeps_old_exports(device):
    if device != "cuda":
        pytest.skip("Requires a CUDA to CPU to CUDA conversion")
    model = AliasModule(device)
    parameter = model.weight
    gpu_export = model.state_dict()
    original_device = parameter.device
    assert model.cpu() is model
    assert model.weight is parameter
    assert model.tied is parameter
    assert parameter.device.type == "cpu"
    assert gpu_export["weight"].device == original_device
    assert gpu_export["counter"].device == original_device
    cpu_export = model.state_dict()
    with torch.no_grad():
        parameter.add_(10)
        check(gpu_export["weight"], [1., 2.])
        gpu_export["weight"].fill_(7)
        check(parameter, [11., 12.])
    assert model.cuda(original_device) is model
    assert model.weight is parameter
    assert model.tied is parameter
    assert parameter.device == original_device
    assert cpu_export["weight"].device.type == "cpu"
    assert cpu_export["counter"].device.type == "cpu"
    assert gpu_export["weight"].device == original_device
    with torch.no_grad():
        parameter.add_(10)
        check(cpu_export["weight"], [11., 12.])
        check(gpu_export["weight"], [7., 7.])
        cpu_export["weight"].fill_(9)
        check(parameter, [21., 22.])


def test_parameter_data_rebinding_preserves_old_detached_siblings(device):
    parameter = torch.nn.Parameter(
        torch.tensor([1., 2.], dtype=torch.float32, device=device)
    )
    first = parameter.detach()
    second = parameter.detach()
    with torch.no_grad():
        parameter.data = torch.tensor([3., 4., 5.], device=device)
        assert tuple(parameter.shape) == (3,)
        assert tuple(first.shape) == (2,)
        check(first, [1., 2.])
        check(second, [1., 2.])
        first.add_(2)
        check(second, [3., 4.])
        second.mul_(2)
        check(first, [6., 8.])
        check(parameter, [3., 4., 5.])
        parameter.fill_(9)
        check(first, [6., 8.])
        check(second, [6., 8.])
