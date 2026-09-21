"""Torch contracts for detached storage aliases and checkpoint exports."""
import copy
import gc

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def assert_values(tensor, values):
    expected = torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    assert torch.equal(tensor, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("operation", ["add", "mul", "copy", "fill"])
def test_detach_bidirectional_inplace(device, dtype, operation):
    source = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    alias = source.detach()
    assert alias is not source
    with torch.no_grad():
        if operation == "add":
            alias.add_(2)
            source.add_(3)
            expected = [6, 7, 8]
        elif operation == "mul":
            alias.mul_(2)
            source.mul_(3)
            expected = [6, 12, 18]
        elif operation == "copy":
            alias.copy_(torch.tensor([4, 5, 6], dtype=dtype, device=device))
            assert_values(source, [4, 5, 6])
            source.copy_(torch.tensor([7, 8, 9], dtype=dtype, device=device))
            expected = [7, 8, 9]
        else:
            alias.fill_(4)
            assert_values(source, [4, 4, 4])
            source.fill_(7)
            expected = [7, 7, 7]
    assert_values(source, expected)
    assert_values(alias, expected)


def test_detach_siblings_survive_source_collection(device):
    source = torch.tensor([1., 2., 3.], device=device)
    first = source.detach()
    second = source.detach()
    third = first.detach()
    del source
    gc.collect()
    with torch.no_grad():
        first.add_(1)
        assert_values(second, [2, 3, 4])
        third.mul_(2)
        assert_values(first, [4, 6, 8])
        del first
        gc.collect()
        second.fill_(9)
    assert_values(third, [9, 9, 9])


@pytest.mark.parametrize("copier", [lambda x: x.clone(), copy.deepcopy])
def test_detach_clone_and_deepcopy_are_independent(device, copier):
    source = torch.tensor([1., 2.], device=device)
    alias = source.detach()
    independent = copier(alias)
    with torch.no_grad():
        alias.add_(2)
        assert_values(independent, [1, 2])
        independent.fill_(8)
    assert_values(source, [3, 4])
    assert_values(alias, [3, 4])


def test_detach_gradient_metadata_is_independent(device):
    source = torch.tensor([1., 2.], device=device, requires_grad=True)
    alias = source.detach()
    assert source.requires_grad
    assert not alias.requires_grad
    with torch.no_grad():
        alias.add_(2)
    (source * source).sum().backward()
    assert_values(source.grad, [6, 8])
    assert alias.grad is None
    alias.requires_grad_(True)
    (alias * 3).sum().backward()
    assert_values(alias.grad, [3, 3])
    assert_values(source.grad, [6, 8])


class AliasModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1., 2.], device=device))
        self.tied = self.weight
        self.register_buffer("counter", torch.tensor([0], dtype=torch.int64, device=device))


def test_state_dict_exports_alias_parameters_and_integer_buffers(device):
    model = AliasModule(device)
    exported = model.state_dict()
    assert not exported["weight"].requires_grad
    with torch.no_grad():
        exported["weight"].add_(2)
        assert_values(model.weight, [3, 4])
        assert_values(model.tied, [3, 4])
        assert_values(exported["tied"], [3, 4])
        model.tied.mul_(2)
        assert_values(exported["weight"], [6, 8])
        exported["counter"].fill_(4)
        assert_values(model.counter, [4])
        model.counter.add_(1)
        assert_values(exported["counter"], [5])
    assert model.weight.requires_grad


def test_state_dict_alias_survives_module_collection(device):
    model = AliasModule(device)
    exported = model.state_dict()
    sibling = model.weight.detach()
    del model
    gc.collect()
    with torch.no_grad():
        exported["tied"].fill_(7)
        assert_values(sibling, [7, 7])
        sibling.add_(2)
    assert_values(exported["weight"], [9, 9])


@pytest.mark.parametrize("detach_first", [False, True])
@pytest.mark.parametrize("view_kind", ["slice", "transpose"])
def test_detach_and_views_preserve_bidirectional_updates(device, detach_first, view_kind):
    source = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device=device)
    base = source.detach() if detach_first else source
    view = base[:, 1:] if view_kind == "slice" else base.transpose(0, 1)
    alias = view if detach_first else view.detach()
    with torch.no_grad():
        alias.add_(10)
        expected = [[1, 12, 13], [4, 15, 16]] if view_kind == "slice" else [[11, 12, 13], [14, 15, 16]]
        assert_values(source, expected)
        source.fill_(2)
        assert_values(alias, [[2, 2], [2, 2]] if view_kind == "slice" else [[2, 2], [2, 2], [2, 2]])


def test_detach_does_not_bypass_saved_tensor_version_check(device):
    source = torch.tensor([1., 2.], device=device, requires_grad=True)
    loss = (source * source).sum()
    with torch.no_grad():
        source.detach().add_(1)
    with pytest.raises(RuntimeError):
        loss.backward()


def test_module_dtype_conversion_leaves_retained_exports_on_old_storage(device):
    model = torch.nn.Linear(2, 2).to(device)
    with torch.no_grad():
        model.weight.fill_(1)
    weight = model.weight
    exported = model.state_dict()["weight"]
    model.double()
    assert model.weight is weight
    assert weight.dtype == torch.float64 and weight.requires_grad and weight.is_leaf
    assert exported.dtype == torch.float32
    with torch.no_grad():
        weight.fill_(2)
        assert_values(exported, [[1, 1], [1, 1]])
        exported.fill_(3)
    assert_values(weight, [[2, 2], [2, 2]])
