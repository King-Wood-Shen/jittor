"""Ordinary load_state_dict copies preserve aliases and gradient scopes."""
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


class LoadModule(torch.nn.Module):
    def __init__(self, device, frozen=False):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.tensor([1., 2.], dtype=torch.float64, device=device),
            requires_grad=not frozen,
        )
        self.register_buffer("counter", torch.tensor([1], dtype=torch.int64, device=device))


def checkpoint(device):
    return {
        "weight": torch.tensor([4., 5.], dtype=torch.float64, device=device),
        "counter": torch.tensor([7], dtype=torch.int64, device=device),
    }


def check(tensor, values):
    torch.testing.assert_close(
        tensor, torch.tensor(values, dtype=tensor.dtype, device=tensor.device)
    )


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_load_preserves_aliases_identity_and_scope(device, frozen, grad_enabled):
    model = LoadModule(device, frozen)
    parameter = model.weight
    dtype, original_device = parameter.dtype, parameter.device
    exported = model.state_dict()
    kept = model.state_dict(keep_vars=True)
    detached = parameter.detach()
    state = checkpoint(device)
    before = {key: value.clone() for key, value in state.items()}
    outer_mode = torch.is_grad_enabled()
    with torch.set_grad_enabled(grad_enabled):
        result = model.load_state_dict(state)
        assert torch.is_grad_enabled() is grad_enabled
        assert result.missing_keys == []
        assert result.unexpected_keys == []
    assert torch.is_grad_enabled() is outer_mode
    assert model.weight is parameter
    assert kept["weight"] is parameter
    assert parameter.dtype == dtype
    assert parameter.device == original_device
    assert parameter.requires_grad is (not frozen)
    for alias in [parameter, exported["weight"], kept["weight"], detached]:
        check(alias, [4., 5.])
    check(exported["counter"], [7])
    check(kept["counter"], [7])
    assert not exported["weight"].requires_grad
    assert not detached.requires_grad
    for key in state:
        torch.testing.assert_close(state[key], before[key])
    # Loading must copy, rather than retaining the checkpoint's storage.
    with torch.no_grad():
        state["weight"].fill_(99)
    check(parameter, [4., 5.])
    if not frozen:
        with torch.enable_grad():
            (parameter * parameter).sum().backward()
        check(parameter.grad, [8., 10.])


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape"])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_load_errors_preserve_gradient_flags(device, failure, grad_enabled):
    model = LoadModule(device)
    parameter = model.weight
    detached = parameter.detach()
    state = checkpoint(device)
    if failure == "missing":
        del state["counter"]
    elif failure == "unexpected":
        state["extra"] = torch.tensor([0.], device=device)
    else:
        state["weight"] = torch.tensor([4., 5., 6.], dtype=torch.float64, device=device)
    before = {key: value.clone() for key, value in state.items()}
    outer_mode = torch.is_grad_enabled()
    with torch.set_grad_enabled(grad_enabled):
        with pytest.raises(RuntimeError):
            model.load_state_dict(state, strict=True)
        assert torch.is_grad_enabled() is grad_enabled
    assert torch.is_grad_enabled() is outer_mode
    assert model.weight is parameter
    assert parameter.requires_grad
    assert not detached.requires_grad
    for key in state:
        torch.testing.assert_close(state[key], before[key])
    # A strict error may follow partial copies: do not assert transactionality.


@pytest.mark.parametrize("operation", ["square", "linear"])
def test_load_obeys_saved_value_version_checks(device, operation):
    model = LoadModule(device)
    parameter = model.weight
    detached = parameter.detach()
    with torch.enable_grad():
        old_loss = (parameter * parameter if operation == "square" else parameter * 2).sum()
        model.load_state_dict(checkpoint(device))
        assert torch.is_grad_enabled()
        check(detached, [4., 5.])
        if operation == "square":
            with pytest.raises(RuntimeError):
                old_loss.backward()
        else:
            old_loss.backward()
            check(parameter.grad, [2., 2.])


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_native_copy_failure_restores_grad_scope(monkeypatch, grad_enabled):
    # Shim-only fault injection reaches an exception inside the copy call,
    # after all public checkpoint validation has already succeeded.
    from jittor.compat.torch.installers.nn import module_methods

    class CopyFailure(RuntimeError):
        pass

    model = LoadModule("cpu")
    original_mode = torch.is_grad_enabled()

    def fail_copy(module, state):
        assert not torch.is_grad_enabled()
        raise CopyFailure("injected native copy failure")

    with monkeypatch.context() as patch:
        patch.setattr(module_methods, "_ORIG_MODULE_LOAD_STATE_DICT", fail_copy)
        with torch.set_grad_enabled(grad_enabled):
            with pytest.raises(CopyFailure):
                model.load_state_dict(checkpoint("cpu"))
            assert torch.is_grad_enabled() is grad_enabled
    assert torch.is_grad_enabled() is original_mode
    assert model.weight.requires_grad
    with torch.enable_grad():
        (model.weight * 2).sum().backward()
    check(model.weight.grad, [2., 2.])
