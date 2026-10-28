"""Portable optimizer FQNs follow model checkpoint hooks across wrapper layouts."""
import os
from types import SimpleNamespace

import jittor as jt
import numpy as np
import pytest
import torch
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.compat.fsdp2._state_dict import _shard_optimizer_state_dict
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict,
    set_model_state_dict, set_optimizer_state_dict,
)


def _device():
    return os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]


@pytest.fixture(autouse=True)
def checkpoint_device():
    device = _device()
    assert device in ("cpu", "cuda")
    if device == "cuda":
        assert torch.cuda.is_available()
    with jt.flag_scope(use_cuda=int(device == "cuda")), forbid_backend_fallbacks():
        yield
        jt.sync_all(True)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2, device=_device())


def _layout(block, layout, *, wrap):
    if layout == "literal_child":
        module = torch.nn.Module()
        module.add_module("_checkpoint_wrapped_module", block)
        return module, "_checkpoint_wrapped_module.linear.weight"
    body = checkpoint_wrapper(block) if wrap else block
    if layout == "nested_wrapper":
        return torch.nn.Sequential(body), "0.linear.weight"
    return body, "linear.weight"


def _snapshot(value):
    assert jt.core.dispatch_context([value])[0] == _device()
    return value.detach().cpu().numpy().copy()


@pytest.mark.parametrize("layout", ["root_wrapper", "nested_wrapper", "literal_child"])
def test_full_optimizer_checkpoint_uses_exported_names_and_resumes_without_wrapper(layout):
    block = _Block()
    model, name = _layout(block, layout, wrap=True)
    parameter = block.linear.weight
    shard_value = parameter.detach().reshape(-1).clone().requires_grad_(True)
    optimizer = torch.optim.SGD([shard_value], lr=.1, momentum=.9)
    optimizer.state[shard_value] = {"momentum_buffer": torch.full_like(shard_value, .5)}
    entry = SimpleNamespace(
        name="weight", owner=block.linear, attr="weight", shard=shard_value,
        shape=tuple(parameter.shape), numel=parameter.numel(),
        padded_numel=parameter.numel(), shard_numel=parameter.numel(),
        requires_grad=True, full_param=None)
    state = SimpleNamespace(
        true_fsdp_initialized=True, true_fsdp_params=[entry], true_fsdp_flat=False,
        true_fsdp_rank=0, true_fsdp_unsharded=False, frontend_type=torch.Tensor,
        shard_group=SimpleNamespace(ranks=(0,), rank=lambda: 0, size=lambda: 1))
    object.__setattr__(block.linear, "_is_fsdp_module", True)
    object.__setattr__(block.linear, "_fsdp_state", state)
    object.__setattr__(block.linear, "weight", shard_value)
    object.__setattr__(shard_value, "_jittor_fsdp2_state", state)
    object.__setattr__(shard_value, "_jittor_fsdp2_entry", entry)

    options = StateDictOptions(full_state_dict=True)
    model_state = get_model_state_dict(model, options=options)
    assert name in model_state
    saved = get_optimizer_state_dict(model, optimizer, options=options)
    assert set(saved["state"]) == {name}
    assert saved["param_groups"][0]["params"] == [name]
    assert tuple(saved["state"][name]["momentum_buffer"].shape) == tuple(parameter.shape)
    local = _shard_optimizer_state_dict(model, saved)
    assert set(local["state"]) == {name}
    assert tuple(local["state"][name]["momentum_buffer"].shape) == tuple(shard_value.shape)
    np.testing.assert_array_equal(
        _snapshot(local["state"][name]["momentum_buffer"]),
        np.full(tuple(shard_value.shape), .5, dtype=np.float32))

    target_block = _Block()
    target, target_name = _layout(target_block, layout, wrap=False)
    assert target_name == name
    target_parameter = target_block.linear.weight
    restored = torch.optim.SGD([target_parameter], lr=.9, momentum=.9)
    set_model_state_dict(target, model_state, options=options)
    set_optimizer_state_dict(target, restored, saved, options=options)
    assert restored.param_groups[0]["lr"] == .1
    np.testing.assert_array_equal(
        _snapshot(restored.state[target_parameter]["momentum_buffer"]),
        np.full(tuple(parameter.shape), .5, dtype=np.float32))
    before = _snapshot(target_parameter)
    target_parameter.grad = torch.full_like(target_parameter, 2.)
    restored.step()
    np.testing.assert_allclose(
        _snapshot(target_parameter), before - .1 * (.9 * .5 + 2.), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        _snapshot(saved["state"][name]["momentum_buffer"]),
        np.full(tuple(parameter.shape), .5, dtype=np.float32))
