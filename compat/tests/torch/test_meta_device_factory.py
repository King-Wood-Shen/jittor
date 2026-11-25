"""A ``device="meta"`` factory argument is observable on the result.

Transformers' meta-state-dict path asks for metadata-only tensors through the
factory argument as well as through ``with torch.device("meta")``. Jittor has
no meta backend, so the result is really allocated on the host and reports
``meta`` through the same mark the context uses; it must not silently claim a
CPU identity.
"""

import pytest
import torch


def test_factory_meta_argument_reports_meta():
    value = torch.empty((2, 3), device="meta")
    assert value.device.type == "meta"
    assert value.is_meta
    assert tuple(value.shape) == (2, 3)


def test_meta_factory_result_does_not_leak_into_later_factories():
    assert torch.zeros(4, device="meta").device.type == "meta"
    assert torch.zeros(4).device.type == "cpu"


def test_fp8_parameter_data_retains_meta_and_validates_normal_arguments():
    value = torch.nn.Parameter(torch.empty(2, device="meta", dtype=torch.float8_e4m3fn))
    alias = value.data
    assert value.is_meta and alias.is_meta
    assert alias.normal_(std=0.02) is alias
    with pytest.raises(RuntimeError, match="std >= 0.0"):
        alias.normal_(std=-1)

    real = torch.empty(2, device="cpu", dtype=torch.float8_e4m3fn)
    with pytest.raises(RuntimeError, match="supported floating dtype"):
        real.normal_()


def test_to_meta_creates_a_separate_typed_placeholder():
    real = torch.ones((2, 3), device="cpu")
    placeholder = real.to(torch.device("meta"), dtype=torch.float8_e4m3fn)
    assert real.is_cpu
    assert placeholder.is_meta
    assert placeholder.dtype is torch.float8_e4m3fn
    assert tuple(placeholder.shape) == (2, 3)
    assert placeholder.to("meta") is placeholder


def test_torch_layer_and_parameter_assignment_honor_registration_hook(monkeypatch):
    original = torch.nn.Module.register_parameter

    def register_empty_parameter(module, name, parameter):
        original(module, name, parameter)
        if parameter is not None:
            attributes = dict(parameter.__dict__, requires_grad=parameter.requires_grad)
            module._parameters[name] = torch.nn.Parameter(
                parameter.to(torch.device("meta")), **attributes)

    monkeypatch.setattr(torch.nn.Module, "register_parameter", register_empty_parameter)
    layer = torch.nn.Linear(4, 3)
    assert layer.weight.is_meta and layer.bias.is_meta
    layer.weight = torch.nn.Parameter(torch.empty(3, 4, dtype=torch.float8_e4m3fn))
    assert layer.weight.is_meta and layer.weight.dtype is torch.float8_e4m3fn


def test_parameter_replacement_uses_source_placement_not_old_attributes():
    source = torch.ones((2, 3), device="cpu")
    old = torch.nn.Parameter(source.to("meta"))
    assert old.is_meta
    replacement = type(old)(source, **old.__dict__)
    assert replacement.is_cpu and not replacement.is_meta


def test_assign_state_replaces_meta_placement_with_checkpoint_placement():
    layer = torch.nn.Linear(4, 3, bias=False)
    real = torch.ones_like(layer.weight)
    layer.weight = torch.nn.Parameter(layer.weight.to("meta"))
    assert layer.weight.is_meta
    layer.load_state_dict({"weight": real}, assign=True)
    assert layer.weight.is_cpu and not layer.weight.is_meta
    torch.testing.assert_close(layer.weight, real)
