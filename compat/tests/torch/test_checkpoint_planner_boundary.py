"""The public planner exposes only its verified same-world storage scope."""

import importlib

import pytest
import torch

from jittor.compat.torch.fidelity import Fidelity, fidelity_of


@pytest.mark.parametrize("name", ["DefaultSavePlanner", "DefaultLoadPlanner"])
def test_default_planner_import_has_same_world_scope(name):
    module = importlib.import_module("torch.distributed.checkpoint.default_planner")
    assert torch.distributed.checkpoint.default_planner is module
    planner = getattr(module, name)
    record = fidelity_of("torch.distributed.checkpoint.default_planner." + name)
    assert record.level is Fidelity.APPROXIMATE
    assert record.implementation is planner
    assert isinstance(planner(), planner)
    with pytest.raises(NotImplementedError, match="custom DCP"):
        planner(custom=True)
