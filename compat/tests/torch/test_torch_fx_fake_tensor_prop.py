"""Import-only FX fake tensor propagation marker used by optional compilers."""

import pytest
import torch


def test_fake_tensor_prop_marker_refuses_execution():
    import torch.fx as fx
    from torch.fx.experimental.symbolic_shapes import ShapeEnv
    from torch.fx.passes.fake_tensor_prop import FakeTensorProp

    assert fx.passes.fake_tensor_prop.FakeTensorProp is FakeTensorProp
    with pytest.raises(NotImplementedError, match="not implemented by Jittor compat"):
        FakeTensorProp(None)

    assert fx.experimental.symbolic_shapes.ShapeEnv is ShapeEnv
    with pytest.raises(NotImplementedError, match="not implemented by Jittor compat"):
        ShapeEnv()
