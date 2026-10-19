"""Import-only FX interpreter marker used by DeepSpeed's optional compiler."""

import pytest
import torch


def test_fx_interpreter_marker_is_explicitly_non_executable():
    import torch.fx as fx
    from torch.fx import Interpreter

    assert fx.Interpreter is Interpreter
    assert torch.fx.Interpreter is Interpreter
    with pytest.raises(NotImplementedError, match="not implemented by Jittor compat"):
        Interpreter(None)
