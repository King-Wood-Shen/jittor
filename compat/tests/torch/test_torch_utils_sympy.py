"""Private symbolic shape marker consumed by some Torch ecosystem packages."""

import pytest
import torch


def test_floor_div_marker_imports_without_claiming_symbolic_shape_support():
    from torch.utils._sympy.functions import FloorDiv

    assert torch.utils._sympy.functions.FloorDiv is FloorDiv
    assert isinstance(FloorDiv, type)
    with pytest.raises(NotImplementedError, match="symbolic shape arithmetic"):
        FloorDiv(8, 2)
