"""Torch archive interoperability requires a Torch-owned test process."""

import jittor as jt
import numpy as np
import pytest


def test_torch_archive_save_load_roundtrip(tmp_path):
    pytest.importorskip("torch")
    path = str(tmp_path / "weights.pth")
    source = {"weight": jt.array([[1.25, -2.0], [3.5, 4.0]]),
              "steps": jt.array([1, 3, 7]).int32()}
    jt.save(source, path)
    restored = jt.load(path)
    for name in source:
        assert restored[name].dtype == source[name].dtype
        np.testing.assert_array_equal(restored[name].numpy(), source[name].numpy())
