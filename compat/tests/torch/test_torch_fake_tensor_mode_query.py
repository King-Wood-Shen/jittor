"""Torch fake tensor mode query over Jittor-backed tensor values."""

import torch


def test_maybe_get_fake_mode_returns_none_for_jittor_values():
    import jittor as jt
    from torch._subclasses.fake_tensor import maybe_get_fake_mode

    assert maybe_get_fake_mode(jt.array([1.0, 2.0])) is None
    assert maybe_get_fake_mode(object()) is None
