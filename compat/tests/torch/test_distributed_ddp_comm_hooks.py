"""The default Accelerate DDP path imports Torch comm hooks unconditionally."""

import pytest

import torch
from jittor.compat import stub_policy


def test_default_ddp_comm_hook_import_surface_refuses_hook_execution():
    from torch.distributed.algorithms.ddp_comm_hooks import (
        default_hooks,
        powerSGD_hook,
    )

    assert torch.distributed.algorithms.ddp_comm_hooks.default_hooks is default_hooks
    assert torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook is powerSGD_hook
    assert callable(default_hooks.fp16_compress_hook)
    assert callable(default_hooks.bf16_compress_hook)
    assert callable(default_hooks.fp16_compress_wrapper)
    assert callable(default_hooks.bf16_compress_wrapper)
    assert callable(powerSGD_hook.powerSGD_hook)
    assert callable(powerSGD_hook.batched_powerSGD_hook)

    previous = stub_policy.set_allow_stub(False)
    try:
        with pytest.raises(NotImplementedError, match="cannot execute"):
            default_hooks.fp16_compress_hook()
        with pytest.raises(NotImplementedError, match="cannot execute"):
            powerSGD_hook.powerSGD_hook()
        with pytest.raises(NotImplementedError, match="cannot execute"):
            powerSGD_hook.PowerSGDState()
    finally:
        stub_policy.set_allow_stub(previous)
