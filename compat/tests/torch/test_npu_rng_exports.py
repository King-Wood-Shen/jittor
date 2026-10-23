"""Top-level NPU RNG state APIs share the native checkpoint implementation."""
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
@pytest.mark.parametrize("all_devices", [False, True])
def test_npu_rng_state_exports_restore_serialized_stream(tmp_path, all_devices):
    assert torch.npu.is_available()
    for name in ("get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all"):
        assert getattr(torch.npu, name) is getattr(torch.npu.random, name)

    torch.npu.manual_seed_all((1 << 33) + 17)
    prefix = torch.rand((17,), device="npu")
    torch.npu.synchronize()
    assert prefix.location() == "device"
    if all_devices:
        state = torch.npu.get_rng_state_all()
        assert len(state) == torch.npu.device_count()
    else:
        state = torch.npu.get_rng_state(0)
    path = tmp_path / "rng.pt"
    torch.save(state, path)
    expected = torch.rand((32,), device="npu")
    torch.npu.synchronize()
    state = torch.load(path, weights_only=False)
    if all_devices:
        torch.npu.set_rng_state_all(state)
    else:
        torch.npu.set_rng_state(state, 0)
    restored = torch.rand((32,), device="npu")
    torch.npu.synchronize()
    assert expected.location() == restored.location() == "device"
    assert torch.equal(expected, restored)
