"""CPU training may call CUDA seeding without initializing a CUDA generator."""

import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


@pytest.mark.parametrize("api", ["manual_seed", "manual_seed_all"])
@pytest.mark.parametrize("seed", [7, -1, (1 << 64) - 1, 1 << 80,
                                  -(1 << 80), "7", 1.5, True])
def test_cuda_seed_without_visible_devices_preserves_cpu_state(monkeypatch, api, seed):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with jt.runtime.scope(use_cuda=0):
        before = torch.get_rng_state().numpy().copy()
        assert not torch.cuda.is_available()
        assert torch.cuda.device_count() == 0
        assert not torch.cuda.is_initialized()
        assert getattr(torch.cuda, api)(seed) is None
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)
        assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("api", ["manual_seed", "manual_seed_all"])
@pytest.mark.parametrize("seed,exception", [("invalid", ValueError), (None, TypeError)])
def test_cuda_seed_still_converts_arguments_without_devices(monkeypatch, api, seed, exception):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with jt.runtime.scope(use_cuda=0):
        before = torch.get_rng_state().numpy().copy()
        with pytest.raises(exception):
            getattr(torch.cuda, api)(seed)
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)


@pytest.mark.parametrize("api", ["manual_seed", "manual_seed_all"])
@pytest.mark.parametrize("seed", [7, -1, 1 << 63, (1 << 64) - 1])
def test_cuda_seed_rewinds_real_default_stream(api, seed):
    if not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")
    with jt.runtime.scope(use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        cpu_before = torch.get_rng_state().numpy().copy()
        assert getattr(torch.cuda, api)(seed) is None
        assert torch.cuda.initial_seed() == seed & ((1 << 64) - 1)
        first = torch.rand(19, device="cuda")
        assert first.device.type == "cuda"
        first = first.detach().cpu().numpy().copy()
        torch.rand(11, device="cuda").sum().item()
        getattr(torch.cuda, api)(seed)
        again = torch.rand(19, device="cuda")
        assert again.device.type == "cuda"
        np.testing.assert_array_equal(again.detach().cpu().numpy(), first)
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), cpu_before)


def test_cuda_seed_facade_uses_acl_counter_stream():
    if getattr(getattr(jt.compiler, "build_config", None), "backend", None) != "acl":
        pytest.skip("requires an ACL build")
    with jt.runtime.scope(use_acl=1, use_cuda=1, backend_fallback="error"), forbid_backend_fallbacks():
        assert torch.cuda.device_count() > 0
        host_before = torch.get_rng_state().numpy().copy()
        torch.cuda.manual_seed_all(1729)
        assert jt.core.rng_initial_seed("acl", jt.core.current_device()) == 1729
        first = torch.rand(8, device="npu")
        assert first.device.type == "npu"
        first = first.detach().cpu().numpy().copy()
        torch.cuda.manual_seed_all(1729)
        again = torch.rand(8, device="npu")
        np.testing.assert_array_equal(again.detach().cpu().numpy(), first)
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), host_before)
