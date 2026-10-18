"""``torch.npu`` answers about the ACL runtime, not with placeholders.

Every device namespace the CUDA installer creates used to share the same
stubs, so on a machine with eight 910Bs ``torch.npu.is_available()`` was
False and ``device_count()`` was 0. A library that asks before choosing a
device believed it and went to the host.

Both halves are asserted here: on a CPU-only process the facade still refuses
rather than inventing a device, and on ACL it selects, computes and reports
the device that was selected.
"""

import os

import numpy as np
import pytest
import torch

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks


def _on_acl():
    return bool(getattr(jt.compiler, "has_acl", 0)) and bool(jt.flags.use_cuda)


def test_npu_namespace_refuses_rather_than_inventing_a_device():
    if _on_acl():
        pytest.skip("this is the CPU half; the ACL half is the next case")
    assert torch.npu.is_available() is False
    assert torch.npu.device_count() == 0
    for operation in (torch.npu.current_device,
                      lambda: torch.npu.set_device(0),
                      torch.npu.synchronize,
                      torch.npu.empty_cache):
        with pytest.raises(RuntimeError, match="ACL"):
            operation()


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_discovery_selection_and_synchronization():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    count = torch.npu.device_count()
    assert torch.npu.is_available() and count > 0
    assert torch.npu.is_bf16_supported() is True

    original = torch.npu.current_device()
    try:
        # Every spelling of a device torch accepts, including the bare type
        # and None, which both mean "the current one".
        for argument in (0, "npu:%d" % (count - 1), torch.device("npu:0"),
                         None, torch.device("npu")):
            torch.npu.set_device(argument)
            selected = torch.npu.current_device()
            x = torch.tensor([selected + 1.0, selected + 2.0],
                             dtype=torch.float32, device="npu")
            y = x.square() + 3
            torch.npu.synchronize(torch.device("npu", selected))
            assert torch.npu.current_device() == selected
            assert x.device.type == y.device.type == "npu"
            assert y.cpu().tolist() == [(selected + 1.0) ** 2 + 3,
                                        (selected + 2.0) ** 2 + 3]
    finally:
        torch.npu.set_device(original)

    # An ordinal past the end is a mistake, not a device.
    for operation in (torch.npu.set_device, torch.npu.synchronize):
        with pytest.raises(ValueError, match="out of range"):
            operation(count)
    with pytest.raises(TypeError, match="bool"):
        torch.npu.set_device(True)
    assert torch.npu.current_device() == original


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_device_scope_restores_the_previous_device():
    original = torch.npu.current_device()
    other = (original + 1) % torch.npu.device_count()
    with torch.npu.device(other):
        assert torch.npu.current_device() == other
    assert torch.npu.current_device() == original


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_seed_apis_seed_acl_streams_and_cuda_seed_is_inactive():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    seed = 1729
    cpu_state = torch.get_rng_state().numpy().copy()
    with forbid_backend_fallbacks():
        torch.npu.manual_seed_all(seed)
        first = torch.rand((19,), dtype=torch.float32, device="npu").cpu().numpy()
        torch.npu.manual_seed_all(seed)
        # Transformers set_seed calls this before torch.npu.manual_seed_all.
        assert torch.cuda.manual_seed_all(seed + 1) is None
        second = torch.rand((19,), dtype=torch.float32, device="npu").cpu().numpy()
        torch.npu.manual_seed(seed)
        third = torch.rand((19,), dtype=torch.float32, device="npu").cpu().numpy()
        torch.npu.manual_seed(seed)
        fourth = torch.rand((19,), dtype=torch.float32, device="npu").cpu().numpy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(third, fourth)
    assert np.array_equal(torch.get_rng_state().numpy(), cpu_state)


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_random_state_round_trips_the_acl_stream():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    before = torch.npu.random.get_rng_state()
    try:
        torch.npu.manual_seed_all(2718)
        torch.rand((7,), dtype=torch.float32, device="npu")
        checkpoint_state = torch.npu.random.get_rng_state()
        expected = torch.rand((19,), dtype=torch.float32, device="npu").cpu().numpy()
        torch.npu.random.set_rng_state(checkpoint_state)
        actual = torch.rand((19,), dtype=torch.float32, device="npu").cpu().numpy()
        np.testing.assert_array_equal(actual, expected)

        states = torch.npu.random.get_rng_state_all()
        assert len(states) == torch.npu.device_count()
        torch.npu.random.set_rng_state_all(states)
        assert torch.npu.random.initial_seed() >= 0
    finally:
        torch.npu.random.set_rng_state(before)


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_torch_npu_primitives_match_their_definitions():
    import torch_npu

    assert torch_npu.__jittor_acl_facade__ is True
    assert torch_npu.npu is torch.npu

    hidden = torch.tensor([[1.0, -2.0, 3.0, -4.0]], device="npu")
    weight = torch.tensor([0.5, 1.5, 2.5, 3.5], device="npu")
    normalized, auxiliary = torch_npu.npu_rms_norm(hidden, weight, 1e-6)
    assert auxiliary is None
    reference = hidden * torch.rsqrt((hidden * hidden).mean(-1, keepdim=True) + 1e-6) * weight
    assert torch.allclose(normalized, reference, atol=1e-5)

    cos = torch.full((1, 4), 0.6, device="npu")
    sin = torch.full((1, 4), 0.8, device="npu")
    rotated = torch_npu.npu_rotary_mul(hidden, cos, sin)
    half = hidden.shape[-1] // 2
    expected = hidden * cos + torch.cat((-hidden[..., half:], hidden[..., :half]), -1) * sin
    assert torch.allclose(rotated, expected, atol=1e-5)

    paired = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="npu")
    gate, value = paired.chunk(2, dim=-1)
    assert torch.allclose(torch_npu.npu_swiglu(paired),
                          gate * torch.sigmoid(gate) * value, atol=1e-5)


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_torch_npu_fusion_attention_bsnd_acl_graph_forward_backward():
    import torch_npu

    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    batch, sequence, heads, dim = 1, 5, 2, 8
    q = torch.randn((batch, sequence, heads, dim), dtype=torch.bfloat16,
                    device="npu", requires_grad=True)
    k = torch.randn((batch, sequence, heads, dim), dtype=torch.bfloat16,
                    device="npu", requires_grad=True)
    v = torch.randn((batch, sequence, heads, dim), dtype=torch.bfloat16,
                    device="npu", requires_grad=True)
    blocked = torch.triu(torch.ones((sequence, sequence), dtype=torch.bool,
                                    device="npu"), diagonal=1)

    output = torch_npu.npu_fusion_attention(
        q, k, v, heads, "BSND", atten_mask=blocked, scale=dim ** -0.5,
        sparse_mode=3)[0]
    allowed = ~blocked
    expected = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=allowed, scale=dim ** -0.5)
    expected = expected.transpose(1, 2)
    torch.npu.synchronize()
    assert output.shape == q.shape
    assert output.dtype == q.dtype
    assert torch.allclose(output, expected, atol=2e-2, rtol=2e-2)
    output.float().sum().backward()
    torch.npu.synchronize()
    assert all(tensor.grad is not None
               and bool(torch.isfinite(tensor.grad).all().item())
               for tensor in (q, k, v))

    # TND is the variable-length layout emitted by padding-free packing. Check
    # separate packed segments against the same per-segment SDPA definition.
    qt = torch.randn((5, heads, dim), dtype=torch.bfloat16,
                     device="npu", requires_grad=True)
    kt = torch.randn((5, heads, dim), dtype=torch.bfloat16,
                     device="npu", requires_grad=True)
    vt = torch.randn((5, heads, dim), dtype=torch.bfloat16,
                     device="npu", requires_grad=True)
    blocked_t = torch.triu(torch.ones((5, 5), dtype=torch.bool,
                                      device="npu"), diagonal=1)
    output_t = torch_npu.npu_fusion_attention(
        qt, kt, vt, heads, "TND", atten_mask=blocked_t, scale=dim ** -0.5,
        actual_seq_qlen=(2, 5), actual_seq_kvlen=(2, 5), sparse_mode=3)[0]
    expected_t = torch.cat([
        torch.nn.functional.scaled_dot_product_attention(
            qt[start:end].unsqueeze(0).transpose(1, 2),
            kt[start:end].unsqueeze(0).transpose(1, 2),
            vt[start:end].unsqueeze(0).transpose(1, 2),
            attn_mask=~blocked_t[:end-start, :end-start],
            scale=dim ** -0.5).transpose(1, 2).squeeze(0)
        for start, end in ((0, 2), (2, 5))
    ])
    torch.npu.synchronize()
    assert output_t.shape == qt.shape
    assert torch.allclose(output_t, expected_t, atol=2e-2, rtol=2e-2)
    output_t.float().sum().backward()
    torch.npu.synchronize()
    assert all(tensor.grad is not None
               and bool(torch.isfinite(tensor.grad).all().item())
               for tensor in (qt, kt, vt))


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_unsupported_npu_apis_refuse_instead_of_reporting_zero():
    for name in ("memory_allocated", "max_memory_allocated", "memory_stats",
                 "get_device_properties", "current_stream", "mem_get_info"):
        with pytest.raises(NotImplementedError):
            getattr(torch.npu, name)()


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_manual_seed_apis_reset_acl_generators():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    seed = (1 << 32) + 29
    torch.npu.manual_seed_all(seed)
    first = torch.rand((16,), device="npu")
    torch.npu.manual_seed_all(seed)
    second = torch.rand((16,), device="npu")
    torch.npu.synchronize()
    assert torch.equal(first, second)

    torch.npu.manual_seed(seed + 1)
    third = torch.rand((16,), device="npu")
    torch.npu.manual_seed(seed + 1)
    fourth = torch.rand((16,), device="npu")
    torch.npu.synchronize()
    assert torch.equal(third, fourth)

    torch.npu.manual_seed_all(seed + 2)
    torch.rand((5,), device="npu")
    state = torch.npu.random.get_rng_state()
    expected = torch.rand((16,), device="npu")
    torch.npu.random.set_rng_state(state)
    restored = torch.rand((16,), device="npu")
    torch.npu.synchronize()
    assert torch.equal(expected, restored)


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_fp16_autocast_matmul_same_dtype_inputs():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    inputs = torch.ones((2, 4), dtype=torch.float32, device="npu",
                        requires_grad=True)
    weight = torch.eye(4, dtype=torch.float32, device="npu",
                       requires_grad=True)
    with torch.autocast("npu", dtype=torch.float16):
        output = torch.nn.functional.linear(inputs, weight)
    assert output.dtype == torch.float16
    output.float().sum().backward()
    torch.npu.synchronize()
    assert inputs.grad.dtype == weight.grad.dtype == torch.float32
@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_npu_class_forward_precedes_generic_rms_norm_fastpath():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    previous_policy = jt.flags.backend_fallback
    jt.flags.backend_fallback = "error"
    fallback_start = jt.core.backend_fallback_count()

    class ContractRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.full((8,), 0.5, dtype=torch.bfloat16, device="npu"))
            self.variance_epsilon = 1e-6

        def forward(self, hidden):
            return hidden + self.weight

    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(
        device="npu", dtype=torch.bfloat16)
    expected = hidden + torch.full((8,), 0.5, dtype=torch.bfloat16, device="npu")
    module = ContractRMSNorm()
    try:
        with torch.no_grad():
            # First call is eligible for the C++ module-call shortcut.
            actual = module(hidden)
            torch.npu.synchronize()
            assert torch.equal(actual, expected)

            # A hook disables that shortcut and exercises the Python dispatch.
            module.register_forward_hook(lambda _module, _args, output: output)
            hooked = module(hidden)
            torch.npu.synchronize()
            assert torch.equal(hooked, expected)
        assert jt.flags.backend_fallback == "error"
        assert jt.core.backend_fallback_count() == fallback_start == 0
    finally:
        jt.flags.backend_fallback = previous_policy


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_bf16_single_token_sdpa_acl_hook_uses_flash_score_for_matching_heads():
    assert _on_acl(), "JITTOR_TEST_DEVICES=npu but ACL is not active"
    from unittest import mock
    from backends.acl.kernels.ops import flashattention_op as acl_sdpa

    rng = np.random.RandomState(5102)
    q = torch.tensor(rng.normal(0, 0.2, (1, 16, 1, 128)).astype(np.float32),
                     device="npu", dtype=torch.bfloat16)
    k = torch.tensor(rng.normal(0, 0.2, (1, 16, 29, 128)).astype(np.float32),
                     device="npu", dtype=torch.bfloat16)
    v = torch.tensor(rng.normal(0, 0.2, (1, 16, 29, 128)).astype(np.float32),
                     device="npu", dtype=torch.bfloat16)
    original_flash = acl_sdpa.FlashAttentionACL
    original_incremental = acl_sdpa.IncreFlashAttentionACL
    with mock.patch.object(acl_sdpa, "FlashAttentionACL", wraps=original_flash) as flash, \
            mock.patch.object(acl_sdpa, "IncreFlashAttentionACL",
                              wraps=original_incremental) as incremental, \
            jt.no_grad():
        output = acl_sdpa.scaled_dot_product_attention_acl(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
            scale=0.08838834764831845)
        output.sync()

    assert flash.call_count == 1
    assert incremental.call_count == 0
    assert acl_sdpa.scaled_dot_product_attention_acl.backend_name == "acl_flash_attention_score_v2"
    assert tuple(output.shape) == tuple(q.shape)
    assert str(output.dtype).split(".")[-1] == "bfloat16"
    assert np.isfinite(output.float().cpu().numpy()).all()
    assert jt.core.backend_fallback_count() == 0
