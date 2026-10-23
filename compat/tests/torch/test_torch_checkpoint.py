"""Activation checkpointing must recompute and preserve gradients/RNG state."""
import functools
import os

import pytest
import torch


class _CheckpointLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.25, -0.5], [0.75, 0.125]]))
        self.dropout = torch.nn.Dropout(p=0.25)
        self.forward_calls = 0

    def forward(self, value):
        self.forward_calls += 1
        return self.dropout(value @ self.weight).tanh()


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
@pytest.mark.parametrize("callable_form", ["module", "bound", "partial"])
def test_checkpoint_module_callable_preserves_parameter_gradients(callable_form):
    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks

    assert jt.flags.use_acl and jt.flags.use_cuda

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(
                [[0.25, -0.5], [0.75, 0.125]], device="npu"))
            self.calls = 0

        def forward(self, value):
            self.calls += 1
            return (value @ self.weight).tanh()

    plain, checked = Layer(), Layer()
    x = torch.tensor([[0.2, 0.4], [-0.1, 0.8]], device="npu", requires_grad=True)
    checked_x = x.detach().clone().requires_grad_(True)
    fn = {"module": checked, "bound": checked.forward,
          "partial": functools.partial(checked)}[callable_form]
    before = checked.weight.detach().clone()
    with forbid_backend_fallbacks():
        expected = plain(x)
        expected.square().sum().backward()
        actual = torch.utils.checkpoint.checkpoint(fn, checked_x, use_reentrant=False)
        actual.square().sum().backward()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(checked_x.grad, x.grad)
        torch.testing.assert_close(checked.weight.grad, plain.weight.grad)
        torch.optim.SGD(checked.parameters(), lr=0.1).step()
        torch.npu.synchronize()
        assert not torch.equal(checked.weight, before)
        assert checked.calls == 2
        assert checked.weight.device.type == "npu"
        assert jt.core.backend_fallback_count() == 0


def test_checkpoint_recomputes_and_matches_input_and_parameter_gradients():
    from jittor._runtime.fallback import forbid_backend_fallbacks
    import jittor as jt

    # Keep this semantic contract on CPU even when the host also has an NPU/GPU.
    with jt.flag_scope(use_cuda=0, use_acl=0):
        plain = _CheckpointLayer()
        checked = _CheckpointLayer()
        checked.load_state_dict(plain.state_dict())
        plain_input = torch.tensor([[0.2, 0.4], [-0.1, 0.8]], requires_grad=True)
        checked_input = plain_input.detach().clone().requires_grad_(True)

        with forbid_backend_fallbacks():
            torch.manual_seed(913)
            plain_output = plain(plain_input)
            plain_output.square().sum().backward()

            torch.manual_seed(913)
            checked_output = torch.utils.checkpoint.checkpoint(
                checked.__call__, checked_input, use_reentrant=True)
            checked_rng_after_forward = torch.get_rng_state().numpy().copy()
            checked_output.square().sum().backward()
            checked_rng_after_backward = torch.get_rng_state().numpy().copy()

        torch.testing.assert_close(checked_output, plain_output, rtol=0, atol=0)
        torch.testing.assert_close(checked_input.grad, plain_input.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(checked.weight.grad, plain.weight.grad, rtol=1e-6, atol=1e-7)
        assert checked.forward_calls == 2
        assert (checked_rng_after_backward == checked_rng_after_forward).all()
        assert jt.core.backend_fallback_count() == 0


@pytest.mark.skipif(os.environ.get("JITTOR_TEST_DEVICES") != "npu",
                    reason="requires a selected ACL device")
def test_checkpoint_preserves_acl_dropout_rng_state():
    from jittor._runtime.fallback import forbid_backend_fallbacks
    import jittor as jt

    with jt.flag_scope(use_cuda=1, use_acl=1):
        assert jt.flags.use_cuda and jt.flags.use_acl
        plain = _CheckpointLayer()
        checked = _CheckpointLayer()
        checked.load_state_dict(plain.state_dict())
        plain_input = torch.tensor([[0.2, 0.4], [-0.1, 0.8]], requires_grad=True)
        checked_input = plain_input.detach().clone().requires_grad_(True)

        with forbid_backend_fallbacks():
            torch.npu.manual_seed(913)
            plain_output = plain(plain_input)
            plain_output.square().sum().backward()

            torch.npu.manual_seed(913)
            checked_output = torch.utils.checkpoint.checkpoint(
                checked.__call__, checked_input, use_reentrant=True)
            rng_after_forward = jt.core.rng_state("acl", int(jt.current_device()))
            checked_output.square().sum().backward()
            rng_after_backward = jt.core.rng_state("acl", int(jt.current_device()))

        torch.testing.assert_close(checked_output, plain_output, rtol=0, atol=0)
        torch.testing.assert_close(checked_input.grad, plain_input.grad, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(checked.weight.grad, plain.weight.grad, rtol=1e-6, atol=1e-7)
        assert checked.forward_calls == 2
        assert rng_after_backward == rng_after_forward
        assert jt.core.backend_fallback_count() == 0


@pytest.mark.skipif(
    os.environ.get("JITTOR_TEST_DEVICES") != "npu", reason="requires a selected ACL device"
)
def test_acl_sharded_checkpoint_accepts_torch_dtype_alias(tmp_path):
    import jittor as jt
    from jittor._runtime.fallback import forbid_backend_fallbacks
    from jittor.distributed import checkpoint

    from jittor.serialization.sharded import TensorShard

    with jt.flag_scope(use_cuda=1, use_acl=1), forbid_backend_fallbacks():
        value = torch.tensor([0.25, -0.5, 0.75, 1.0], device="npu")
        assert str(value.dtype).startswith("torch.")
        assert jt.core.dispatch_context([value])[0] == "acl"
        shard = TensorShard(value, tuple(value.shape), 0)
        checkpoint_path = tmp_path / "checkpoint"
        checkpoint.save({"weight": shard}, checkpoint_path)
        restored = checkpoint.load(
            checkpoint_path,
            tensor_shards={
                "/weight": TensorShard(torch.zeros_like(value), tuple(value.shape), 0)
            },
        )["weight"]
        torch.testing.assert_close(restored.tensor, value)
        assert jt.core.backend_fallback_count() == 0
