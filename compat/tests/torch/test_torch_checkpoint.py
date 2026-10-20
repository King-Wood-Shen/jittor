"""Activation checkpointing must recompute and preserve gradients/RNG state."""
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


def test_checkpoint_preserves_acl_dropout_rng_state():
    from jittor._runtime.fallback import forbid_backend_fallbacks
    import jittor as jt

    with jt.flag_scope(use_cuda=0, use_acl=1):
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
