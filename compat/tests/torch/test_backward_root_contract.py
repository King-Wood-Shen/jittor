"""A disconnected loss must fail before it can populate optimizer gradients."""
import numpy as np
import torch

from _helpers.device_types import instantiate_device_type_tests


class TestBackwardRootContract:
    def test_constant_and_detached_losses_reject_backward(self, device):
        parameter = torch.tensor([1., 2.], device=device, requires_grad=True)
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        for make_loss in (
                lambda: torch.tensor(0., device=device),
                lambda: parameter.new_zeros(()),
                lambda: parameter.new_zeros(()) * 0,
                lambda: (parameter * parameter).sum().detach()):
            for backward in (lambda loss: loss.backward(), torch.autograd.backward):
                optimizer.zero_grad(set_to_none=True)
                loss = make_loss()
                assert loss.device.type == device and not loss.requires_grad
                with self.assertRaisesRegex(RuntimeError, 'does not require grad.*grad_fn'):
                    backward(loss)
                assert parameter.grad is None
                assert not optimizer.state

    def test_new_factories_require_grad_only_when_requested(self, device):
        prototype = torch.ones((2,), device=device, requires_grad=True)
        factories = (
            lambda **kw: prototype.new_zeros((2,), **kw),
            lambda **kw: prototype.new_ones((2,), **kw),
            lambda **kw: prototype.new_empty((2,), **kw),
            lambda **kw: prototype.new_full((2,), 3., **kw),
            lambda **kw: prototype.new_tensor([1., 2.], **kw),
        )
        for factory in factories:
            default = factory()
            trainable = factory(requires_grad=True)
            assert default.device.type == trainable.device.type == device
            assert not default.requires_grad and default.is_leaf
            assert default.grad_fn is None
            assert trainable.requires_grad and trainable.is_leaf
        copied = prototype.new_tensor(prototype, requires_grad=True)
        copied.sum().backward()
        assert prototype.grad is None
        np.testing.assert_array_equal(copied.grad.cpu().numpy(), [1., 1.])

    def test_connected_zero_loss_and_trainable_leaf_remain_valid(self, device):
        parameter = torch.tensor([1., 2.], device=device, requires_grad=True)
        (parameter.square().sum() * 0).backward()
        assert parameter.grad.device.type == device
        np.testing.assert_array_equal(parameter.grad.cpu().numpy(), [0., 0.])
        leaf = torch.tensor(3., device=device, requires_grad=True)
        leaf.backward()
        assert leaf.grad.device.type == device
        assert leaf.grad.item() == 1.


instantiate_device_type_tests(TestBackwardRootContract, globals())
