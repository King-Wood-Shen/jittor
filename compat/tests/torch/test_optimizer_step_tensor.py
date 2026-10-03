"""Adam step metadata is a persistent, writable tensor."""
import unittest
import jittor as jt
import torch
from _helpers import capability

class TestOptimizerStepTensor(unittest.TestCase):
    def test_step_identity_inplace_write_and_restore(self):
        devices = [("cpu", 0)]
        if capability.any_accelerator_enabled(backend=jt):
            devices.append(("cuda", 1))
        for device, use_cuda in devices:
            with jt.flag_scope(use_cuda=use_cuda):
                value = torch.tensor([1., -2.], requires_grad=True, device=device)
                optimizer = torch.optim.AdamW([value], foreach=False, fused=False)
                value.grad = torch.ones_like(value)
                optimizer.step()
                step = optimizer.state[value]["step"]
                self.assertIsInstance(step, torch.Tensor)
                self.assertEqual(step.dtype, torch.float32)
                self.assertEqual(step.device.type, "cpu")
                self.assertIs(step, optimizer.state_dict()["state"][0]["step"])
                step.add_(3)
                optimizer.step()
                self.assertIs(step, optimizer.state[value]["step"])
                self.assertEqual(step.item(), 5)
                restored = torch.optim.AdamW([value], foreach=False, fused=False)
                restored.load_state_dict(optimizer.state_dict())
                self.assertEqual(restored.state[value]["step"].item(), 5)
                value.grad = torch.ones_like(value)
                restored.step()
                self.assertEqual(restored.state[value]["step"].item(), 6)
