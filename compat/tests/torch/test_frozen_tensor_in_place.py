"""An in-place write leaves ``requires_grad`` where it was.

``requires_grad_(False)`` need not set the native stop-grad bit, and the
in-place paths read that bit alone to decide what to restore after assigning:
a frozen tensor written under ``no_grad`` -- ``copy_``, ``add_``, ``x.data =``,
``x.data[i] =`` -- came back trainable.
"""
import unittest

import torch


def _write(tensor, how):
    with torch.no_grad():
        if how == "copy_":
            tensor.copy_(torch.zeros(3, device=tensor.device))
        elif how == "add_":
            tensor.add_(1.0)
        elif how == "data":
            tensor.data = torch.zeros(3, device=tensor.device)
        else:
            tensor.data[0] = 5.0


class TestFrozenTensorInPlace(unittest.TestCase):
    def _check_data_iadd_dtype(self, device):
        delta = torch.tensor([0.000123], dtype=torch.float32, device=device)
        for make_parameter in (False, True):
            weight = torch.tensor([0.123456], dtype=torch.bfloat16, device=device)
            if make_parameter:
                weight = torch.nn.Parameter(weight)
            weight.data += delta
            self.assertEqual(weight.dtype, torch.bfloat16)
            self.assertEqual(float(weight.detach().float().cpu().numpy()[0]), 0.12353515625)

        # Explicit replacement has different Torch semantics from `data +=`.
        weight = torch.nn.Parameter(torch.tensor([0.123456], dtype=torch.bfloat16, device=device))
        weight.data = weight.data + delta
        self.assertEqual(weight.dtype, torch.float32)

    def _check(self, device):
        for how in ("copy_", "add_", "data", "data_item"):
            frozen = torch.ones(3, device=device, requires_grad=True)
            frozen.requires_grad_(False)
            _write(frozen, how)
            self.assertFalse(frozen.requires_grad, how)
            trainable = torch.ones(3, device=device, requires_grad=True)
            _write(trainable, how)
            self.assertTrue(trainable.requires_grad, how)

    def test_cpu(self):
        self._check("cpu")

    def test_data_iadd_dtype_cpu(self):
        self._check_data_iadd_dtype("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA found")
    def test_cuda(self):
        self._check("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "No CUDA found")
    def test_data_iadd_dtype_cuda(self):
        self._check_data_iadd_dtype("cuda")


if __name__ == "__main__":
    unittest.main()
