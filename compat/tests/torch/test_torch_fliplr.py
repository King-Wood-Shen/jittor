import unittest
import numpy as np
import torch


class TestTorchFliplr(unittest.TestCase):
    def test_cuda_values_and_gradient(self):
        for shape in ((3, 4), (2, 3, 4)):
            with self.subTest(shape=shape):
                count=int(np.prod(shape))
                x=torch.arange(count,device="cuda",dtype=torch.float32).reshape(shape).requires_grad_()
                y=torch.fliplr(x)
                self.assertEqual(y.device.type,"cuda")
                np.testing.assert_array_equal(y.detach().cpu().numpy(),np.flip(np.arange(count).reshape(shape),1))
                weights=torch.arange(count,device="cuda",dtype=torch.float32).reshape(shape)
                (y*weights).sum().backward()
                self.assertEqual(x.grad.device.type,"cuda")
                np.testing.assert_array_equal(x.grad.cpu().numpy(),np.flip(np.arange(count).reshape(shape),1))

    def test_minimum_rank(self):
        with self.assertRaises(RuntimeError):
            torch.fliplr(torch.ones(3,device="cuda"))

    def test_noncontiguous_and_copy(self):
        x=torch.arange(12,device="cuda").reshape(3,4).transpose(0,1)
        original=x.cpu().numpy().copy()
        y=torch.fliplr(x)
        np.testing.assert_array_equal(y.cpu().numpy(),np.flip(original,1))
        y[0,0]=-1
        np.testing.assert_array_equal(x.cpu().numpy(),original)
