import unittest
import numpy as np
import torch


class TestFactoryLeaf(unittest.TestCase):
    def test_factory_gradient_after_dtype_conversion(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for name in ("zeros", "ones", "full", "rand", "randn"):
                with self.subTest(dtype=dtype, factory=name):
                    args = ((2, 3), 2.0) if name == "full" else ((2, 3),)
                    x = getattr(torch, name)(*args, dtype=dtype, device="cuda", requires_grad=True)
                    self.assertTrue(x.is_leaf)
                    self.assertEqual(x.device.type, "cuda")
                    weights = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")
                    (x.float() * weights).sum().backward()
                    self.assertIsNotNone(x.grad)
                    self.assertEqual(x.grad.device.type, "cuda")
                    self.assertEqual(x.grad.dtype, dtype)
                    np.testing.assert_array_equal(x.grad.float().cpu().numpy(), weights.cpu().numpy())

    def test_like_factory_does_not_inherit_source_graph(self):
        source = torch.ones(2, 3, device="cuda", requires_grad=True)
        x = torch.zeros_like(source, dtype=torch.bfloat16, requires_grad=True)
        self.assertTrue(x.is_leaf)
        x.float().sum().backward()
        self.assertIsNone(source.grad)
        np.testing.assert_array_equal(x.grad.float().cpu().numpy(), np.ones((2, 3)))

    def test_generator_factory_is_leaf(self):
        generator = torch.Generator(device="cpu").manual_seed(42)
        x = torch.randn(2, 3, generator=generator, dtype=torch.bfloat16, requires_grad=True)
        self.assertTrue(x.is_leaf)
        x.float().sum().backward()
        np.testing.assert_array_equal(x.grad.float().numpy(), np.ones((2, 3)))
