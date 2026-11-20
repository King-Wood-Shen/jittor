"""Empty matrix products must not reach a nonempty GEMM kernel."""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability


class TestEmptyMatmul(unittest.TestCase):
    def check_empty(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            a = jt.ones((3, 2), dtype="float32")
            b = jt.zeros((2, 0), dtype="float32")
            result = jt.matmul(a, b)
            self.assertEqual(tuple(result.shape), (3, 0))
            self.assertEqual(str(result.dtype), "float32")
            self.assertEqual(result.numpy().shape, (3, 0))

            batched_a = jt.ones((2, 3, 2), dtype="float32")
            batched_b = jt.zeros((2, 2, 0), dtype="float32")
            batched = jt.matmul(batched_a, batched_b)
            self.assertEqual(tuple(batched.shape), (2, 3, 0))
            self.assertEqual(batched.numpy().shape, (2, 3, 0))

            control = jt.matmul(jt.ones((2, 3)), jt.ones((3, 4)))
            np.testing.assert_array_equal(control.numpy(), np.full((2, 4), 3.0))
            if use_cuda:
                self.assertIn("cuda", str(a.device))
                jt.sync_all(True)

    def test_cpu(self):
        self.check_empty(0)

    @unittest.skipUnless(capability.check_accelerator("cuda", backend=jt).enabled,
                         "CUDA is required")
    def test_cuda(self):
        self.check_empty(1)
