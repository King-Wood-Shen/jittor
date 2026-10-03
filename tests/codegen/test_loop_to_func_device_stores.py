"""Retained scalar outputs must be written on the selected device."""
import unittest

import jittor as jt
import numpy as np
from _helpers import capability


class TestLoopToFuncDeviceStores(unittest.TestCase):
    def check_retained_scalars(self, use_cuda):
        with jt.flag_scope(use_cuda=use_cuda):
            left = jt.array(1.25)
            right = jt.array(.5)
            output = left.broadcast((7,)) * right.broadcast((7,))
            np.testing.assert_array_equal(output.numpy(), np.full(7, .625))
            # The scalars remain externally visible after their producer is
            # fused into the broadcast. Both outputs must be materialized.
            self.assertEqual(left.item(), 1.25)
            self.assertEqual(right.item(), .5)
            second = left.broadcast((11,)) + right.broadcast((11,))
            np.testing.assert_array_equal(second.numpy(), np.full(11, 1.75))

    def test_retained_scalars_cpu(self):
        self.check_retained_scalars(0)

    @unittest.skipIf(
        not capability.check_accelerator("cuda", backend=jt).enabled,
        "No CUDA found")
    def test_retained_scalars_cuda(self):
        self.check_retained_scalars(1)


if __name__ == "__main__":
    unittest.main()
