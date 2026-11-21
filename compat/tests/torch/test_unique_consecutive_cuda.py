"""CUDA placement regression from Qwen2.5-VL visual window indexing."""

import unittest

import jittor as jt
import torch


class TestUniqueConsecutiveCuda(unittest.TestCase):
    def test_cuda_input_stays_cuda_under_host_caller_request(self):
        if jt.flags.use_cuda != 1:
            self.skipTest("requires real CUDA")
        values = torch.tensor([0, 2, 2, 4], device="cuda", dtype=torch.int32)
        result = torch.unique_consecutive(values)
        self.assertEqual(str(result.device), "cuda:0")
        self.assertEqual(result.numpy().tolist(), [0, 2, 4])

    def test_explicit_cpu_input_stays_cpu_in_cuda_process(self):
        if jt.flags.use_cuda != 1:
            self.skipTest("requires real CUDA")
        values = torch.tensor([0, 2, 2, 4], device="cpu", dtype=torch.int32)
        result = torch.unique_consecutive(values)
        self.assertEqual(str(result.device), "cpu")
        self.assertEqual(result.numpy().tolist(), [0, 2, 4])


if __name__ == "__main__":
    unittest.main()
