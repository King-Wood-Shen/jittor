import unittest
import torch

class TestInitFan(unittest.TestCase):
    def test_tensor_fans(self):
        for device in ("cpu", "cuda"):
            for shape, expected in [((8, 896), (896, 8)), ((6, 4, 3, 5), (60, 90)), ((2, 3, 0), (0, 0)), ((0, 7), (7, 0))]:
                with self.subTest(device=device, shape=shape):
                    tensor = torch.empty(shape, device=device)
                    actual = torch.nn.init._calculate_fan_in_and_fan_out(tensor)
                    self.assertEqual(actual, expected)
                    self.assertTrue(all(isinstance(v, int) for v in actual))
                    self.assertEqual(torch.nn.init._calculate_correct_fan(tensor, "FAN_IN"), expected[0])
                    self.assertEqual(torch.nn.init._calculate_correct_fan(tensor, "fan_out"), expected[1])

    def test_invalid_rank_and_mode(self):
        for device in ("cpu", "cuda"):
            for shape in ((), (4,)):
                with self.assertRaises(ValueError):
                    torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(shape, device=device))
            with self.assertRaises(ValueError):
                torch.nn.init._calculate_correct_fan(torch.empty((2, 3), device=device), "invalid")
