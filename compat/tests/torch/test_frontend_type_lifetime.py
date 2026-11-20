"""Native caches keyed by a frontend type do not keep the type alive.

The module-call, tensor-frontend policy, result-type and dtype caches used to
hold every type they had seen, so a class made at run time -- a Module defined
in a function, a Parameter subclass -- lived for the rest of the process with
everything its body closed over. They now drop an entry when its type is
collected, before the address can be reused by another type.
"""
import gc
import unittest
import weakref

import numpy as np
import torch


def _collected(make):
    ref = make()
    gc.collect()
    return ref() is None


class TestFrontendTypeLifetime(unittest.TestCase):
    def test_make_subclass_keeps_type_leaf_and_device(self):
        class Quantized(torch.nn.Parameter):
            pass

        source = torch.tensor([1.0, 2.0], device="cuda")
        value = torch.Tensor._make_subclass(Quantized, source, False)
        self.assertIs(type(value), Quantized)
        self.assertTrue(value.is_leaf)
        self.assertFalse(value.requires_grad)
        self.assertEqual(value.device.type, "cuda")
        np.testing.assert_array_equal(value.cpu().numpy(), [1.0, 2.0])
        with self.assertRaises(TypeError):
            torch.Tensor._make_subclass(Quantized, [1.0, 2.0])

    def test_parameter_subclass_new_consumes_metadata_keywords(self):
        class Quantized(torch.nn.Parameter):
            def __new__(cls, data, requires_grad=False, compress_statistics=True):
                value = torch.Tensor._make_subclass(cls, data, requires_grad)
                value.compress_statistics = compress_statistics
                return value

        value = Quantized(torch.tensor([3.0], device="cuda"), compress_statistics=False)
        self.assertIs(type(value), Quantized)
        self.assertFalse(value.compress_statistics)
        self.assertFalse(value.requires_grad)
        self.assertTrue(value.is_leaf)
        self.assertEqual(value.device.type, "cuda")

    def test_module_to_calls_parameter_subclass_conversion(self):
        class Converted(torch.nn.Parameter):
            def to(self, *args, **kwargs):
                self.convert_count = getattr(self, "convert_count", 0) + 1
                return super().to(*args, **kwargs)

        class Holder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = Converted(torch.tensor([1.0]))

        module = Holder()
        original = module.weight
        module.to("cuda:0")
        self.assertIs(module.weight, original)
        self.assertEqual(original.convert_count, 1)
        self.assertEqual(module.weight.device.type, "cuda")

    def test_module_defined_in_a_function_is_collected(self):
        def make():
            offset = torch.ones(16)

            class Local(torch.nn.Module):
                def forward(self, x):
                    return x + offset

            self.assertEqual(Local()(torch.ones(16)).sum().item(), 32.0)
            return weakref.ref(Local)

        self.assertTrue(_collected(make))

    def test_parameter_subclass_is_collected(self):
        def make():
            class Local(torch.nn.Parameter):
                pass

            p = Local(torch.ones(3))
            self.assertEqual(p.dtype, torch.float32)
            self.assertEqual((p * 2).sum().item(), 6.0)
            return weakref.ref(Local)

        self.assertTrue(_collected(make))

    def test_a_new_type_never_sees_a_collected_one_s_entry(self):
        # Alternate classes whose dispatch differs: one overrides forward over
        # the builtin's execute, one does not. Collected each round, so a new
        # class may well land on the address of the previous one.
        x = torch.tensor([-1.0, 2.0])
        for i in range(40):
            if i % 2:
                class Act(torch.nn.ReLU):
                    def forward(self, x):
                        return x * 3.0
                want = [-3.0, 6.0]
            else:
                class Act(torch.nn.ReLU):
                    pass
                want = [0.0, 2.0]
            np.testing.assert_array_equal(Act()(x).numpy(), np.array(want, "float32"), str(i))
            del Act
            gc.collect()


if __name__ == "__main__":
    unittest.main()
