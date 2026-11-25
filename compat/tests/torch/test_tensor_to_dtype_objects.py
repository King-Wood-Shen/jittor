"""`.to()` has to accept a torch dtype object, not only a string.

torch spells a dtype as an object. Code written against torch hands it straight
to `.to()`, and `transformers/image_processing_backends.py:327` does exactly
that for every image it preprocesses::

    images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)

`_dtype_spec` accepted only `NanoString`, callables and strings, so both the
positional and the keyword form raised, and the failure surfaced from inside
transformers as `TypeError: to() expected dtype to be a dtype spelling`. Found
by running MiniMax-H3 reference-to-video, whose Qwen2-VL image processor takes
that path for each reference image.
"""
import unittest

import jittor as jt
import torch


class TestToAcceptsTorchDtypes(unittest.TestCase):
    def setUp(self):
        self.x = jt.ones((2, 2))

    def test_positional_dtype_object(self):
        self.assertEqual(str(self.x.to(torch.float32).dtype), "float32")

    def test_keyword_dtype_object(self):
        self.assertEqual(str(self.x.to(dtype=torch.float32).dtype), "float32")

    def test_half_precision_dtype_objects(self):
        for spelling, expected in ((torch.float16, "float16"),
                                   (torch.bfloat16, "bfloat16")):
            with self.subTest(str(spelling)):
                self.assertEqual(str(self.x.to(dtype=spelling).dtype), expected)

    def test_integer_dtype_objects(self):
        self.assertEqual(str(self.x.to(dtype=torch.int64).dtype), "int64")

    def test_string_spelling_still_works(self):
        self.assertEqual(str(self.x.to(dtype="float32").dtype), "float32")

    def test_a_device_object_is_still_a_device(self):
        # `torch.device` has no `.name`, which is what keeps the dtype branch
        # from swallowing it; assert that rather than trusting it.
        moved = self.x.to(torch.device("cpu"))
        self.assertEqual(str(moved.dtype), "float32")

    def test_a_mistyped_device_is_still_rejected(self):
        with self.assertRaises(TypeError):
            self.x.to("gpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_keyword_cuda_device_index_with_dtype(self):
        source = torch.tensor([1., -2.], device="cpu")
        device_index = torch.cuda.current_device()
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                result = source.to(device=device_index, dtype=dtype)
                self.assertTrue(result.is_cuda)
                self.assertEqual(result.get_device(), device_index)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(result.float().cpu().tolist(), [1., -2.])

    def test_check_with_validates_and_defers_message(self):
        def message():
            self.fail("successful checks must skip the message")

        torch._check_with(ValueError, True, message)
        with self.assertRaisesRegex(ValueError, "image mismatch"):
            torch._check_with(ValueError, False, lambda: "image mismatch")
        with self.assertRaisesRegex(ValueError, "^42$"):
            torch._check_with(ValueError, False, lambda: 42)
        with self.assertRaisesRegex(ValueError, "Expected cond to be True"):
            torch._check_with(ValueError, False, None)
        with self.assertRaisesRegex(TypeError, "cond must be a bool"):
            torch._check_with(ValueError, 1, message)
        with self.assertRaisesRegex(TypeError, "message must be a callable"):
            torch._check_with(ValueError, False, "image mismatch")
        with self.assertRaisesRegex(AssertionError, "subclass of Exception"):
            torch._check_with(UserWarning, False, message)

    def test_dtype_and_device_together(self):
        out = self.x.to(device="cpu", dtype=torch.float16)
        self.assertEqual(str(out.dtype), "float16")


if __name__ == "__main__":
    unittest.main()
