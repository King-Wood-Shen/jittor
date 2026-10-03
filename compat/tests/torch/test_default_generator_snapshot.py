"""Default generator copies own a CPU stream at the serialized position."""
import pickle
import unittest

import numpy as np
import torch


def draw(count, generator=None, normal=False):
    value = (torch.randn if normal else torch.rand)(
        count, device="cpu", generator=generator)
    return value.numpy().copy()


class TestDefaultGeneratorSnapshot(unittest.TestCase):
    def test_explicit_default_advances_without_reseeding(self):
        torch.manual_seed(321)
        expected = [draw(8), draw(8)]
        torch.manual_seed(321)
        for reference in expected:
            np.testing.assert_array_equal(draw(8, torch.default_generator), reference)

    def test_pickle_resumes_mixed_draws_without_advancing_global(self):
        torch.manual_seed(719)
        draw(7)
        blob = pickle.dumps(torch.default_generator)
        expected = [draw(11), draw(5, normal=True)]
        draw(13)
        before = torch.get_rng_state().numpy().copy()
        restored = pickle.loads(blob)
        self.assertIsNot(restored, torch.default_generator)
        np.testing.assert_array_equal(draw(11, restored), expected[0])
        np.testing.assert_array_equal(draw(5, restored, normal=True), expected[1])
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)
        # Serializing the copy retains its own advanced position too.
        clone = pickle.loads(pickle.dumps(restored))
        np.testing.assert_array_equal(draw(9, restored), draw(9, clone))

    def test_private_state_roundtrip_and_invalid_state_are_atomic(self):
        restored = pickle.loads(pickle.dumps(torch.default_generator))
        restored.manual_seed(27)
        before = torch.get_rng_state().numpy().copy()
        draw(4, restored)
        state = restored.get_state()
        expected = draw(6, restored)
        restored.set_state(state)
        np.testing.assert_array_equal(draw(6, restored), expected)
        own_before = restored.get_state().numpy().copy()
        with self.assertRaises(RuntimeError):
            restored.set_state(torch.tensor([0, 1, 2], dtype=torch.uint8))
        np.testing.assert_array_equal(restored.get_state().numpy(), own_before)
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)

    def test_cpu_generator_seed_does_not_reset_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        torch.manual_seed(89)
        torch.rand(5, device="cuda").cpu().numpy()
        state = torch.cuda.get_rng_state().cpu().numpy().copy()
        restored = pickle.loads(pickle.dumps(torch.default_generator))
        before = torch.get_rng_state().numpy().copy()
        restored.manual_seed(92)
        np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)
        torch.default_generator.manual_seed(93)
        np.testing.assert_array_equal(torch.cuda.get_rng_state().cpu().numpy(), state)

    def test_cpu_generator_rejects_cuda_draws(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        for generator in (torch.default_generator,
                          pickle.loads(pickle.dumps(torch.default_generator))):
            with self.assertRaises(RuntimeError):
                torch.rand(2, device="cuda", generator=generator)

    def test_sampling_factories_replay_the_saved_stream(self):
        probabilities = torch.tensor([.2, .4, .8], device="cpu")
        calls = {
            "integers": lambda g: torch.randint(0, 100, (9,), device="cpu", generator=g),
            "permutation": lambda g: torch.randperm(13, device="cpu", generator=g),
            "normal": lambda g: torch.normal(.3, .8, size=(9,), device="cpu", generator=g),
            "bernoulli": lambda g: torch.bernoulli(probabilities, generator=g),
            "multinomial": lambda g: torch.multinomial(
                probabilities, 7, replacement=True, generator=g),
        }
        for name, call in calls.items():
            with self.subTest(name=name):
                torch.manual_seed(613)
                draw(7)
                blob = pickle.dumps(torch.default_generator)
                expected = call(None).numpy().copy()
                draw(11)
                before = torch.get_rng_state().numpy().copy()
                actual = call(pickle.loads(blob)).numpy().copy()
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(torch.get_rng_state().numpy(), before)
