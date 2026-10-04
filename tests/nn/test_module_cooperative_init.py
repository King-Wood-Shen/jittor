import unittest
import jittor as jt


class TestModuleCooperativeInit(unittest.TestCase):
    def test_mixin_receives_constructor_arguments(self):
        class Mixin:
            def __init__(self, key, *, enabled):
                self.state = (key, enabled)

        class Layer(jt.Module, Mixin):
            def __init__(self):
                super().__init__("adapter", enabled=True)

        self.assertEqual(Layer().state, ("adapter", True))

    def test_object_boundary_remains_noop(self):
        # Native Module historically accepts and ignores constructor arguments.
        self.assertIsInstance(jt.Module("legacy", option=True), jt.Module)
