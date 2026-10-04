import unittest
import torch


class StateMixin:
    def __init__(self, key="default"):
        self.mixin_key = key


class SkipModule(torch.nn.Module, StateMixin):
    def __init__(self, key):
        super(torch.nn.Module, self).__init__(key)


class OrdinaryModule(torch.nn.Module, StateMixin):
    def __init__(self):
        super().__init__()


class CooperativeModule(torch.nn.Module, StateMixin):
    call_super_init = True

    def __init__(self, key):
        super().__init__(key)


class TestModuleMixinInit(unittest.TestCase):
    def test_explicit_skip_reaches_mixin(self):
        self.assertEqual(SkipModule("adapter").mixin_key, "adapter")

    def test_default_stops_mixin_initialization(self):
        self.assertFalse(hasattr(OrdinaryModule(), "mixin_key"))

    def test_cooperative_opt_in(self):
        self.assertEqual(CooperativeModule("enabled").mixin_key, "enabled")

    def test_default_rejects_arguments(self):
        with self.assertRaises(TypeError):
            torch.nn.Module("unexpected")
