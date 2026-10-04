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


class TestLayerSuperInit(unittest.TestCase):
    def test_skip_public_linear_constructor(self):
        class Adapter(torch.nn.Linear):
            def __init__(self):
                super(torch.nn.Linear, self).__init__()
                self.adapter_weight = torch.nn.Parameter(torch.ones(2, device="cuda"))

        layer = Adapter()
        self.assertFalse(hasattr(layer, "in_features"))
        self.assertEqual([n for n, _ in layer.named_parameters()], ["adapter_weight"])
        self.assertEqual(layer.adapter_weight.device.type, "cuda")

    def test_normal_linear_constructor_and_forward(self):
        layer = torch.nn.Linear(3, 2).to("cuda")
        with torch.no_grad():
            layer.weight.fill_(1)
            layer.bias.fill_(2)
        out = layer(torch.ones((1, 3), device="cuda"))
        self.assertEqual(out.detach().cpu().tolist(), [[5, 5]])
        self.assertEqual(set(layer.state_dict()), {"weight", "bias"})


    def test_sequential_numeric_replacement(self):
        layer = torch.nn.Sequential(torch.nn.Linear(3, 2))
        replacement = torch.nn.Linear(3, 4)
        setattr(layer, "0", replacement)
        self.assertIs(layer[0], replacement)
        self.assertEqual(tuple(layer.state_dict()["0.weight"].shape), (4, 3))
