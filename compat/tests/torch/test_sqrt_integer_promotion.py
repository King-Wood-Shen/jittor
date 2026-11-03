"""Torch-compatible dtype promotion for square root of integer tensors."""


def test_integer_sqrt_uses_default_float32_and_preserves_float_inputs():
    import torch

    dim = torch.tensor(39 * 896)
    root = torch.sqrt(dim)
    magnitude = 5.0 / root

    assert root.dtype == torch.float32
    assert magnitude.dtype == torch.float32
    assert abs(root.item() - 186.9331512451172) < 1e-5
    assert abs(magnitude.item() - 0.026747528463602066) < 1e-8

    assert torch.sqrt(torch.tensor(4.0, dtype=torch.float32)).dtype == torch.float32
    assert torch.sqrt(torch.tensor(4.0, dtype=torch.float64)).dtype == torch.float64
