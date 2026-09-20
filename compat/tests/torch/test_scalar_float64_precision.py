"""Wrapped Python scalars keep double precision before native dispatch."""
import operator
import numpy as np
import pytest
import torch

VALUES = [.2, .03, .123456789012345]
OPS = [operator.add, operator.sub, operator.mul, operator.truediv, operator.pow]

@pytest.mark.parametrize('value', VALUES)
@pytest.mark.parametrize('op', OPS)
@pytest.mark.parametrize('reverse', [False, True])
def test_double_scalar_arithmetic(value, op, reverse):
    source = np.array([.123456789012345, .987654321098765, 1.23456789012345], dtype=np.float64)
    tensor = torch.from_numpy(source.copy())
    expected = op(value, source) if reverse else op(source, value)
    result = op(value, tensor) if reverse else op(tensor, value)
    assert result.dtype == torch.float64
    assert result.device.type == 'cpu'
    np.testing.assert_allclose(result.numpy(), expected, rtol=2e-15, atol=1e-16)

@pytest.mark.parametrize('value', VALUES)
@pytest.mark.parametrize('like', [False, True])
@pytest.mark.parametrize('shape', [(), (2, 3), (0,)])
def test_double_fill(value, like, shape):
    result = (torch.full_like(torch.empty(shape, dtype=torch.float64), value)
              if like else torch.full(shape, value, dtype=torch.float64))
    assert result.dtype == torch.float64
    assert result.device.type == 'cpu'
    np.testing.assert_array_equal(result.numpy(), np.full(shape, value, dtype=np.float64))

@pytest.mark.parametrize('value', VALUES)
@pytest.mark.parametrize('kind', ['mul', 'div'])
def test_double_value_argument(value, kind):
    p = torch.tensor([.123456789012345, .987654321098765], dtype=torch.float64)
    a = torch.tensor([.314159265358979, .271828182845904], dtype=torch.float64)
    b = torch.tensor([.161803398874989, .112358132134558], dtype=torch.float64)
    fn = p.addcmul if kind == 'mul' else p.addcdiv
    expected = p.numpy() + value * (a.numpy() * b.numpy() if kind == 'mul' else a.numpy() / b.numpy())
    np.testing.assert_allclose(fn(a, b, value=value).numpy(), expected, rtol=2e-15, atol=1e-16)

def test_double_factory_default_and_override():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        for kwargs in ({}, {'dtype': None}, {'dtype': torch.float64}):
            x = torch.full((2,), .2, **kwargs)
            assert x.dtype == torch.float64
            np.testing.assert_array_equal(x.numpy(), [.2, .2])
        source = torch.empty((2,), dtype=torch.float32)
        assert torch.full_like(source, .2, dtype=None).dtype == torch.float32
        x = torch.full_like(source, .2, dtype=torch.float64, device='cpu', requires_grad=True)
        assert x.requires_grad and x.device.type == 'cpu'
        np.testing.assert_array_equal(x.detach().numpy(), [.2, .2])
    finally:
        torch.set_default_dtype(old)

def test_double_full_out():
    out = torch.empty((2,), dtype=torch.float64)
    assert torch.full((2,), .2, out=out) is out
    np.testing.assert_array_equal(out.numpy(), [.2, .2])

@pytest.mark.parametrize('requires_grad', [False, True])
@pytest.mark.parametrize('dtype', [None, torch.float32, torch.float64])
def test_double_full_out_protocol(requires_grad, dtype):
    out = torch.empty((2,), dtype=torch.float64)
    if dtype == torch.float32:
        with pytest.raises(RuntimeError):
            torch.full(size=(2,), fill_value=.2, out=out, dtype=dtype, requires_grad=requires_grad)
        return
    result = torch.full(size=(2,), fill_value=.2, out=out, dtype=dtype, requires_grad=requires_grad)
    assert result is out
    assert result.requires_grad == requires_grad
    assert result.dtype == torch.float64
    np.testing.assert_array_equal(result.detach().numpy(), [.2, .2])

@pytest.mark.parametrize('call', ['full_keywords', 'full_value_keyword', 'like_keywords', 'like_value_keyword'])
def test_double_factory_keyword_equivalence(call):
    source = torch.tensor([1., 2.], dtype=torch.float64)
    if call == 'full_keywords':
        result = torch.full(size=(2,), fill_value=.2, dtype=torch.float64)
    elif call == 'full_value_keyword':
        result = torch.full((2,), fill_value=.2, dtype=torch.float64)
    elif call == 'like_keywords':
        result = torch.full_like(input=source, fill_value=.2)
    else:
        result = torch.full_like(source, fill_value=.2, dtype=None)
    assert result.dtype == source.dtype
    assert result.device == source.device
    np.testing.assert_array_equal(result.numpy(), [.2, .2])

@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
@pytest.mark.parametrize('reverse', [False, True])
def test_wrapped_scalar_preserves_promotion(dtype, reverse):
    x = torch.tensor([1, 1], dtype=dtype)
    for op in [operator.add, operator.mul, operator.truediv]:
        result = op(.03, x) if reverse else op(x, .03)
        assert result.dtype == torch.float32
        expected = op(np.float32(.03), np.ones(2, dtype=np.float32)) if reverse else op(np.ones(2, dtype=np.float32), np.float32(.03))
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-6, atol=1e-7)
