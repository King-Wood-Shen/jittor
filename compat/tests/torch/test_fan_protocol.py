import pytest
import torch

@pytest.mark.parametrize('shape,expected',[
    ((5,3),(3,5)),((4,2,3),(6,12)),((4,2,3,3),(18,36)),
    ((4,2,2,3,5),(60,120)),((0,3,2),(6,0)),((4,0,3),(0,12))])
def test_public_fan_helper_returns_python_integer_counts(shape,expected):
    tensor=torch.empty(shape)
    actual=torch.nn.init._calculate_fan_in_and_fan_out(tensor)
    assert all(type(value) is int for value in actual)
    assert actual==expected
    for mode,value in zip(['fan_in','fan_out'],expected):
        assert torch.nn.init._calculate_correct_fan(tensor,mode)==value

@pytest.mark.parametrize('shape',[(),(3,)])
def test_fan_requires_at_least_two_dimensions(shape):
    with pytest.raises(ValueError): torch.nn.init._calculate_fan_in_and_fan_out(torch.empty(shape))

def test_native_shape_helper_keeps_its_signature():
    if not hasattr(torch,'_torch_compat_install_context'): pytest.skip('native isolation applies to shim')
    import jittor as jt
    assert jt.init._calculate_fan_in_and_fan_out((4,2,3,3))==(18,36)
