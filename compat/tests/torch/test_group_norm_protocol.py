from contextlib import ExitStack
import numpy as np
import pytest
import torch

@pytest.mark.parametrize('device',['cpu','cuda'])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('affine',[True,False])
def test_group_norm_constructor_preserves_dtype_device_and_values(device,dtype,affine):
    with ExitStack() as stack:
        if hasattr(torch,'_torch_compat_install_context'):
            import jittor as jt
            stack.enter_context(jt.runtime.scope(use_cuda=int(device=='cuda'),backend_fallback='error'))
        norm=torch.nn.GroupNorm(2,4,eps=1e-5,affine=affine,device=device,dtype=dtype)
        assert norm.num_groups==2 and norm.num_channels==4
        if affine:
            assert norm.weight.dtype==norm.bias.dtype==dtype
            assert norm.weight.device.type==norm.bias.device.type==device
            assert norm.weight.requires_grad and norm.bias.requires_grad
            np.testing.assert_array_equal(norm.weight.detach().cpu().numpy(),np.ones(4))
            np.testing.assert_array_equal(norm.bias.detach().cpu().numpy(),np.zeros(4))
        else:
            assert norm.weight is None and norm.bias is None
            assert not list(norm.parameters()) and not norm.state_dict()
        data=np.arange(72,dtype='float64' if dtype==torch.float64 else 'float32').reshape(2,4,3,3)/7
        x=torch.tensor(data,device=device,dtype=dtype)
        with torch.no_grad(): output=norm(x)
        assert output.device.type==device and output.dtype==dtype
        if hasattr(torch,'_torch_compat_install_context'):
            jt.sync([output]); backend,index=jt.core.dispatch_context([output])
            assert backend==device
            assert output.location()==('cpu' if device=='cpu' else 'device')
        grouped=data.reshape(2,2,-1)
        expected=((grouped-grouped.mean(-1,keepdims=True))/np.sqrt(grouped.var(-1,keepdims=True)+1e-5)).reshape(data.shape)
        np.testing.assert_allclose(output.detach().cpu().numpy(),expected,rtol=2e-5,atol=2e-5)

def test_group_norm_subclass_accepts_default_factory_kwargs():
    class GroupNorm1(torch.nn.GroupNorm):
        def __init__(self,channels,**kwargs): super().__init__(1,channels,**kwargs)
    norm=GroupNorm1(4,device=None,dtype=None)
    assert len(list(norm.parameters()))==2

def test_group_norm_rejects_indivisible_channels():
    with pytest.raises(ValueError): torch.nn.GroupNorm(3,4)

@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_group_norm_uses_default_dtype(dtype):
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        norm=torch.nn.GroupNorm(2,4)
        assert norm.weight.dtype==norm.bias.dtype==dtype
    finally:
        torch.set_default_dtype(previous)
