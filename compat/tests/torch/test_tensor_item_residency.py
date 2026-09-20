"""Scalar reads preserve tensor placement and do not become differentiable."""
import numpy as np
import pytest
import torch

@pytest.fixture(autouse=True)
def require_requested_device(request):
    if request.node.callspec.params['device'] == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA device required')

def resident(t, device):
    assert t.device.type == device
    if hasattr(torch.optim, '_native_optimizer_module'):
        import jittor as jt
        t.sync()
        assert t.location() == ('device' if device == 'cuda' else 'cpu')
        backend, index = jt.core.dispatch_context([t])
        assert backend == device
        if device == 'cuda':
            assert index == torch.cuda.current_device() == t.device.index

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('shape', [(), (1,1)])
@pytest.mark.parametrize('dtype,value,pytype', [(torch.bool,True,bool),(torch.int64,2**40+3,int),(torch.float32,.2,float),(torch.float64,.123456789012345,float)])
def test_item_type_value_and_residency(device,shape,dtype,value,pytype):
    t=torch.tensor(value,dtype=dtype,device=device).reshape(shape)
    resident(t,device)
    result=t.item()
    assert type(result) is pytype
    expected=np.array(value,dtype=str(dtype).replace('torch.','')).item()
    assert result == expected
    resident(t,device)
    subsequent=t.clone()
    resident(subsequent,device)
    np.testing.assert_array_equal(subsequent.cpu().numpy(),np.full(shape,value,dtype=str(dtype).replace('torch.','')))
    resident(t,device)

@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('shape', [(0,), (2,)])
def test_item_requires_one_element(device,shape):
    t=torch.zeros(shape,device=device)
    with pytest.raises(RuntimeError):t.item()
    assert t.device.type==device

@pytest.mark.parametrize('device', ['cpu','cuda'])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_item_non_differentiable_read(device,dtype):
    t=torch.tensor(.25,dtype=dtype,device=device,requires_grad=True)
    assert t.item()==.25
    assert t.requires_grad and t.grad is None
    resident(t,device)
    (t*2).sum().backward()
    np.testing.assert_array_equal(t.grad.cpu().numpy(),np.array(2.,dtype=str(dtype).replace('torch.','')))
    resident(t,device)

@pytest.mark.parametrize('device', ['cpu','cuda'])
def test_item_view_and_shared_reference(device):
    base=torch.tensor([.125,.25],dtype=torch.float64,device=device)
    view=base[:1].reshape(1,1);alias=view
    resident(base,device);resident(view,device)
    assert view.item()==.125
    assert alias is view
    resident(base,device);resident(view,device);resident(alias,device)
    result=base+1
    resident(result,device)
    np.testing.assert_array_equal(result.cpu().numpy(),[1.125,1.25])
    resident(base,device);resident(view,device)
