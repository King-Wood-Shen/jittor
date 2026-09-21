import pytest
import numpy as np
import torch

@pytest.fixture(params=["cpu","cuda"])
def device(request):
 if request.param=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA unavailable")
 return request.param

def check_resident(t,device):
 assert t.device.type==device
 if hasattr(t,"location"):
  t.sync();assert t.location()==("device" if device=="cuda" else "cpu")

@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
@pytest.mark.parametrize("container",[list,tuple])
@pytest.mark.parametrize("weight",[0.,.2,.8,1.,-.3,True])
def test_foreach_lerp_scalar_protocol(device,dtype,container,weight):
 a=container([torch.tensor([[.2,-.7],[1.,2.]],dtype=dtype,device=device),torch.tensor([3.,-2.],dtype=dtype,device=device)])
 b=container([torch.tensor([[1.,.4],[-.2,3.]],dtype=dtype,device=device),torch.tensor([1.,4.],dtype=dtype,device=device)])
 av=[x.detach().cpu().numpy().copy() for x in a];bv=[x.detach().cpu().numpy().copy() for x in b]
 result=torch._foreach_lerp_(a,b,weight=weight)
 assert result is a
 for i,(x,y) in enumerate(zip(a,b)):
  assert result[i] is x
  check_resident(x,device);check_resident(y,device)
  tol=dict(rtol=3e-6,atol=3e-7) if dtype==torch.float32 else dict(rtol=1e-10,atol=1e-12)
  np.testing.assert_allclose(x.detach().cpu().numpy(),av[i]+weight*(bv[i]-av[i]),**tol)
  np.testing.assert_array_equal(y.detach().cpu().numpy(),bv[i])

def test_foreach_lerp_position_same_pair(device):
 x=torch.tensor([1.,2.],device=device);before=x.detach().cpu().numpy().copy()
 a=[x];assert torch._foreach_lerp_(a,[x],.2) is a
 np.testing.assert_array_equal(x.detach().cpu().numpy(),before)

@pytest.mark.parametrize("kind",["empty","length","shape","dtype","integer","leaf"])
def test_foreach_lerp_errors(device,kind):
 a=[torch.ones(2,device=device)];b=[torch.zeros(2,device=device)]
 if kind=="empty":a=[];b=[]
 if kind=="length":b=[]
 if kind=="shape":b=[torch.ones(3,device=device)]
 if kind=="dtype":b=[torch.ones(2,dtype=torch.float64,device=device)]
 if kind=="integer":a=[torch.ones(2,dtype=torch.int64,device=device)];b=[torch.ones(2,dtype=torch.int64,device=device)]
 if kind=="leaf":a=[torch.ones(2,device=device,requires_grad=True)]
 with pytest.raises(RuntimeError):torch._foreach_lerp_(a,b,.2)



def test_new_tensor_preserves_float64_scalar(device):
    target = torch.empty(1, dtype=torch.float64, device=device)
    scalar = target.new_tensor(0.2)
    assert scalar.dtype == torch.float64 and scalar.device == target.device
    check_resident(scalar, device)
    assert scalar.item() == 0.2
