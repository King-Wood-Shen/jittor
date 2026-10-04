import unittest
import numpy as np
import torch

class AxisLogSoftmax(unittest.TestCase):
 def test_forward_backward_axes(self):
  for shape,axis in (((2,151936,6),1),((17,2,3),0),((2,31,3),-2),((3,1,2),1)):
   with self.subTest(shape=shape,axis=axis):
    size=int(np.prod(shape))
    raw=(np.sin(np.arange(size,dtype=np.float64)*.017)*8).reshape(shape).astype(np.float32)
    weights=(np.cos(np.arange(size,dtype=np.float64)*.013)*.1).reshape(shape).astype(np.float32)
    x=torch.tensor(raw,device="cuda",requires_grad=True)
    g=torch.tensor(weights,device="cuda")
    y=torch.nn.functional.log_softmax(x,dim=axis)
    dx,=torch.autograd.grad(y,x,g)
    self.assertEqual(y.device.type,"cuda");self.assertEqual(dx.device.type,"cuda")
    a=raw.astype(np.float64);shift=a-a.max(axis,keepdims=True)
    expected=shift-np.log(np.exp(shift).sum(axis,keepdims=True))
    gradient=weights-np.exp(expected)*weights.sum(axis,keepdims=True,dtype=np.float64)
    np.testing.assert_allclose(y.detach().cpu().numpy(),expected,rtol=1e-5,atol=1e-5)
    np.testing.assert_allclose(dx.cpu().numpy(),gradient,rtol=2e-4,atol=2e-5)
 def test_mask_and_nonfinite(self):
  for raw in (
   np.array([[[1.,2.],[-np.inf,3.],[4.,-np.inf]]],dtype=np.float32),
   np.full((1,3,2),-np.inf,dtype=np.float32),
   np.full((1,3,2),np.inf,dtype=np.float32),
   np.full((1,3,2),np.nan,dtype=np.float32),
  ):
   x=torch.tensor(raw,device="cuda")
   y=torch.nn.functional.log_softmax(x,dim=1)
   with np.errstate(invalid="ignore",divide="ignore"):
    shift=raw.astype(np.float64)-raw.max(1,keepdims=True)
    expected=shift-np.log(np.exp(shift).sum(1,keepdims=True))
   np.testing.assert_allclose(y.cpu().numpy(),expected,rtol=1e-5,atol=1e-5,equal_nan=True)
 def test_empty(self):
  for shape in ((0,3,2),(2,0,3)):
   x=torch.empty(shape,device="cuda")
   y=torch.nn.functional.log_softmax(x,dim=1)
   self.assertEqual(tuple(y.shape),shape)
if __name__=="__main__":unittest.main()
