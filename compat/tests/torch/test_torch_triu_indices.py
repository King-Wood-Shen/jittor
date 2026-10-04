import unittest
import numpy as np
import torch


class TestTriuIndices(unittest.TestCase):
    def test_coordinates(self):
        for device in ("cpu", "cuda"):
            for dtype in (torch.int32, torch.int64):
                for row, col, offset in ((3, 5, 0), (5, 3, 1), (4, 4, -2), (3, 4, 7), (0, 3, 0), (3, 0, 0)):
                    with self.subTest(device=device, dtype=dtype, shape=(row,col), offset=offset):
                        x=torch.triu_indices(row,col,offset,dtype=dtype,device=device)
                        expected=np.asarray([(i,j) for i in range(row) for j in range(col) if j-i>=offset],dtype=np.int64).reshape(-1,2).T
                        self.assertEqual(x.device.type,device)
                        self.assertEqual(x.dtype,dtype)
                        np.testing.assert_array_equal(x.cpu().numpy(),expected)

    def test_default_and_invalid(self):
        self.assertEqual(torch.triu_indices(2,2).dtype,torch.int64)
        for row,col in ((-1,2),(2,-1)):
            with self.assertRaises(RuntimeError):
                torch.triu_indices(row,col)
        with self.assertRaises(RuntimeError):
            torch.triu_indices(2,2,dtype=torch.float32)

    def test_cuda_index_assignment(self):
        rows,cols=torch.triu_indices(4,4,1,device="cuda")
        matrix=torch.zeros(4,4,device="cuda")
        matrix[rows,cols]=1
        np.testing.assert_array_equal(matrix.cpu().numpy(),np.triu(np.ones((4,4)),1))

    def test_cpu_tuple_indices_on_cuda_with_gradient(self):
        rows,cols=torch.triu_indices(4,4,1,device="cpu")
        values=torch.ones(2,6,device="cuda",requires_grad=True)
        matrix=torch.zeros(2,4,4,device="cuda")
        matrix[:,rows,cols]=values
        selected=matrix[:,rows,cols]
        self.assertEqual(selected.device.type,"cuda")
        np.testing.assert_array_equal(selected.detach().cpu().numpy(),np.ones((2,6)))
        selected.sum().backward()
        self.assertEqual(values.grad.device.type,"cuda")
        np.testing.assert_array_equal(values.grad.cpu().numpy(),np.ones((2,6)))
        self.assertEqual(rows.device.type,"cpu")
