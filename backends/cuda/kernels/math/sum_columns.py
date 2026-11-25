"""CUDA float32 sum of short columns in contiguous batched matrices."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import dispatch_context


def sum_columns_strict(x):
    rows, columns = map(int, x.shape[-2:])
    if (jt.flags.cuda_kernel_math != "strict" or dtype_name(x.dtype) != "float32"
            or not 2 <= rows <= 64 or columns < 2 or x.numel() // rows < 128
            or not x.is_contiguous() or dispatch_context(x).backend != "cuda"):
        return None
    return _SumColumns.apply(x)


class _SumColumns(jt.Function):
    def execute(self, x):
        self.input_shape = tuple(x.shape)
        rows, columns = self.input_shape[-2:]
        batches = x.numel() // (rows * columns)
        flat = x.reshape((batches, rows, columns))
        result = jt.code((batches, columns), x.dtype, [flat], cuda_src="""
            __global__ void sum_columns_cuda(@ARGS_DEF) {
                @PRECALC
                const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
                if (i >= int64_t(in0_shape0) * in0_shape2) return;
                const int column = i % in0_shape2;
                const int64_t batch = i / in0_shape2;
                const float* values = in0_p + batch * in0_shape1 * in0_shape2 + column;
                // Keep four independent accumulators before combining them
                // in order, as the vectorized PyTorch reduction does.
                float sums[4] = {0.f, 0.f, 0.f, 0.f};
                for (int row = 0; row < in0_shape1; row += 4) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j)
                        if (row + j < in0_shape1)
                            sums[j] += values[(row + j) * in0_shape2];
                }
                out0_p[i] = ((sums[0] + sums[1]) + sums[2]) + sums[3];
            }
            sum_columns_cuda<<<(int64_t(in0_shape0) * in0_shape2 + 127) / 128, 128>>>(@ARGS);
        """)
        return result.reshape(self.input_shape[:-2] + (1, columns))

    def grad(self, dout):
        return dout.broadcast(self.input_shape)
