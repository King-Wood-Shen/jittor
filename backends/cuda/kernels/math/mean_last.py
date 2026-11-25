"""Vectorized float32 CUDA mean for contiguous rows."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import dispatch_context


def mean_last_strict(x):
    width = int(x.shape[-1])
    if (jt.flags.cuda_kernel_math != "strict" or dtype_name(x.dtype) != "float32"
            or not 128 <= width <= 8160 or width % 4
            or x.numel() // width < 16 or not x.is_contiguous()
            or x.storage_offset() % 4
            or dispatch_context(x).backend != "cuda"):
        return None
    return _MeanLast.apply(x)


class _MeanLast(jt.Function):
    def execute(self, x):
        self.input_shape = tuple(x.shape)
        width = self.input_shape[-1]
        self.scale = 1.0 / width
        rows = x.numel() // width
        flat = x.reshape((rows, width))
        means = jt.code((rows,), x.dtype, [flat], cuda_src="""
            __global__ void mean_last_cuda(@ARGS_DEF) {
                @PRECALC
                const int row = blockIdx.x * blockDim.y + threadIdx.y;
                if (row >= in0_shape0) return;
                const float* values = in0_p + int64_t(row) * in0_shape1;
                float sums[4] = {0.f, 0.f, 0.f, 0.f};
                for (int i = threadIdx.x * 4; i < in0_shape1; i += 128) {
                    #pragma unroll
                    for (int j = 0; j < 4; j++) sums[j] += values[i + j];
                }
                float total = ((sums[0] + sums[1]) + sums[2]) + sums[3];
                for (int offset = 16; offset > 0; offset >>= 1)
                    total += __shfl_down_sync(0xffffffff, total, offset);
                if (!threadIdx.x) out0_p[row] = total * (1.f / float(in0_shape1));
            }
            mean_last_cuda<<<(in0_shape0 + 3) / 4, dim3(32, 4)>>>(@ARGS);
        """)
        return means.reshape(self.input_shape[:-1] + (1,))

    def grad(self, dout):
        return dout.broadcast(self.input_shape) * self.scale
