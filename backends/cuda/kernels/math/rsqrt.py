"""CUDA float32 reciprocal square root."""

import jittor as jt


class Rsqrt(jt.Function):
    def execute(self, x):
        flat = x.reshape((-1,))
        result = jt.code(flat.shape, flat.dtype, [flat], cuda_src="""
            __global__ void rsqrt_cuda(@ARGS_DEF) {
                @PRECALC
                const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
                if (i < in0_shape0) out0_p[i] = rsqrtf(in0_p[i]);
            }
            if (in0_shape0) rsqrt_cuda<<<(in0_shape0 + 255) / 256, 256>>>(@ARGS);
        """)
        self.result = result.reshape(x.shape)
        return self.result

    def grad(self, dout):
        y = self.result
        return -0.5 * dout * y * y * y
