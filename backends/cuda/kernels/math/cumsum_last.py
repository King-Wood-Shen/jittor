"""CUDA inclusive scan for contiguous float32 rows."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import dispatch_context


def cumsum_last_strict(x, dim):
    if (jt.flags.cuda_kernel_math != "strict" or not isinstance(x, jt.Var)
            or not x.ndim or dim not in (-1, x.ndim - 1)
            or dtype_name(x.dtype) != "float32" or not x.is_contiguous()
            or dispatch_context(x).backend != "cuda"):
        return None
    width = int(x.shape[-1])
    if not 0 < width <= 256 or x.numel() <= width:
        return None
    return _CumsumLast.apply(x)


class _CumsumLast(jt.Function):
    def execute(self, x):
        width = int(x.shape[-1])
        rows = x.numel() // width
        # The upstream scan combines pairs of power-of-two groups in each
        # tile. Match its tile width so partial sums round in the same order.
        log_lanes = min(9, max(4, (9 + (width - 1).bit_length() - (rows - 1).bit_length()) // 2))
        lanes = 1 << log_lanes
        flat = x.reshape((rows, width))
        result = jt.code(flat.shape, x.dtype, [flat], cuda_src=f"""
            __global__ void scan_last_cuda(@ARGS_DEF) {{
                @PRECALC
                constexpr int lanes = {lanes};
                const int thread = threadIdx.x;
                const int width = in0_shape1;
                const float* input = in0_p + int64_t(blockIdx.x) * width;
                float* output = out0_p + int64_t(blockIdx.x) * width;
                __shared__ float values[2 * lanes];
                float carry = 0.f;
                for (int tile = 0; tile < width; tile += 2 * lanes) {{
                    const int left = tile + thread, right = left + lanes;
                    values[thread] = left < width ? input[left] : 0.f;
                    values[thread + lanes] = right < width ? input[right] : 0.f;
                    if (!thread) values[0] += carry;
                    __syncthreads();
                    for (int group = 1; group <= lanes; group *= 2) {{
                        const int boundary = (thread / group) * 2 * group + group;
                        values[boundary + thread % group] += values[boundary - 1];
                        __syncthreads();
                    }}
                    if (left < width) output[left] = values[thread];
                    if (right < width) output[right] = values[thread + lanes];
                    carry = values[2 * lanes - 1];
                    __syncthreads();
                }}
            }}
            scan_last_cuda<<<in0_shape0, {lanes}>>>(@ARGS);
        """)
        return result.reshape(x.shape)

    def grad(self, dout):
        return jt.flip(_CumsumLast.apply(jt.flip(dout, (-1,))), (-1,))
