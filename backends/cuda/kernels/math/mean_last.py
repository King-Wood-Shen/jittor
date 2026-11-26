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


def mean_vector_strict(x):
    """Mean of one aligned FP32 vector with the Torch CUDA reduction order."""
    width = x.numel()
    if (jt.flags.cuda_kernel_math != "strict" or dtype_name(x.dtype) != "float32"
            or not 128 <= width <= 131072 or width % 4
            or not x.is_contiguous() or x.storage_offset() % 4
            or dispatch_context(x).backend != "cuda"):
        return None
    return _MeanVector.apply(x)


class _MeanVector(jt.Function):
    def execute(self, x):
        width = x.numel()
        self.scale = 1.0 / width
        self.input_shape = tuple(x.shape)
        threads = min(512, 1 << ((width // 4).bit_length() - 1))
        values_per_thread = (width + threads - 1) // threads
        blocks = (values_per_thread + 15) // 16 if values_per_thread >= 256 else 1
        mean, _ = jt.code([(1,), (blocks,)], [x.dtype, x.dtype], [x], cuda_src="""
            __device__ float strict_mean_sum(float value, float* shared) {
                const int lane = threadIdx.x;
                shared[lane] = value;
                for (int distance = blockDim.x / 2; distance >= 32; distance /= 2) {
                    __syncthreads();
                    if (lane < distance) {
                        value = __fadd_rn(value, shared[lane + distance]);
                        shared[lane] = value;
                    }
                }
                if (lane < 32) {
                    for (int distance = 16; distance; distance /= 2) {
                        value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, distance));
                    }
                }
                return value;
            }

            __global__ void strict_mean_partials(@ARGS_DEF) {
                @PRECALC
                extern __shared__ float scratch[];
                const int first = blockIdx.x * blockDim.x + threadIdx.x;
                const int stride = gridDim.x * blockDim.x;
                const float4* values = reinterpret_cast<const float4*>(in0_p);
                float4 acc = {0.f, 0.f, 0.f, 0.f};
                for (int i = first; i < in0_shape0 / 4; i += stride) {
                    const float4 next = values[i];
                    acc.x = __fadd_rn(acc.x, next.x);
                    acc.y = __fadd_rn(acc.y, next.y);
                    acc.z = __fadd_rn(acc.z, next.z);
                    acc.w = __fadd_rn(acc.w, next.w);
                }
                float total = __fadd_rn(__fadd_rn(__fadd_rn(acc.x, acc.y), acc.z), acc.w);
                total = strict_mean_sum(total, scratch);
                if (threadIdx.x == 0) {
                    if (gridDim.x == 1)
                        out0_p[0] = __fmul_rn(total, __fdiv_rn(1.f, float(in0_shape0)));
                    else
                        out1_p[blockIdx.x] = total;
                }
            }

            __global__ void strict_mean_finalize(@ARGS_DEF) {
                @PRECALC
                extern __shared__ float scratch[];
                float value = 0.f;
                for (int i = threadIdx.x; i < out1_shape0; i += blockDim.x)
                    value = __fadd_rn(value, out1_p[i]);
                value = strict_mean_sum(value, scratch);
                if (threadIdx.x == 0)
                    out0_p[0] = __fmul_rn(value, __fdiv_rn(1.f, float(in0_shape0)));
            }

            const int threads = in0_shape0 >= 2048
                ? 512 : 1 << (31 - __builtin_clz(in0_shape0 / 4));
            strict_mean_partials<<<out1_shape0, threads, threads * sizeof(float)>>>(@ARGS);
            if (out1_shape0 > 1)
                strict_mean_finalize<<<1, threads, threads * sizeof(float)>>>(@ARGS);
        """)
        return mean.reshape(())

    def grad(self, dout):
        return dout.broadcast(self.input_shape) * self.scale
