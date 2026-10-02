#pragma once
// ATen/cuda/CUDAContext.h — only ever included by .cu TUs (nvcc), so it MAY pull
// the CUDA stream shim (which includes <cuda_runtime.h>). The public
// torch/extension.h must stay CUDA-free, so the stream API lives here, not there.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace at { namespace cuda {
inline cudaDeviceProp* getCurrentDeviceProperties() {
    static thread_local cudaDeviceProp properties;
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    return &properties;
}
}} // namespace at::cuda
