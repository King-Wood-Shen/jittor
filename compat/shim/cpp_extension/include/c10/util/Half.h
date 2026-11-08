#pragma once
#include <torch/extension.h>
#include <cuda_fp16.h>

// Extensions opt into the CUDA SDK FP16 scalar through this header. The
// CUDA-free torch/extension.h must not include it: client TUs may define float3.
namespace c10 { using Half = ::__half; }
namespace at { using Half = c10::Half; }
namespace torch { using Half = c10::Half; }
