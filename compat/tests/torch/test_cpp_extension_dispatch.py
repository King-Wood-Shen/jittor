"""Public ATen dispatch compiles real CUDA kernels, including FP16 storage."""
import functools

import numpy as np
import pytest
import torch
import jittor as jt

from jittor._runtime.fallback import forbid_backend_fallbacks

_SOURCE = r'''
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

int dispatch_size(int dtype) {
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        static_cast<at::ScalarType>(dtype), "dispatch_size", [&] {
            return static_cast<int>(sizeof(scalar_t));
        });
}
int floating_size(int dtype) {
    return AT_DISPATCH_FLOATING_TYPES(
        static_cast<at::ScalarType>(dtype), "floating_size", [&] {
            return static_cast<int>(sizeof(scalar_t));
        });
}
int dispatch_once() {
    int evaluations = 0;
    int width = AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        (++evaluations, at::kFloat), "dispatch_once", [&] {
            return static_cast<int>(sizeof(scalar_t));
        });
    return evaluations * 100 + width;
}
template <typename T> __global__ void copy_kernel(const T* input, T* output) {
    output[threadIdx.x] = input[threadIdx.x];
}
at::Tensor copy_tensor(at::Tensor input) {
    auto output = at::empty_like(input);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "copy_tensor", [&] {
        copy_kernel<scalar_t><<<1, input.numel(), 0, at::cuda::getCurrentCUDAStream()>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>());
    });
    return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch_size", &dispatch_size);
    m.def("floating_size", &floating_size);
    m.def("dispatch_once", &dispatch_once);
    m.def("copy_tensor", &copy_tensor);
}
'''


@functools.lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load_inline
    return load_inline('jittor_aten_dispatch_probe', cuda_sources=_SOURCE,
                       extra_cuda_cflags=['-std=c++17'], verbose=False)


def test_dispatch_scalar_types_return_values_and_single_evaluation():
    mod = extension()
    assert [mod.dispatch_size(n) for n in (5, 6, 7)] == [2, 4, 8]
    assert [mod.floating_size(n) for n in (6, 7)] == [4, 8]
    assert mod.dispatch_once() == 104
    for dtype in (3, 11, 15):
        with pytest.raises(RuntimeError, match='dispatch_size.*not implemented'):
            mod.dispatch_size(dtype)
    with pytest.raises(RuntimeError, match='floating_size.*not implemented'):
        mod.floating_size(5)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
def test_dispatch_cuda_pointer_types_preserve_values(dtype):
    with forbid_backend_fallbacks():
        x = torch.tensor([0.0, 1.5, -2.0, 65504.0], dtype=dtype, device='cuda')
        result = extension().copy_tensor(x)
        torch.cuda.synchronize()
        assert result.device.type == 'cuda' and result.dtype == dtype
        np.testing.assert_array_equal(result.cpu().numpy(), x.cpu().numpy())


def test_extension_header_remains_cuda_free_for_client_vector_types():
    from torch.utils.cpp_extension import load_inline
    source = r'''
#include <torch/extension.h>
struct float3 { float x, y, z; };
int client_vector_size() { return sizeof(float3); }
'''
    mod = load_inline('jittor_cuda_free_extension_header_probe', cpp_sources=source,
                      functions=['client_vector_size'], verbose=False)
    assert mod.client_vector_size() == 12


def test_host_dispatch_half_aliases_and_sdk_conversion():
    from torch.utils.cpp_extension import load_inline
    source = r'''
#include <ATen/ATen.h>
static_assert(std::is_same<c10::Half, at::Half>::value, "Half alias");
static_assert(std::is_same<c10::Half, torch::Half>::value, "Half alias");
static_assert(sizeof(at::Half) == 2 && alignof(at::Half) == 2, "Half ABI");
float host_half_conversion() {
    return AT_DISPATCH_FLOATING_TYPES_AND_HALF(at::kHalf, "host_half_conversion", [&] {
        scalar_t value = static_cast<scalar_t>(1.5f);
        return static_cast<float>(value);
    });
}
'''
    mod = load_inline('jittor_host_dispatch_half_probe', cpp_sources=source,
                      functions=['host_half_conversion'], verbose=False)
    assert mod.host_half_conversion() == 1.5


def test_cpp_extension_preserves_parameter_result_type_and_restores_scope():
    with forbid_backend_fallbacks():
        parameter = torch.nn.Parameter(torch.ones(4, device='cuda'))
        result = extension().copy_tensor(parameter)
        assert type(result) is torch.Tensor
        assert not isinstance(result, torch.nn.Parameter)
        native = jt.ones(4)
        assert type(native) is jt.Var
        native_result = extension().copy_tensor(native)
        assert type(native_result) is jt.Var
        np.testing.assert_array_equal(result.cpu().numpy(), np.ones(4))
        np.testing.assert_array_equal(native_result.numpy(), np.ones(4))
