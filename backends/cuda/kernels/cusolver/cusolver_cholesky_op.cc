// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "core/var.h"
#include "core/executor.h"
#include "cusolver_cholesky_op.h"
#include "cusolver_solve_op.h"
#include "cusolver_wrapper.h"
#include "cusolver_layout.h"
#include "runtime/device.h"
#include "runtime/graph_capture.h"
#include "stream_compat.h"

namespace jittor {

#ifndef JIT

CusolverCholeskyOp::CusolverCholeskyOp(Var* a, bool upper) : a(a), upper(upper) {
    set_flag(OpFlags::_cuda, 1);
    set_flag(OpFlags::_cpu, 0);
    cusolver_check_square(a, "cusolver_cholesky");
    factor = create_output(nullptr, a->dtype());
}

void CusolverCholeskyOp::infer_shape() {
    USER_CHECK(a->is_contiguous()) << "cusolver_cholesky needs a dense input";
    factor->set_shape(a->shape);
}

void CusolverCholeskyOp::jit_prepare(JK& jk) {
    const auto dtype = a->dtype();
    jk << "«LPREFIX:" << cusolver_letter(dtype);
    jk << "«Tc:" << cusolver_ctype(dtype);
    jk << "«TYPE:" << (dtype == ns_float32 ? "CUDA_R_32F" : dtype == ns_float64 ? "CUDA_R_64F"
        : dtype == ns_complex64 ? "CUDA_C_32F" : "CUDA_C_64F");
    jk << "«UPPER=" << JK::hex(upper);
}

#else
#ifdef JIT_cuda
void CusolverCholeskyOp::jit_run() {
    const int n = a->shape[0];
    if (!n) return;
    auto handle = cusolver_bind_stream();
    auto stream = (cudaStream_t)cuda_compute_stream(current_device());
    auto& executor = runtime_executor();
    auto buffer = executor.alloc_temp(sizeof(Tc) * (size_t)n * n);
    auto info = executor.alloc_temp(sizeof(int));
    auto* matrix = (Tc*)buffer.ptr;
    cusolver_copy2d(n, n, a->ptr<Tc>(), n, 1, matrix, 1, n, false, stream);
    constexpr auto fill = UPPER ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
#if CUDART_VERSION >= 11010
    if (graph_capture_recording) {
        graph_capture_saw_host_work = true;
        return;
    }
    struct Params {
        cusolverDnParams_t value;
        Params() { checkCudaErrors(cusolverDnCreateParams(&value)); }
        ~Params() { cusolverDnDestroyParams(value); }
    } params;
    size_t device_bytes = 0, host_bytes = 0;
    checkCudaErrors(cusolverDnXpotrf_bufferSize(handle, params.value, fill, n, TYPE,
        matrix, n, TYPE, &device_bytes, &host_bytes));
    auto device = executor.alloc_temp(std::max(device_bytes, size_t(1)));
    std::vector<unsigned char> host(host_bytes);
    checkCudaErrors(cusolverDnXpotrf(handle, params.value, fill, n, TYPE, matrix, n, TYPE,
        device.ptr, device_bytes, host.data(), host_bytes, (int*)info.ptr));
#else
    int count = 0;
    checkCudaErrors(cusolverDn@LPREFIX@@potrf_bufferSize(handle, fill, n, matrix, n, &count));
    auto work = executor.alloc_temp(sizeof(Tc) * (size_t)std::max(count, 1));
    checkCudaErrors(cusolverDn@LPREFIX@@potrf(handle, fill, n, matrix, n,
        (Tc*)work.ptr, count, (int*)info.ptr));
    if (graph_capture_recording) {
        graph_capture_saw_host_work = true;
        return;
    }
#endif
    int status = 0;
    checkCudaErrors(cudaMemcpyAsync(&status, info.ptr, sizeof(status), cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
    ASSERT(status >= 0) << "cuSOLVER Cholesky reported an invalid argument" << status;
    _TYPED_ERROR_IF(UserLinAlgError, status > 0)
        << "linalg.cholesky: input is not positive definite at leading minor" << status;
    // The lower case copies the upper triangle of the transposed view.
    cusolver_copy2d(n, n, matrix, UPPER ? 1 : n, UPPER ? n : 1,
        factor->ptr<Tc>(), UPPER ? n : 1, UPPER ? 1 : n, true, stream);
}
#endif
#endif

} // jittor
