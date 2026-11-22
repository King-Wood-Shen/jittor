"""Select real vLLM kernels supported by the Jittor Torch frontend.

This changes vLLM's implicit attention choice for FP32 V1 and its implicit
compilation choice for FP32 classification.  An explicit attention backend or
compilation level remains the caller's choice.
"""

from functools import wraps

from jittor.compat.transaction import set_attr


def patch_cuda_platform(module):
    platform = getattr(module, "CudaPlatformBase", None)
    if platform is None or getattr(platform, "_jittor_fp32_pool_backend", False):
        return False
    original = platform.get_attn_backend_cls

    @wraps(original)
    def choose_backend(cls, selected_backend, head_size, dtype,
                       kv_cache_dtype, block_size, use_v1, use_mla,
                       has_sink, use_sparse):
        if selected_backend is None and use_v1 and not use_mla:
            import torch
            if dtype == torch.float32:
                selected_backend = module._Backend.TRITON_ATTN
        return original(selected_backend, head_size, dtype, kv_cache_dtype,
                        block_size, use_v1, use_mla, has_sink, use_sparse)

    set_attr(platform, "get_attn_backend_cls", classmethod(choose_backend))
    set_attr(platform, "_jittor_fp32_pool_backend", True)
    return True


def patch_vllm_config(module):
    config = getattr(module, "VllmConfig", None)
    if config is None or getattr(config, "_jittor_pool_no_torch_fx", False):
        return False
    original = config.__post_init__

    @wraps(original)
    def initialise(self):
        model = self.model_config
        compilation = self.compilation_config
        if (model is not None and getattr(model, "pooler_config", None) is not None
                and model.dtype == module.torch.float32
                and compilation.level is None):
            # vLLM's implicit PIECEWISE choice needs torch.fx and a custom
            # torch.compile backend.  Jittor executes real kernels eagerly;
            # it does not claim either PyTorch graph API.  Keep explicit
            # compilation requests untouched so unsupported ones fail loudly.
            compilation.level = module.CompilationLevel.NO_COMPILATION
            if compilation.cudagraph_mode is None:
                compilation.cudagraph_mode = module.CUDAGraphMode.NONE
        return original(self)

    set_attr(config, "__post_init__", initialise)
    set_attr(config, "_jittor_pool_no_torch_fx", True)
    return True


PATCHES = {
    "vllm.platforms.cuda": patch_cuda_platform,
    "vllm.config": patch_vllm_config,
}
