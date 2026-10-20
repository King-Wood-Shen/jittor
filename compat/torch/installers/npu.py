"""A Torch NPU device facade over the native ACL runtime.

``torch.npu`` is one of the device namespaces the CUDA installer creates, and
until now every one of them got the same placeholders: ``is_available()``
answered False and ``device_count()`` answered 0 on a machine with eight
910Bs. A downstream library that asks before choosing a device -- which is
what Transformers, ms-swift and DeepSpeed all do -- was told there was no NPU
and went to the host.

What is published here is discovery, selection and synchronization, plus the
selected ``torch_npu`` APIs a Torch model reaches for directly. It is not
the torch_npu extension: streams, RNG state and the memory statistics have
their own contracts and are left refusing rather than answered with a zero.
"""
import contextlib
import types

import jittor as jt

from ..fidelity import Fidelity, register_api_bindings
from ..types import device as TorchDevice


def _acl_build():
    """Whether this Jittor was built against ACL, not merely able to load it."""
    return getattr(jt.compiler, "has_acl", 0) and getattr(
        getattr(jt.compiler, "build_config", None), "backend", None) == "acl"


def device_count():
    if not _acl_build():
        return 0
    # Ask the runtime, not the environment: a CANN or driver failure has to
    # surface, never be rounded down to "no devices".
    return int(jt.core.backend_device_count("acl"))


def is_available():
    return device_count() > 0


def _require_acl():
    if not is_available():
        raise RuntimeError("The Jittor ACL/NPU backend has no available devices")


def current_device():
    _require_acl()
    return int(jt.core.current_device())


def _index(value=None):
    _require_acl()
    if value is None:
        return current_device()
    if isinstance(value, bool):
        raise TypeError("NPU device index must not be bool")
    if isinstance(value, int):
        index = value
    else:
        parsed = TorchDevice(value)
        if parsed.type != "npu":
            raise ValueError("Expected an npu device, got {}".format(value))
        index = current_device() if parsed.index is None else parsed.index
    if index < 0 or index >= device_count():
        raise ValueError("NPU device index out of range: {}".format(index))
    return int(index)


def set_device(value):
    # The native setter owns the ACL context and the device-switch hooks.
    jt.core.set_device(_index(value))


@contextlib.contextmanager
def device(value=None):
    previous = current_device()
    selected = _index(value)
    if selected != previous:
        set_device(selected)
    try:
        yield
    finally:
        if selected != previous:
            set_device(previous)


def synchronize(value=None):
    _index(value)
    # Jittor's work is lazy, so a device wait alone would wait for nothing.
    # This submits what is pending and waits for every device it touched,
    # which is stronger than the per-device wait torch documents.
    jt.sync_all(True)


def empty_cache():
    _require_acl()
    jt.sync_all(True)
    jt.gc()


def is_bf16_supported():
    _require_acl()
    return True


def manual_seed(seed):
    _require_acl()
    from .core import _normalize_seed
    jt.core.rng_manual_seed("acl", _normalize_seed(seed), current_device())


def manual_seed_all(seed):
    _require_acl()
    from .core import _normalize_seed
    jt.core.rng_manual_seed_all("acl", _normalize_seed(seed))


def npu_rms_norm(input, weight, epsilon=1e-6):
    """``torch_npu.npu_rms_norm``, as an ACL tensor graph.

    Returns the pair torch_npu returns, with None for the reciprocal-norm
    output the fused kernel also produces and callers rarely read.
    """
    _require_acl()
    mean_square = jt.mean(input * input, dim=-1, keepdims=True)
    return input * jt.rsqrt(mean_square + epsilon) * weight, None


def npu_rotary_mul(input, cos, sin):
    _require_acl()
    half = input.shape[-1] // 2
    rotated = jt.concat((-input[..., half:], input[..., :half]), dim=-1)
    return input * cos + rotated * sin


def npu_swiglu(input, dim=-1):
    _require_acl()
    gate, value = jt.chunk(input, 2, dim=dim)
    return (gate * jt.sigmoid(gate)) * value


def npu_fusion_attention(
    query, key, value, head_num, input_layout, pse=None, padding_mask=None,
    atten_mask=None, scale=1.0, keep_prob=1.0,
    pre_tockens=2147483647, next_tokens=2147483647, inner_precise=0,
    prefix=None, actual_seq_qlen=None, actual_seq_kvlen=None, sparse_mode=0,
    gen_mask_parallel=True, sync=False, softmax_layout="",
):
    """Run the BSND and TND layouts used by Transformers' NPU FA wrapper.

    Jittor evaluates the attention graph on ACL; this does not claim to call
    CANN's fused operator. Unsupported fused-attention features fail loudly.
    """
    _require_acl()
    if input_layout not in ("BSND", "TND"):
        raise NotImplementedError(
            "Jittor ACL npu_fusion_attention currently supports BSND and TND")
    if any(value is not None for value in (pse, padding_mask, prefix)):
        raise NotImplementedError(
            "Jittor ACL npu_fusion_attention does not support pse, padding masks, or prefix")
    if (int(pre_tockens) != 2147483647
            or int(next_tokens) != 2147483647 or int(inner_precise) != 0
            or not gen_mask_parallel or sync or softmax_layout):
        raise NotImplementedError(
            "Jittor ACL npu_fusion_attention supports only full-window default settings")
    keep_prob = float(keep_prob)
    if not 0.0 < keep_prob <= 1.0:
        raise ValueError("keep_prob must be in (0, 1]")
    sparse_mode = int(sparse_mode)
    if sparse_mode not in (0, 2, 3):
        raise NotImplementedError("unsupported npu_fusion_attention sparse_mode")

    rank = 4 if input_layout == "BSND" else 3
    if any(len(tensor.shape) != rank for tensor in (query, key, value)):
        raise ValueError("query, key and value must match the selected attention layout")
    query_heads, head_dim = int(query.shape[-2]), int(query.shape[-1])
    key_heads, key_dim = int(key.shape[-2]), int(key.shape[-1])
    if (tuple(key.shape) != tuple(value.shape) or head_dim != key_dim
            or int(value.shape[-1]) != head_dim):
        raise ValueError("query, key and value have incompatible attention shapes")
    if input_layout == "BSND" and int(query.shape[0]) != int(key.shape[0]):
        raise ValueError("query, key and value batch dimensions must match")
    if int(head_num) != query_heads:
        raise ValueError("head_num must equal the query head count")
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError("key/value heads must divide the query head count")
    causal = sparse_mode in (2, 3)

    def _mask_for(length_q, length_k):
        if atten_mask is None:
            return None
        if str(atten_mask.dtype).lower().rsplit(".", 1)[-1] not in ("bool", "bool_"):
            raise NotImplementedError(
                "Jittor ACL npu_fusion_attention currently requires a boolean mask")
        # torch_npu masks mark blocked positions; SDPA masks mark positions
        # which participate. Each TND segment starts at local position zero.
        return ~atten_mask[..., :length_q, :length_k]

    from .nn.attention import scaled_dot_product_attention

    def _run_one(q, k, v, mask):
        if input_layout == "BSND":
            q_sdpa, k_sdpa, v_sdpa = (
                tensor.transpose(1, 2) for tensor in (q, k, v))
        else:
            q_sdpa, k_sdpa, v_sdpa = (
                tensor.unsqueeze(0).transpose(1, 2) for tensor in (q, k, v))
        output = scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa, attn_mask=mask,
            dropout_p=1.0 - keep_prob, is_causal=causal and mask is None,
            scale=float(scale), enable_gqa=key_heads != query_heads)
        if input_layout == "BSND":
            return output.transpose(1, 2)
        return output.transpose(1, 2).squeeze(0)

    if input_layout == "BSND":
        length_q, length_k = int(query.shape[1]), int(key.shape[1])
        if causal and length_q != length_k:
            raise NotImplementedError(
                "causal attention requires equal query and key lengths")
        output = _run_one(query, key, value, _mask_for(length_q, length_k))
    else:
        def _ends(values, total, name):
            if values is None:
                raise ValueError("TND attention requires %s cumulative lengths" % name)
            ends = tuple(int(end) for end in values)
            previous = 0
            for end in ends:
                if end <= previous or end > total:
                    raise ValueError("%s must be strictly increasing cumulative lengths" % name)
                previous = end
            if not ends or ends[-1] != total:
                raise ValueError("%s must end at the corresponding token count" % name)
            return ends

        q_ends = _ends(actual_seq_qlen, int(query.shape[0]), "actual_seq_qlen")
        k_ends = _ends(actual_seq_kvlen, int(key.shape[0]), "actual_seq_kvlen")
        if len(q_ends) != len(k_ends):
            raise ValueError("query and key cumulative lengths must have equal size")
        outputs = []
        q_start = k_start = 0
        for q_end, k_end in zip(q_ends, k_ends):
            length_q, length_k = q_end - q_start, k_end - k_start
            if causal and length_q != length_k:
                raise NotImplementedError(
                    "causal TND attention requires equal query and key lengths")
            outputs.append(_run_one(
                query[q_start:q_end], key[k_start:k_end], value[k_start:k_end],
                _mask_for(length_q, length_k)))
            q_start, k_start = q_end, k_end
        output = jt.concat(outputs, dim=0)

    # Transformers consumes attention_out ([0]); the auxiliary fused-kernel
    # workspaces are not materialized by this graph implementation.
    return output, None, None, None, None, None, None


def _unsupported(*args, **kw):
    raise NotImplementedError(
        "This NPU API is outside the Jittor ACL device facade")


#: Published as refusals rather than left to the shared placeholders, which
#: answer a question they cannot answer: a zero peak or an absent stream reads
#: as a measurement, and a caller cannot tell it from one.
_REFUSED = (
    "is_initialized", "get_device_name", "get_device_properties",
    "reset_max_memory_allocated", "reset_peak_memory_stats", "memory_stats",
    "memory_allocated", "max_memory_allocated", "memory_reserved",
    "max_memory_reserved", "mem_get_info",
    "current_stream", "default_stream", "set_stream", "Stream", "Event",
    "set_compile_mode", "ipc_collect",
)

_SUPPORTED = ("is_available", "device_count", "current_device", "set_device",
              "device", "synchronize", "empty_cache", "is_bf16_supported",
              "manual_seed", "manual_seed_all", "npu_rms_norm",
              "npu_rotary_mul", "npu_swiglu")


def install(ctx):
    target, registry = ctx.target_namespace, ctx.registry
    npu = types.ModuleType("torch.npu")
    npu.__path__ = []
    for name in _SUPPORTED:
        setattr(npu, name, globals()[name])
    for name in _REFUSED:
        setattr(npu, name, _unsupported)
    registry.publish("torch.npu", npu, replace=True)
    target.npu = npu

    from .cuda.api import (
        _api_cuda_get_rng_state, _api_cuda_get_rng_state_all,
        _api_cuda_set_rng_state, _api_cuda_set_rng_state_all,
    )
    random = types.ModuleType("torch.npu.random")
    random.get_rng_state = _api_cuda_get_rng_state
    random.get_rng_state_all = _api_cuda_get_rng_state_all
    random.set_rng_state = _api_cuda_set_rng_state
    random.set_rng_state_all = _api_cuda_set_rng_state_all
    npu.random = random
    registry.publish("torch.npu.random", random, bind_parent=False)

    register_api_bindings(
        npu, "torch.npu",
        ("is_available", "device_count", "current_device", "set_device",
         "device", "is_bf16_supported", "manual_seed", "manual_seed_all"),
        Fidelity.EXACT,
        "Native ACL discovery and selection; the index accepts an int, a "
        "string, a torch.device or None, as torch's does. Seed APIs reset "
        "the native ACL stream generators.")
    register_api_bindings(
        random, "torch.npu.random",
        ("get_rng_state", "get_rng_state_all", "set_rng_state", "set_rng_state_all"),
        Fidelity.EXACT,
        "Serializes and restores native ACL stream counter states as CPU byte tensors.")
    register_api_bindings(
        npu, "torch.npu", ("synchronize", "empty_cache"), Fidelity.APPROXIMATE,
        "Submits Jittor's pending work and waits for every device it "
        "touched, which is stronger than torch's per-device wait.")
    register_api_bindings(
        npu, "torch.npu", ("npu_rms_norm", "npu_rotary_mul", "npu_swiglu"),
        Fidelity.APPROXIMATE,
        "Executed as ACL tensor graphs rather than the fused CANN kernels; "
        "npu_rms_norm does not produce the auxiliary reciprocal-norm output.")
    register_api_bindings(
        npu, "torch.npu", _REFUSED, Fidelity.UNIMPLEMENTED,
        "Refused rather than answered: no synthetic device metadata, memory "
        "peaks or streams.")

    if not _acl_build():
        return
    # `import torch_npu` is what a Torch model does before calling any of the
    # three primitives above. Publishing a facade is only safe where no real
    # one is loaded -- otherwise the native extension and this would both
    # claim the name and the model would get whichever imported last.
    if registry.get("torch_npu") is not None:
        return
    facade = types.ModuleType("torch_npu")
    facade.__file__ = __file__
    facade.__jittor_acl_facade__ = True
    facade.__jittor_provider__ = "jittor.backends.acl"
    facade.npu = npu
    facade.npu_rms_norm = npu_rms_norm
    facade.npu_rotary_mul = npu_rotary_mul
    facade.npu_swiglu = npu_swiglu
    facade.npu_fusion_attention = npu_fusion_attention
    register_api_bindings(
        facade, "torch_npu",
        ("npu_rms_norm", "npu_rotary_mul", "npu_swiglu",
         "npu_fusion_attention"), Fidelity.APPROXIMATE,
        "Runs supported BSND and TND attention as ACL SDPA graphs; this does "
        "not expose CANN fused-kernel workspaces or fused-kernel performance.")
    registry.publish("torch_npu", facade, bind_parent=False)
