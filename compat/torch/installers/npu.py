"""A Torch NPU device facade over the native ACL runtime.

``torch.npu`` is one of the device namespaces the CUDA installer creates, and
until now every one of them got the same placeholders: ``is_available()``
answered False and ``device_count()`` answered 0 on a machine with eight
910Bs. A downstream library that asks before choosing a device -- which is
what Transformers, ms-swift and DeepSpeed all do -- was told there was no NPU
and went to the host.

What is published here is discovery, selection, synchronization, exact ACL RNG
state, plus the three ``torch_npu`` primitives a Torch model reaches for
directly. It is not the torch_npu extension: streams and memory statistics
have their own contracts and are left refusing rather than answered with a
zero.
"""
import contextlib
import types

import numpy as np
import jittor as jt

from ..context import get_install_context
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
    """Seed the current native ACL RNG stream without changing CPU state."""
    _require_acl()
    from .core import _normalize_seed
    value = _normalize_seed(seed, integer=True)
    jt.core.rng_manual_seed("acl", value, current_device())


def manual_seed_all(seed):
    """Seed every visible native ACL RNG stream."""
    _require_acl()
    from .core import _normalize_seed
    value = _normalize_seed(seed, integer=True)
    jt.core.rng_manual_seed_all("acl", value)


_RNG_STATE_MAGIC = b"JTACL_RNG_1"


def _pack_rng_state(device):
    payload = jt.core.rng_state("acl", int(device)).encode("ascii")
    owner = get_install_context(jt).target_namespace
    data = _RNG_STATE_MAGIC + payload
    return owner.tensor(np.frombuffer(data, dtype=np.uint8).copy(),
                        dtype=owner.uint8, device="cpu")


def _unpack_rng_state(state):
    from jittor._core.dtypes import dtype_name
    if not isinstance(state, jt.Var) or dtype_name(state.dtype) != "uint8":
        raise TypeError("NPU RNG state must be a CPU ByteTensor")
    if len(state.shape) != 1 or state.device.type != "cpu":
        raise TypeError("NPU RNG state must be a one-dimensional CPU ByteTensor")
    raw = state.numpy().tobytes()
    if not raw.startswith(_RNG_STATE_MAGIC):
        raise ValueError("not a Jittor ACL RNG state")
    try:
        return raw[len(_RNG_STATE_MAGIC):].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("invalid Jittor ACL RNG state encoding") from exc


def _random_get_rng_state(device=None):
    index = _index(device)
    jt.sync_all(True)
    return _pack_rng_state(index)


def _random_get_rng_state_all():
    _require_acl()
    jt.sync_all(True)
    return [_pack_rng_state(index) for index in range(device_count())]


def _random_set_rng_state(state, device=None):
    payload = _unpack_rng_state(state)
    index = _index(device)
    jt.sync_all(True)
    jt.core.set_rng_state("acl", payload, index)


def _random_set_rng_state_all(states):
    payloads = [_unpack_rng_state(state) for state in states]
    if len(payloads) != device_count():
        raise ValueError("NPU RNG state count must match visible devices")
    jt.sync_all(True)
    jt.core.set_rng_states("acl", payloads)


def _random_initial_seed():
    return int(jt.core.rng_initial_seed("acl", current_device()))


def _random_manual_seed(seed):
    return manual_seed(seed)


def _random_manual_seed_all(seed):
    return manual_seed_all(seed)


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


def npu_fusion_attention(query, key, value, head_num, input_layout,
                         pse=None, atten_mask=None, scale=1.0,
                         keep_prob=1.0, pre_tockens=2147483647,
                         next_tockens=2147483647, inner_precise=0,
                         seed=0, offset=0, *, actual_seq_qlen=None,
                         actual_seq_kvlen=None, sparse_mode=0, **kwargs):
    """Torch NPU attention semantics using JTorch SDPA tensor operations.

    This compatibility implementation supports the layouts used by
    Transformers' Ascend FlashAttention adapter. It preserves causal and
    variable-length boundaries, but does not use a fused CANN kernel.
    """
    _require_acl()
    import torch
    import torch.nn.functional as F

    if kwargs:
        raise TypeError("unsupported npu_fusion_attention options: {}".format(
            ", ".join(sorted(kwargs))))
    if pse is not None:
        raise NotImplementedError("npu_fusion_attention pse is not supported")
    if atten_mask is not None and int(sparse_mode) not in (2, 3):
        raise NotImplementedError(
            "npu_fusion_attention only supports the NPU causal mask modes")
    if not 0.0 <= float(keep_prob) <= 1.0:
        raise ValueError("keep_prob must be between zero and one")
    if input_layout not in ("BSND", "TND"):
        raise NotImplementedError("npu_fusion_attention layout {} is not supported".format(
            input_layout))
    if int(sparse_mode) not in (0, 2, 3):
        raise NotImplementedError("npu_fusion_attention sparse_mode {} is not supported".format(
            sparse_mode))

    def attend(q, k, v):
        if q.shape[1] != int(head_num) or k.shape[1] != int(head_num) \
                or v.shape[1] != int(head_num):
            raise ValueError("head_num does not match the attention tensor layout")
        q_len, kv_len = int(q.shape[0]), int(k.shape[0])
        qh = q.permute(1, 0, 2).unsqueeze(0)
        kh = k.permute(1, 0, 2).unsqueeze(0)
        vh = v.permute(1, 0, 2).unsqueeze(0)
        causal = int(sparse_mode) in (2, 3)
        options = {"dropout_p": 1.0 - float(keep_prob),
                   "is_causal": causal and q_len == kv_len,
                   "scale": float(scale)}
        if causal and q_len != kv_len:
            q_pos = torch.arange(q_len, device=q.device) + (kv_len - q_len)
            k_pos = torch.arange(kv_len, device=q.device)
            options["attn_mask"] = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
            options["is_causal"] = False
        out = F.scaled_dot_product_attention(qh, kh, vh, **options)
        return out.squeeze(0).permute(1, 0, 2)

    if input_layout == "BSND":
        if len(query.shape) != 4 or len(key.shape) != 4 or len(value.shape) != 4:
            raise ValueError("BSND attention expects rank-four tensors")
        outputs = [attend(query[i], key[i], value[i])
                   for i in range(int(query.shape[0]))]
        output = torch.stack(outputs, dim=0)
        return output, None, None, None

    if len(query.shape) != 3 or len(key.shape) != 3 or len(value.shape) != 3:
        raise ValueError("TND attention expects rank-three tensors")
    if actual_seq_qlen is None or actual_seq_kvlen is None:
        raise ValueError("TND attention requires actual_seq_qlen and actual_seq_kvlen")
    q_ends = [int(x) for x in actual_seq_qlen]
    kv_ends = [int(x) for x in actual_seq_kvlen]
    if len(q_ends) != len(kv_ends) or not q_ends or q_ends[-1] != query.shape[0] \
            or kv_ends[-1] != key.shape[0] or value.shape[0] != key.shape[0] \
            or any(a <= b for a, b in zip(q_ends, [0] + q_ends[:-1])) \
            or any(a <= b for a, b in zip(kv_ends, [0] + kv_ends[:-1])):
        raise ValueError("TND sequence boundaries must be increasing cumulative ends")
    outputs = []
    q_start = kv_start = 0
    for q_end, kv_end in zip(q_ends, kv_ends):
        outputs.append(attend(query[q_start:q_end], key[kv_start:kv_end],
                              value[kv_start:kv_end]))
        q_start, kv_start = q_end, kv_end
    output = torch.cat(outputs, dim=0)
    return output, None, None, None


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
              "manual_seed", "manual_seed_all",
              "npu_rms_norm", "npu_rotary_mul", "npu_swiglu",
              "npu_fusion_attention")


def install(ctx):
    target, registry = ctx.target_namespace, ctx.registry
    npu = types.ModuleType("torch.npu")
    npu.__path__ = []
    for name in _SUPPORTED:
        setattr(npu, name, globals()[name])
    for name in _REFUSED:
        setattr(npu, name, _unsupported)
    random_api = types.ModuleType("torch.npu.random")
    random_api.get_rng_state = _random_get_rng_state
    random_api.get_rng_state_all = _random_get_rng_state_all
    random_api.set_rng_state = _random_set_rng_state
    random_api.set_rng_state_all = _random_set_rng_state_all
    random_api.initial_seed = _random_initial_seed
    random_api.manual_seed = _random_manual_seed
    random_api.manual_seed_all = _random_manual_seed_all
    npu.random = random_api
    registry.publish("torch.npu.random", random_api, bind_parent=False, replace=True)
    registry.publish("torch.npu", npu, replace=True)
    target.npu = npu

    register_api_bindings(
        npu, "torch.npu",
        ("is_available", "device_count", "current_device", "set_device",
         "device", "is_bf16_supported"),
        Fidelity.EXACT,
        "Native ACL discovery and selection; the index accepts an int, a "
        "string, a torch.device or None, as torch's does.")
    register_api_bindings(
        npu, "torch.npu", ("manual_seed", "manual_seed_all"), Fidelity.EXACT,
        "Seeds the native ACL random streams, current device or all visible "
        "devices respectively.")
    register_api_bindings(
        random_api, "torch.npu.random",
        ("get_rng_state", "get_rng_state_all", "set_rng_state",
         "set_rng_state_all", "initial_seed", "manual_seed",
         "manual_seed_all"),
        Fidelity.EXACT,
        "Versioned byte snapshots preserve and restore native ACL seed and "
        "counter state for the selected device or all visible devices.")
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
        npu, "torch.npu", ("npu_fusion_attention",), Fidelity.APPROXIMATE,
        "BSND and TND attention execute through JTorch SDPA with causal and "
        "sequence-boundary semantics; no fused CANN kernel or fused-memory "
        "benefit is provided.")
    register_api_bindings(
        npu, "torch.npu", _REFUSED, Fidelity.UNIMPLEMENTED,
        "Refused rather than answered: no synthetic device metadata, memory "
        "peaks or streams.")

    if not _acl_build():
        return
    # `import torch_npu` is what a Torch model does before calling primitives
    # above. Publishing a facade is only safe where no real
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
        ("npu_rms_norm", "npu_rotary_mul", "npu_swiglu", "npu_fusion_attention"), Fidelity.APPROXIMATE,
        "The same ACL tensor graphs torch.npu publishes.")
    registry.publish("torch_npu", facade, bind_parent=False)
