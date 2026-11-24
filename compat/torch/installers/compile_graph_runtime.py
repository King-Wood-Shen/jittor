"""Bindings for the observed vLLM decoder regions."""
import ast
import operator

from .compile_graph_partitions import bind_decoder_graph, partition_decoder_graph


def _source_path(node):
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        key = node.slice.value
        if isinstance(node.value, ast.Name) and node.value.id == "L":
            if key in ("self", "inputs_embeds", "input_ids", "positions"):
                return (("input", key),)
        elif isinstance(node.value, ast.Attribute) and type(key) is str:
            if node.value.attr in ("_modules", "_buffers", "_parameters"):
                return _source_path(node.value.value) + (("attribute", key),)
        elif isinstance(node.value, ast.Call) and type(key) is int:
            call = node.value
            if (not call.args and not call.keywords
                    and isinstance(call.func, ast.Attribute) and call.func.attr in ("size", "stride")):
                return _source_path(call.func.value) + ((call.func.attr, key),)
    raise ValueError("Unsupported FX placeholder provenance")


def _read_source(path, inputs):
    _, key = path[0]
    value = inputs[key]
    for kind, component in path[1:]:
        if kind == "attribute":
            value = getattr(value, component)
        elif kind == "size":
            value = value.shape[component]
        else:
            value = value.stride()[component]
    return value


def _literal(torch, kind, value):
    if kind == "ellipsis" and value == "Ellipsis":
        return Ellipsis
    if kind == "dtype" and value in ("torch.bfloat16", "torch.float32", "torch.int64"):
        return getattr(torch, value.split(".")[1])
    if kind == "device":
        return torch.device(value)
    raise ValueError(f"Unsupported FX literal {kind} {value}")


def bind_vllm_decoder(program, torch, device_effect, triton_effect, execute_pure):
    """Bind real operators and explicit device and kernel callbacks."""
    functional = torch.nn.functional
    functions = {
        "_operator.add": operator.add,
        "_operator.floordiv": operator.floordiv,
        "_operator.getitem": operator.getitem,
        "_operator.mul": operator.mul,
        "_operator.sub": operator.sub,
        "torch._C._nn.linear": functional.linear,
        "torch.cat": torch.cat,
        "torch.chunk": torch.chunk,
        "torch.empty_like": torch.empty_like,
        "torch.nn.functional.embedding": functional.embedding,
        "torch.nn.functional.silu": functional.silu,
        "torch.rsqrt": torch.rsqrt,
        "torch.zeros": torch.zeros,
    }

    def lower(kind, node):
        target = node["target"]
        if kind == "attention":
            return getattr(torch.ops.vllm, target.removeprefix("torch._ops.vllm."))
        if kind == "device":
            return device_effect(target)
        if kind == "triton":
            return triton_effect(node)
        if node["op"] == "call_method":
            return _method(target)
        return functions[target]

    return bind_decoder_graph(
        program, lower, lambda kind, value: _literal(torch, kind, value), execute_pure,
    )


def _method(target):
    def run(receiver, *args, **kwargs):
        return getattr(receiver, target)(*args, **kwargs)
    return run


def bind_placeholder_sources(program):
    """Return a validated resolver for captured vLLM forward inputs."""
    names, _, _ = partition_decoder_graph(program)
    placeholders = {node["name"]: node for node in program["nodes"]
                    if node["op"] == "placeholder"}
    paths = tuple(_source_path(ast.parse(placeholders[name]["binding"], mode="eval").body)
                  for name in names)

    def resolve(inputs):
        return tuple(_read_source(path, inputs) for path in paths)

    return resolve


def bind_vllm_rotary_effects(program, torch):
    """Lower the observed non-interleaved rotary mutation on the caller's GPU."""
    kernel_name = "vllm.vllm_flash_attn.ops.triton.rotary.rotary_kernel"
    tables = program["triton_tables"]
    expected = {
        "CU_SEQLENS": None, "SEQLEN_OFFSETS": 0, "IS_SEQLEN_OFFSETS_TENSOR": False,
        "IS_VARLEN": False, "INTERLEAVED": False, "CONJUGATE": False,
    }

    def bind(node):
        entry = tables[node["name"]]
        constants = entry["constants"]
        if entry["kernel"] != kernel_name or any(constants[k] != v for k, v in expected.items()):
            raise ValueError("Unsupported vLLM rotary kernel variant")
        width = constants["rotary_dim"]
        if type(width) is not int or width <= 0 or width % 2:
            raise ValueError("Invalid rotary width")
        half = width // 2

        def run(*, kernel_idx, constant_args_idx, grid, tma_descriptor_metadata, kwargs):
            x, out, cos, sin = (kwargs[k] for k in ("X", "OUT", "COS", "SIN"))
            if tma_descriptor_metadata or x.ndim != 4 or x.shape[-1] != width:
                raise ValueError("Unsupported vLLM rotary layout")
            token_rows = x.shape[1]
            if (x.shape[0] != 1 or tuple(x.shape) != tuple(out.shape)
                    or tuple(cos.shape) != (token_rows, half)
                    or tuple(sin.shape) != (token_rows, half)
                    or (kwargs["seqlen"], kwargs["seqlen_ro"]) != (token_rows, token_rows)
                    or x.dtype != out.dtype or any(v.device != x.device for v in (out, cos, sin))
                    or x.device.type != "cuda"):
                raise ValueError("Rotary inputs do not match the captured full-width layout")
            # Match the native Triton loads and stores. In particular, reducing
            # the two products to BF16 before the add changes the Q/K values.
            left, right = torch.chunk(x.float(), 2, dim=-1)
            c, s = cos.float().unsqueeze(0).unsqueeze(-2), sin.float().unsqueeze(0).unsqueeze(-2)
            rotated = torch.cat((left * c - right * s, left * s + right * c), dim=-1)
            out.copy_(rotated.to(out.dtype))

        return run

    return bind


def bind_single_cuda_device_effect(torch, target):
    """Honor device scopes only when the observed and active device agree."""
    def run(device):
        current = torch.cuda.current_device()
        if torch.cuda.device_count() != 1 or device != current:
            raise ValueError(f"Unsupported device transition in {target}: {current} to {device}")
        return current

    return run
