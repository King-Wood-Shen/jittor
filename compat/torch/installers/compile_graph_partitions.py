"""Preserve FX order and reference dependencies across decoder effects."""
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby


@dataclass(frozen=True)
class Partition:
    kind: str
    nodes: tuple
    inputs: tuple
    outputs: tuple


def _references(value):
    if isinstance(value, Mapping):
        if value.keys() == {"ref"}:
            yield value["ref"]
        else:
            for item in value.values():
                yield from _references(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _references(item)


_PURE_FUNCTIONS = frozenset({
    "_operator.add", "_operator.floordiv", "_operator.getitem", "_operator.mul",
    "_operator.sub", "torch._C._nn.linear", "torch.cat", "torch.empty_like",
    "torch.chunk", "torch.nn.functional.embedding", "torch.nn.functional.silu",
    "torch.rsqrt", "torch.zeros",
})
_PURE_METHODS = frozenset({
    "chunk", "contiguous", "flatten", "index_select", "long", "mean",
    "pow", "reshape", "size", "split",
    "squeeze", "to", "unsqueeze", "view",
})
_DEVICE_EFFECTS = frozenset({
    "torch._C._cuda_exchangeDevice", "torch._C._cuda_maybeExchangeDevice",
})


def _kind(node, attention_targets):
    target = node["target"]
    if node["op"] == "call_function":
        if target in attention_targets:
            return "attention"
        if target == "torch.ops.higher_order.triton_kernel_wrapper_mutation":
            return "triton"
        if target in _DEVICE_EFFECTS:
            return "device"
        if target in _PURE_FUNCTIONS:
            return "pure"
    elif node["op"] == "call_method" and target in _PURE_METHODS:
        return "pure"
    raise ValueError(f"Unsupported FX node {node['op']} {target}")


def partition_decoder_graph(program):
    """Plan contiguous regions of a serialized FX graph, rejecting unknown calls.

    An effect is kept as one ordered partition even when its return value is
    unused. Its tensor references cross the boundary as shared objects: a
    later partition can read the output buffer passed to an earlier effect.
    """
    attention_targets = {
        "torch._ops." + op for op in program["splitting_ops"]
        if op in ("vllm.unified_attention", "vllm.unified_attention_with_output")
    }
    nodes = program["nodes"]
    if not nodes or nodes[-1]["op"] != "output":
        raise ValueError("Expected a terminal FX output node")
    seen = set()
    last_read = {}
    inputs = []
    calls = []
    for index, node in enumerate(nodes):
        name = node["name"]
        if name in seen:
            raise ValueError(f"Duplicate FX node {name}")
        refs = tuple(dict.fromkeys(_references((node["args"], node["kwargs"]))))
        for ref in refs:
            if ref not in seen:
                raise ValueError(f"Invalid reference {ref} in FX node {name}")
            last_read[ref] = index
        seen.add(name)
        if node["op"] == "placeholder":
            if calls or refs:
                raise ValueError(f"Unexpected placeholder {name}")
            inputs.append(name)
        elif node["op"] != "output":
            calls.append((index, name, refs, _kind(node, attention_targets)))
        elif index != len(nodes) - 1:
            raise ValueError("Only the terminal FX output is supported")

    partitions = []
    # Use a unique grouping key for effects so two adjacent writes are never
    # merged into a pure compiled region or into a single runtime call.
    for (kind, _), group in groupby(calls, lambda call: (
            call[3], None if call[3] == "pure" else call[0])):
        members = tuple(group)
        names = tuple(name for _, name, _, _ in members)
        local = set(names)
        external = tuple(dict.fromkeys(
            ref for _, _, refs, _ in members for ref in refs if ref not in local
        ))
        end = members[-1][0]
        live = tuple(name for name in names if last_read.get(name, -1) > end)
        partitions.append(Partition(kind, names, external, live))
    return tuple(inputs), tuple(partitions), nodes[-1]["args"]


@dataclass(frozen=True)
class Reference:
    name: str


def _argument(record, literal):
    if type(record) is not dict:
        return record
    keys = record.keys()
    if keys == {"ref"}:
        return Reference(record["ref"])
    if keys == {"kind", "items"}:
        items = [_argument(item, literal) for item in record["items"]]
        if record["kind"] == "tuple":
            return tuple(items)
        if record["kind"] == "list":
            return items
        raise ValueError(f"Unsupported FX argument collection {record['kind']}")
    if keys == {"slice"}:
        return slice(*(_argument(item, literal) for item in record["slice"]))
    if keys == {"kind", "value"}:
        return literal(record["kind"], record["value"])
    return {key: _argument(value, literal) for key, value in record.items()}


def _resolve(argument, bindings):
    if isinstance(argument, Reference):
        return bindings[argument.name]
    if type(argument) is tuple:
        return tuple(_resolve(item, bindings) for item in argument)
    if type(argument) is list:
        return [_resolve(item, bindings) for item in argument]
    if type(argument) is dict:
        return {key: _resolve(value, bindings) for key, value in argument.items()}
    if isinstance(argument, slice):
        return slice(*(_resolve(item, bindings) for item in (
            argument.start, argument.stop, argument.step)))
    return argument


def _named_values(names, values):
    if len(names) != len(values):
        raise ValueError(f"FX region expected {len(names)} values, got {len(values)}")
    return dict(zip(names, values))


def bind_decoder_graph(program, lower, literal, execute_pure):
    """Bind an observed graph to compiler/runtime callables.

    Lowering is resolved at bind time, before any region executes. The pure
    executor may compile and cache the region; ordered effects are dispatched
    once and share the original tensor objects, including output buffers.
    """
    inputs, partitions, exit_args = partition_decoder_graph(program)
    by_name = {node["name"]: node for node in program["nodes"]}

    def bind(partition):
        instructions = []
        for name in partition.nodes:
            node = by_name[name]
            invoke = lower(partition.kind, node)
            if not callable(invoke):
                raise TypeError(f"No callable lowering for FX node {name}")
            instructions.append((
                name, invoke, _argument(node["args"], literal),
                _argument(node["kwargs"], literal),
            ))

        def run(*values):
            bindings = _named_values(partition.inputs, values)
            for name, invoke, args, kwargs in instructions:
                bindings[name] = invoke(*_resolve(args, bindings), **_resolve(kwargs, bindings))
            return tuple(bindings[name] for name in partition.outputs)

        return partition, run

    regions = tuple(bind(partition) for partition in partitions)
    result = _argument(exit_args, literal)
    if type(result) is not tuple or len(result) != 1:
        raise ValueError("FX output must take one graph result")

    def run(*values):
        bindings = _named_values(inputs, values)
        for partition, region in regions:
            arguments = tuple(bindings[name] for name in partition.inputs)
            outputs = (execute_pure(partition, region, arguments)
                       if partition.kind == "pure" else region(*arguments))
            bindings.update(_named_values(partition.outputs, outputs))
        return _resolve(result[0], bindings)

    return run
