"""CUDA Graph specialization for pure regions of an observed FX decoder."""
def _tensor_signature(value):
    return (tuple(value.shape), tuple(value.stride()), value.dtype, value.device)


class DecoderRegionCapture:
    """Keep GPU replay local to pure nodes; mutable effects use the caller."""

    def __init__(self, torch, capacity=8, max_token_rows=8, capture_factory=None):
        if capture_factory is None:
            from jittor._runtime.step_capture import StepCapture

            capture_factory = StepCapture
        if capacity < 1 or max_token_rows < 1:
            raise ValueError("Decoder replay limits must be positive")
        self.tensor_type = torch.Tensor
        self.capacity = capacity
        self.max_token_rows = max_token_rows
        self.capture_factory = capture_factory
        self.cache = {}
        self.refused = set()
        self.replays = 0
        self.refusals = 0

    def _signature(self, partition, values):
        dynamic = []
        signature = []
        for index, (name, value) in enumerate(zip(partition.inputs, values)):
            if isinstance(value, self.tensor_type):
                if value.device.type != "cuda":
                    return None
                layout = _tensor_signature(value)
                if name.startswith("l_self_"):
                    signature.append(("parameter", id(value), layout))
                else:
                    if value.ndim and value.shape[0] > self.max_token_rows:
                        return None
                    dynamic.append(index)
                    signature.append(("tensor", layout))
            elif type(value) in (int, float, bool, type(None)):
                signature.append(("literal", type(value), value))
            else:
                return None
        if not dynamic:
            return None
        return tuple(dynamic), (partition, tuple(signature))

    def __call__(self, partition, region, values):
        if partition.kind != "pure":
            raise ValueError("The CUDA Graph executor only accepts pure regions")
        specialization = self._signature(partition, values)
        if specialization is None:
            return region(*values)
        dynamic, key = specialization
        if key in self.refused:
            return region(*values)
        capture = self.cache.get(key)
        if capture is None:
            if len(self.cache) == self.capacity:
                # Preserve hot entries instead of allocating a graph that would
                # evict a different decoder layer on every model forward.
                return region(*values)
            dynamic_positions = set(dynamic)
            fixed = tuple(None if index in dynamic_positions else value
                          for index, value in enumerate(values))

            def invoke(*tensors):
                arguments = list(fixed)
                for index, tensor in zip(dynamic, tensors):
                    arguments[index] = tensor
                return region(*arguments)

            capture = self.capture_factory(invoke)
            self.cache[key] = capture
        previous = capture.stats["graph"]
        result = capture(*(values[i] for i in dynamic))
        self.replays += capture.stats["graph"] - previous
        if capture.refused is not None:
            self.refused.add(key)
            self.refusals += 1
        return result
