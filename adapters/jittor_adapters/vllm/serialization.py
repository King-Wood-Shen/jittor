"""Keep vLLM's strict tensor wire format on Jittor-backed tensors."""

from jittor.compat.transaction import set_attr


def patch_msgpack_tensor_shape(module):
    """Encode Jittor NanoVector dimensions as native msgpack integers.

    vLLM's tensor encoder already handles the data and dtype. Its final
    ``obj.shape`` is a tuple under PyTorch but a NanoVector under Jittor.
    Leave all other encoder behavior, including the strict fallback policy,
    unchanged.
    """
    encoder = module.MsgpackEncoder
    original = encoder._encode_tensor

    def encode_tensor(self, obj):
        dtype, shape, data = original(self, obj)
        return dtype, tuple(int(dim) for dim in shape), data

    set_attr(encoder, "_encode_tensor", encode_tensor)


PATCHES = {"vllm.v1.serial_utils": patch_msgpack_tensor_shape}
