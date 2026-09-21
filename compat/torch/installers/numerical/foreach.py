"""Batched Torch interpolation over the existing tensor arithmetic owner."""
from numbers import Real

from .elementwise import lerp
from ..core import is_grad_enabled
from jittor._core.dtypes import dtype_name


def _foreach_lerp_(tensors, ends, weight):
    """Update real float32/float64 tensor sequences with a scalar weight."""
    from . import jt

    if not isinstance(tensors, (list, tuple)) or not isinstance(ends, (list, tuple)):
        raise TypeError("_foreach_lerp_ expects lists or tuples of tensors")
    if not tensors or len(tensors) != len(ends):
        raise RuntimeError("_foreach_lerp_ expects nonempty lists of equal length")
    if not isinstance(weight, Real):
        raise TypeError("_foreach_lerp_ currently supports a real scalar weight")
    # Validate all pairs before the first write; no host tensor conversion.
    for target, end in zip(tensors, ends):
        if not isinstance(target, jt.Var) or not isinstance(end, jt.Var):
            raise TypeError("_foreach_lerp_ expects tensor elements")
        if dtype_name(target.dtype) not in ("float32", "float64"):
            raise RuntimeError("_foreach_lerp_ currently supports float32 and float64")
        if target.dtype != end.dtype:
            raise RuntimeError("_foreach_lerp_ requires matching pair dtypes")
        if target.device != end.device:
            raise RuntimeError("_foreach_lerp_ requires matching pair devices")
        if tuple(target.shape) != tuple(end.shape):
            raise RuntimeError("_foreach_lerp_ currently requires matching pair shapes")
        if is_grad_enabled() and (target.requires_grad or end.requires_grad):
            raise RuntimeError("_foreach_lerp_ currently requires no_grad for trainable operands")
    for target, end in zip(tensors, ends):
        # ArrayOp initially reads a bare Python float as float32. Supply the
        # scalar in the pair's dtype before materialization, so float64 does
        # not inherit float32 rounding. This transfers only a Python scalar;
        # tensor arithmetic stays on the destination device.
        scalar = target.new_tensor(weight)
        target.copy_(lerp(target, end, scalar))
    return tensors
