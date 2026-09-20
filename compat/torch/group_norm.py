"""Torch GroupNorm construction over the native normalization implementation."""

def group_norm_init(self, num_groups, num_channels, eps=1e-5, affine=True,
                    device=None, dtype=None):
    import numpy as np
    from jittor._core.dtypes import dtype_name
    from .frontend import _default_tensor_dtype, tensor_frontend
    if num_channels % num_groups != 0:
        raise ValueError("num_channels must be divisible by num_groups")
    owner = type(self)._nn_frontend_owner
    owner.native_module.__init__(self)
    self.num_groups = num_groups
    self.num_channels = num_channels
    self.eps = eps
    self.affine = affine
    floating_dtype = dtype_name(dtype) if dtype is not None else _default_tensor_dtype(owner.backend)
    with tensor_frontend(owner.tensor_type, device=device):
        for name, value in (("weight", 1), ("bias", 0)):
            parameter = (owner.Parameter(owner.tensor_type(
                np.full((num_channels,), value, dtype="float32"),
                dtype=floating_dtype, device=device)) if affine else None)
            self.register_parameter(name, parameter)


def group_norm_execute(self, x):
    if x.ndim < 2:
        raise ValueError("Expected at least 2 dimensions for GroupNorm input")
    if x.shape[1] != self.num_channels:
        raise ValueError("GroupNorm: expected %s channels, got %s" % (
            self.num_channels, x.shape[1]))
    owner = type(self)._nn_frontend_owner
    return owner.backend.nn.functional.group_norm(
        x, self.num_groups, self.weight if self.weight is not None else 1,
        self.bias if self.bias is not None else 0, self.eps)
