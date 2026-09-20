"""Torch BatchNorm module protocol over the shared native normalization math."""
from collections import OrderedDict

import numpy as np

from jittor._core.dtypes import dtype_name
from .frontend import tensor_frontend


def _constant(owner, shape, value, dtype, device):
    # The installation-owned Tensor constructor preserves scalar rank, int64,
    # device placement and frontend identity without rebinding the native Var.
    return owner.tensor_type(np.full(shape, value, dtype=dtype),
                             dtype=dtype, device=device)


def batch_norm_init(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                    track_running_stats=True, device=None, dtype=None):
    from .frontend import _default_tensor_dtype
    owner = type(self)._nn_frontend_owner
    owner.native_module.__init__(self)
    self.num_features = num_features
    self.eps = eps
    self.momentum = momentum
    self.affine = affine
    self.track_running_stats = track_running_stats
    self.is_train = True
    self.sync = False
    floating_dtype = dtype_name(dtype) if dtype is not None else _default_tensor_dtype(owner.backend)
    with tensor_frontend(owner.tensor_type, device=device):
        for name, value in (("weight", 1), ("bias", 0)):
            parameter = (owner.Parameter(_constant(
                owner, (num_features,), value, floating_dtype, device))
                if affine else None)
            self.register_parameter(name, parameter)
        for name, value in (("running_mean", 0), ("running_var", 1)):
            buffer = (_constant(owner, (num_features,), value, floating_dtype, device)
                      if track_running_stats else None)
            self.register_buffer(name, buffer)
        counter = (_constant(owner, (), 0, "int64", device)
                   if track_running_stats else None)
        self.register_buffer("num_batches_tracked", counter)


def batch_norm_reset_running_stats(self):
    if self.track_running_stats:
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.num_batches_tracked.zero_()


def batch_norm_reset_parameters(self):
    batch_norm_reset_running_stats(self)
    if self.affine:
        owner = type(self)._nn_frontend_owner
        with owner.backend.no_grad():
            self.weight.fill_(1)
            self.bias.zero_()


def batch_norm_execute(self, x):
    dimensions = type(self)._batch_norm_input_dimensions
    if dimensions is not None and x.ndim not in dimensions:
        raise ValueError("expected {}D input (got {}D input)".format(
            " or ".join(str(value) for value in dimensions), x.ndim))
    if x.ndim < 2:
        raise ValueError("expected at least 2D input")
    factor = 0.0 if self.momentum is None else self.momentum
    if self.training and self.track_running_stats and self.num_batches_tracked is not None:
        self.num_batches_tracked.add_(1)
        if self.momentum is None:
            factor = 1.0 / self.num_batches_tracked.item()
    training = self.training or (self.running_mean is None and self.running_var is None)
    if training:
        count = int(x.shape[0])
        for dimension in x.shape[2:]:
            count *= int(dimension)
        if count == 1:
            raise ValueError("Expected more than 1 value per channel when training")
    running_mean = self.running_mean if not self.training or self.track_running_stats else None
    running_var = self.running_var if not self.training or self.track_running_stats else None
    owner = type(self)._nn_frontend_owner
    return owner.backend.nn.functional.batch_norm(
        x, running_mean, running_var,
        self.weight if self.weight is not None else 1,
        self.bias if self.bias is not None else 0,
        training=training, momentum=factor, eps=self.eps)


def make_batch_norm_type(owner, native, public_name):
    base = owner.adapters.get(native)
    if base is None:
        base = type("BatchNorm", (native, owner.Module), {
            "__module__": "torch.nn", "__slots__": (),
            "__init__": batch_norm_init, "execute": batch_norm_execute,
            "reset_running_stats": batch_norm_reset_running_stats,
            "reset_parameters": batch_norm_reset_parameters,
            "_torch_native_layer": native, "_version": 2,
            "_batch_norm_input_dimensions": None,
        })
        owner.adapters[native] = base
    if public_name == "BatchNorm":
        return base
    key = (native, public_name)
    adapted = owner.adapters.get(key)
    if adapted is None:
        dimensions = {"BatchNorm1d": (2, 3), "BatchNorm2d": (4,), "BatchNorm3d": (5,)}
        adapted = type(public_name, (base,), {
            "__module__": "torch.nn", "__slots__": (),
            "_batch_norm_input_dimensions": dimensions[public_name],
        })
        owner.adapters[key] = adapted
    return adapted


def is_frontend_batch_norm(module):
    owner = getattr(type(module), "_nn_frontend_owner", None)
    if owner is None:
        return False
    base = owner.adapters.get(owner.backend.nn.BatchNorm)
    return base is not None and isinstance(module, base)


def prepare_legacy_batch_norm_state(root, state_dict):
    """Supply pre-v2 counters without changing a caller's mapping or metadata."""
    metadata = getattr(state_dict, "_metadata", {})
    prepared = state_dict
    for prefix, module in root.named_modules(remove_duplicate=False):
        if not is_frontend_batch_norm(module) or not module.track_running_stats:
            continue
        version = metadata.get(prefix, {}).get("version")
        key = (prefix + "." if prefix else "") + "num_batches_tracked"
        if (version is None or version < 2) and key not in state_dict:
            if prepared is state_dict:
                prepared = OrderedDict(state_dict)
                if hasattr(state_dict, "_metadata"):
                    prepared._metadata = metadata
            prepared[key] = module.num_batches_tracked.detach().clone()
    return prepared
