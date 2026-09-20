"""Independent optimizer types reusing native update implementations."""

from types import ModuleType, MappingProxyType
from collections.abc import Mapping
from collections import defaultdict
from .frontend import tensor_frontend
from .context import get_install_context
import jittor as jt
from .optimizer_required import prepare_param_group, required


def initialize_base(self, params, lr, *args, **kwargs):
    state = get_install_context(jt).state["optimizer_frontend_native"]
    tensor_type = state["tensor_type"]
    if isinstance(params, tensor_type._frontend_backend.Var):
        raise TypeError("optimizer params must be an iterable of tensors or parameter groups")
    params = list(params)
    if isinstance(lr, Mapping):
        if not params:
            raise ValueError("optimizer got an empty parameter list")
        self._torch_defaults = lr
        self._torch_generic_state = defaultdict(dict)
        self.param_groups = []
        groups = params if isinstance(params[0], dict) else [{"params": params}]
        for group in groups:
            self.add_param_group(group)
        base_lr = lr.get("lr", 0.0)
        if base_lr is required:
            base_lr = self.param_groups[0]["lr"]
        with tensor_frontend(tensor_type):
            state["base_init"](self, self.param_groups, base_lr, *args, **kwargs)
        return
    params = [dict(group, params=([group["params"]]
                                 if isinstance(group["params"], tensor_type._frontend_backend.Var)
                                 else list(group["params"])))
              if isinstance(group, Mapping) else group for group in params]
    with tensor_frontend(tensor_type):
        state["base_init"](self, params, lr, *args, **kwargs)


def optimizer_defaults(self):
    if "_torch_defaults" in self.__dict__:
        return self._torch_defaults
    return jt.optim.Optimizer.defaults.__get__(self, type(self))


def add_param_group(self, group):
    if "_torch_defaults" in self.__dict__:
        state = get_install_context(jt).state["optimizer_frontend_native"]
        tensor_type = state["tensor_type"]._frontend_backend.Var
        prepare_param_group(group, self._torch_defaults, tensor_type)
    return jt.optim.Optimizer.add_param_group(self, group)


def validate_foreach_option(value):
    # Do not collapse arbitrary truthy/falsy objects into supported options.
    if value is not None and value is not False:
        raise NotImplementedError('only foreach=None or foreach=False execution is implemented')


def validate_foreach_groups(instance, groups):
    if getattr(instance, '_torch_foreach_protocol', False):
        for group in groups:
            validate_foreach_option(group.get('foreach', instance.foreach))


def _initialize_foreach(instance, value):
    validate_foreach_option(value)
    instance._torch_foreach_protocol = True
    instance.foreach = value


def _initialize_algorithm(name, instance, args, kwargs):
    state = get_install_context(jt).state["optimizer_frontend_native"]
    with tensor_frontend(state["tensor_type"]):
        state["algorithms"][name](instance, *args, **kwargs)
    if getattr(instance, '_torch_foreach_protocol', False):
        validate_foreach_groups(instance, instance.param_groups)
        for group in instance.param_groups:
            group.setdefault('foreach', instance.foreach)


def _add_algorithm_group(name, instance, group):
    validate_foreach_groups(instance, [group])
    group.setdefault('foreach', instance.foreach)
    state = get_install_context(jt).state['optimizer_frontend_native']
    with tensor_frontend(state['tensor_type']):
        return state['add_groups'][name](instance, group)


def add_sgd_param_group(self, group):
    return _add_algorithm_group('SGD', self, group)


def add_adamw_param_group(self, group):
    for key in ("lr", "betas", "eps", "weight_decay"):
        group.setdefault(key, self.defaults[key])
    return _add_algorithm_group('AdamW', self, group)


def initialize_sgd(self, params, lr, momentum=0, weight_decay=0, dampening=0, nesterov=False, *, foreach=None):
    _initialize_foreach(self, foreach)
    return _initialize_algorithm("SGD", self,
        (params, lr, momentum, weight_decay, dampening, nesterov), {})


def initialize_adam(self, *args, **kwargs):
    return _initialize_algorithm("Adam", self, args, kwargs)


def initialize_adamw(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.01, amsgrad=False, *, foreach=None, **kwargs):
    if amsgrad is not False:
        raise NotImplementedError("AdamW amsgrad=True is not implemented")
    _initialize_foreach(self, foreach)
    self.amsgrad = False
    result = _initialize_algorithm("AdamW", self, (params,), dict(
        kwargs, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
    for group in self.param_groups:
        for key in ("lr", "betas", "eps", "weight_decay"):
            group.setdefault(key, self.defaults[key])
    return result


def initialize_rmsprop(self, *args, **kwargs):
    return _initialize_algorithm("RMSprop", self, args, kwargs)


def initialize_adan(self, *args, **kwargs):
    return _initialize_algorithm("Adan", self, args, kwargs)


_ALGORITHM_INITIALIZERS = {
    "SGD": initialize_sgd, "Adam": initialize_adam, "AdamW": initialize_adamw,
    "RMSprop": initialize_rmsprop, "Adan": initialize_adan,
}


def make_optimizer_frontend(native, tensor_type):
    module = ModuleType("torch.optim")
    module.__package__ = "torch.optim"
    module.__path__ = []
    setattr(module, "_native_optimizer_module", native)
    get_install_context(tensor_type._frontend_backend).state["optimizer_frontend_native"] = MappingProxyType({
        "tensor_type": tensor_type,
        "base_init": native.Optimizer.__init__,
        "algorithms": MappingProxyType({name: getattr(native, name).__init__
                                       for name in _ALGORITHM_INITIALIZERS if hasattr(native, name)}),
        "add_groups": MappingProxyType({name: getattr(native, name).add_param_group
                                       for name in ('SGD', 'AdamW') if hasattr(native, name)}),
    })

    base = type("Optimizer", (native.Optimizer,), {
        "__module__": "torch.optim",
        "__init__": initialize_base,
        "defaults": property(optimizer_defaults),
        "add_param_group": add_param_group,
    })
    setattr(module, "Optimizer", base)
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "Adan"):
        algorithm = getattr(native, name, None)
        if algorithm is None:
            continue
        # Native super() calls traverse this MRO through the frontend base,
        # whose state/closure adapters can be installed without changing the
        # original Optimizer or any native algorithm class dictionary.
        namespace = {
            "__module__": "torch.optim",
            "__init__": _ALGORITHM_INITIALIZERS[name],
        }
        if name == 'SGD':
            namespace['add_param_group'] = add_sgd_param_group
        elif name == 'AdamW':
            namespace['add_param_group'] = add_adamw_param_group
        setattr(module, name, type(name, (algorithm, base), namespace))
    for name in ("opt_grad", "LRScheduler", "LambdaLR"):
        if hasattr(native, name):
            setattr(module, name, getattr(native, name))
    setattr(module, "__all__", [name for name in vars(module) if not name.startswith("_")])
    return module
