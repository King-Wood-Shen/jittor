"""Torch-owned defaults and arbitrary state for third-party optimizer subclasses."""
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy


def initialize_protocol(optimizer, defaults):
    optimizer._torch_custom_optimizer_protocol = True
    optimizer._torch_custom_optimizer_defaults = defaults if isinstance(defaults, dict) else dict(defaults)
    optimizer._torch_custom_optimizer_state = defaultdict(dict)


def fill_group_defaults(optimizer, group):
    for name, value in optimizer.defaults.items():
        group.setdefault(name, value)


def add_group(optimizer, group):
    import jittor as jt

    if not isinstance(group, Mapping):
        raise TypeError("optimizer parameter group must be a mapping")
    group = dict(group)
    parameters = group.get("params")
    group["params"] = [parameters] if isinstance(parameters, jt.Var) else list(parameters)
    existing = {id(p) for old in optimizer.param_groups for p in old["params"]}
    if any(id(p) in existing for p in group["params"]):
        raise ValueError("some parameters appear in more than one parameter group")
    fill_group_defaults(optimizer, group)
    optimizer.param_groups.append(group)


def state_dict(optimizer):
    import jittor as jt

    indices = {}
    groups = []
    for group in optimizer.param_groups:
        packed = {key: value for key, value in group.items() if key not in ("params", "grads")}
        identifiers = []
        for parameter in group["params"]:
            identifiers.append(indices.setdefault(id(parameter), len(indices)))
        packed["params"] = identifiers
        groups.append(packed)
    state = {(indices[id(key)] if isinstance(key, jt.Var) else key): value
             for key, value in optimizer.state.items()}
    return {"state": state, "param_groups": groups}


def _cast_state(value, parameter, key=None):
    import jittor as jt

    if isinstance(value, jt.Var):
        # Torch keeps step tensors' own dtype/device unless a particular
        # optimizer asks for capturable state; other state follows parameters.
        if key == "step":
            return value
        kwargs = {"device": parameter.device}
        if parameter.is_floating_point():
            kwargs["dtype"] = parameter.dtype
        return value.to(**kwargs)
    if isinstance(value, Mapping):
        return {name: _cast_state(child, parameter, name) for name, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cast_state(child, parameter, key) for child in value)
    if isinstance(value, list):
        return [_cast_state(child, parameter, key) for child in value]
    return value


def load_state_dict(optimizer, saved):
    if not isinstance(saved, Mapping):
        raise TypeError("loaded optimizer state must be a mapping")
    groups = saved.get("param_groups")
    state = saved.get("state")
    if not isinstance(groups, (list, tuple)) or not isinstance(state, Mapping):
        raise TypeError("loaded optimizer state needs state and parameter groups")
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("loaded state dict has a different number of parameter groups")
    bindings = {}
    for old, current in zip(groups, optimizer.param_groups):
        if not isinstance(old, Mapping) or not isinstance(old.get("params"), (list, tuple)):
            raise TypeError("loaded optimizer parameter group must contain a parameter sequence")
        if len(old["params"]) != len(current["params"]):
            raise ValueError("loaded parameter group does not match the size of optimizer's group")
        bindings.update(zip(old["params"], current["params"]))
    # Validate before copying device state or publishing any new groups. A
    # rejected checkpoint must leave the live optimizer and its tensors alone.
    copied = deepcopy(saved)
    restored = defaultdict(dict)
    for identifier, values in copied["state"].items():
        if identifier in bindings:
            parameter = bindings[identifier]
            restored[parameter] = _cast_state(values, parameter)
        else:
            restored[identifier] = values
    restored_groups = [dict(old, params=current["params"])
                       for old, current in zip(copied["param_groups"], optimizer.param_groups)]
    optimizer._torch_custom_optimizer_state = restored
    optimizer.param_groups = restored_groups
