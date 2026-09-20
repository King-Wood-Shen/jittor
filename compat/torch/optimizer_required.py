"""Required defaults for Torch-owned optimizer parameter groups."""


class _RequiredParameter:
    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


def prepare_param_group(group, defaults, tensor_type):
    """Fill defaults in insertion order, preserving the caller's group."""
    if not isinstance(group, dict):
        raise TypeError("optimizer parameter groups must be dictionaries")
    params = group["params"]
    if isinstance(params, tensor_type):
        group["params"] = [params]
    elif isinstance(params, set):
        raise TypeError("optimizer parameters need to be organized in ordered collections")
    else:
        group["params"] = list(params)
    for param in group["params"]:
        if not isinstance(param, tensor_type):
            raise TypeError("optimizer can only optimize Tensors")
    for name, default in defaults.items():
        if default is required and name not in group:
            raise ValueError(
                "parameter group didn't specify a value of required "
                "optimization parameter " + name
            )
        group.setdefault(name, default)
    return group
