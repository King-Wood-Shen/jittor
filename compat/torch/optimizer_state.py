"""State protocol for Torch optimizers constructed with a defaults mapping.

Native algorithm state remains owned by optimizer_api's existing adapters.
"""
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy
from itertools import chain

import jittor as jt

from .context import get_install_context


def state_dict(optimizer):
    identities = {}
    groups = []
    offset = 0
    for group in optimizer.param_groups:
        packed = {key: value for key, value in group.items()
                  if key != "params"}
        identities.update({id(p): index for index, p in
                           enumerate(group["params"], offset)
                           if id(p) not in identities})
        packed["params"] = [identities[id(p)] for p in group["params"]]
        offset += len(group["params"])
        groups.append(packed)
    state = {
        identities[id(key)] if isinstance(key, jt.Var) else key: value
        for key, value in optimizer.state.items()
    }
    return {"state": state, "param_groups": groups}


def _cast_state(parameter, value, param_id, groups, key=None):
    if isinstance(value, jt.Var):
        if key == "step":
            group = next(group for group in groups if param_id in group["params"])
            if group.get("capturable", False) or group.get("fused", False):
                namespace = get_install_context(jt).target_namespace
                return value.to(dtype=namespace.float32, device=parameter.device)
            return value
        if parameter.is_floating_point():
            return value.to(dtype=parameter.dtype, device=parameter.device)
        return value.to(device=parameter.device)
    if isinstance(value, dict):
        return {k: _cast_state(parameter, v, param_id, groups, key=k)
                for k, v in value.items()}
    if isinstance(value, Iterable):
        return type(value)(_cast_state(parameter, v, param_id, groups) for v in value)
    return value


def load_state_dict(optimizer, payload):
    payload = payload.copy()
    groups = optimizer.param_groups
    saved_groups = deepcopy(payload["param_groups"])
    if len(groups) != len(saved_groups):
        raise ValueError("loaded state dict has a different number of parameter groups")
    if any(len(current["params"]) != len(saved["params"])
           for current, saved in zip(groups, saved_groups)):
        raise ValueError("loaded state dict contains a parameter group that "
                         "doesn't match the size of optimizer's group")
    id_map = dict(zip(chain.from_iterable(g["params"] for g in saved_groups),
                      chain.from_iterable(g["params"] for g in groups)))
    state = defaultdict(dict)
    for key, value in payload["state"].items():
        if key in id_map:
            state[id_map[key]] = _cast_state(
                id_map[key], value, key, payload["param_groups"])
        else:
            state[key] = value
    for current, saved in zip(groups, saved_groups):
        saved["params"] = current["params"]
    optimizer._torch_generic_state = state
    optimizer.param_groups = saved_groups
    optimizer.defaults.setdefault("differentiable", False)
