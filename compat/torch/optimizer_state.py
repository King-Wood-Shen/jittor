"""Live Torch optimizer state views over native parameter-group buffers."""
from collections.abc import Mapping
import jittor as jt
from .optimizer_step_state import (
    _torch_param_steps, _torch_step_tensor, _torch_optimizer_kind,
)


class _ParamState(dict):
    def __init__(self, owner, param, values):
        dict.__init__(self, values)
        self._owner = owner
        self._param = param
    def __setitem__(self, key, value):
        self._owner._set_field(self._param, key, value)
        dict.__setitem__(self, key, value)
    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        for key, value in values.items():
            self[key] = value


class _OptState:
    def __init__(self, opt):
        self._opt = opt
        self._views = opt.__dict__.setdefault("_torch_state_views", {})
    def _view(self, param, values):
        slot = self._views.get(id(param))
        if slot is None:
            state = _ParamState(self, param, values)
            self._views[id(param)] = (param, state)
        else:
            state = slot[1]
            dict.update(state, values)
        return state
    def _find(self, param):
        for pg in self._opt.param_groups:
            for i, p in enumerate(pg.get("params", [])):
                if p is param:
                    return pg, i
        return None, None
    def _params(self):
        for pg in self._opt.param_groups:
            for p in pg.get("params", []):
                marker = object()
                if self.get(p, marker) is not marker:
                    yield p
    def _reset_slot(self, pg, i):
        _torch_param_steps(pg)[i] = 0
        for key in ("m", "values", "v", "d", "pre_grad"):
            buffers = pg.get(key)
            if not isinstance(buffers, list) or i >= len(buffers):
                continue
            buffer = buffers[i]
            buffers[i] = (jt.zeros_like(buffer).stop_grad()
                          if isinstance(buffer, jt.Var) else None)
    def _sync_n_step(self):
        self._opt.n_step = max(
            (int(step) for pg in self._opt.param_groups
             for step in _torch_param_steps(pg)), default=0)
    def _set_field(self, param, key, value):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        kind = _torch_optimizer_kind(self._opt)
        if key == "step":
            _torch_param_steps(pg)[i] = (
                value if isinstance(value, jt.Var) else int(value))
            self._sync_n_step()
            return
        mappings = {
            "adam": {"exp_avg": "m", "exp_avg_sq": "values"},
            "adamw": {"exp_avg": "m", "exp_avg_sq": "values"},
            "sgd": {"momentum_buffer": "values"},
            "rmsprop": {"square_avg": "values"},
            "adan": {"exp_avg": "m", "exp_avg_sq": "v",
                     "exp_avg_diff": "d", "pre_grad": "pre_grad"},
        }
        target = mappings.get(kind, {}).get(key)
        buffers = pg.get(target) if target is not None else None
        if isinstance(buffers, list) and i < len(buffers):
            buffers[i] = value
    def get(self, param, default=None):
        pg, i = self._find(param)
        if pg is None:
            return default
        steps = _torch_param_steps(pg)
        if int(steps[i]) <= 0:
            slot = self._views.get(id(param))
            return default if slot is None else slot[1]
        kind = _torch_optimizer_kind(self._opt)
        if kind in ("adam", "adamw") and "m" in pg and "values" in pg:
            return self._view(param, {
                "exp_avg": pg["m"][i],
                "exp_avg_sq": pg["values"][i],
                "step": _torch_step_tensor(self._opt, pg, i)})
        if kind == "sgd" and "values" in pg and pg.get(
                "momentum", getattr(self._opt, "momentum", 0)):
            return self._view(param, {
                "momentum_buffer": pg["values"][i]})
        if kind == "rmsprop" and "values" in pg:
            return self._view(param, {
                "square_avg": pg["values"][i],
                "step": float(steps[i])})
        if kind == "adan":
            out = {"step": float(steps[i])}
            for source, target in (
                    ("m", "exp_avg"), ("v", "exp_avg_sq"),
                    ("d", "exp_avg_diff"),
                    ("pre_grad", "pre_grad")):
                if source in pg and i < len(pg[source]):
                    out[target] = pg[source][i]
            return self._view(param, out)
        slot = self._views.get(id(param))
        return default if slot is None else slot[1]
    def __getitem__(self, param):
        r = self.get(param, None)
        if r is None:
            if self._find(param)[0] is None:
                raise KeyError(param)
            r = self._view(param, {})
        return r
    def __setitem__(self, param, d):
        pg, i = self._find(param)
        if pg is None:
            raise KeyError(param)
        if not isinstance(d, Mapping):
            raise TypeError("optimizer state must be a mapping")
        values = dict(d)
        self._views.pop(id(param), None)
        self._view(param, values)
        self._reset_slot(pg, i)
        for key, value in values.items():
            if key != "step":
                self._set_field(param, key, value)
        self._set_field(param, "step", values.get("step", 1 if values else 0))
    def __delitem__(self, param):
        pg, i = self._find(param)
        marker = object()
        if pg is None or self.get(param, marker) is marker:
            raise KeyError(param)
        self._reset_slot(pg, i)
        self._views.pop(id(param), None)
        self._sync_n_step()
    def __contains__(self, param):
        marker = object()
        return self.get(param, marker) is not marker
    def __iter__(self):
        return self._params()
    def __len__(self):
        return sum(1 for _ in self._params())
    def keys(self):
        return list(self._params())
    def values(self):
        return [self.get(p, {}) for p in self._params()]
    def items(self):
        return [(p, self.get(p, {})) for p in self._params()]
    def get_state_dict_key(self, param):
        return self._find(param)


def _refresh_state_views(opt):
    """Update retained state dictionaries after native optimizer buffers change."""
    if not opt.__dict__.get("_torch_state_views"):
        return
    state = _OptState(opt)
    for param, _view in list(state._views.values()):
        state.get(param)


