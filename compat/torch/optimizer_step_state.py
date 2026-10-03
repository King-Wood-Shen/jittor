"""Per-parameter optimizer step counters and state-layout identity."""
import jittor as jt
from .context import get_install_context
from .. import optimizer_kinds as _optimizer_kinds


def _torch_param_steps(pg):
    params = list(pg.get("params", []))
    steps = pg.get("_torch_steps")
    if not isinstance(steps, list):
        steps = pg["_torch_steps"] = [0] * len(params)
    while len(steps) < len(params):
        steps.append(0)
    if len(steps) > len(params):
        del steps[len(params):]
    return steps


def _torch_step_tensor(opt, pg, index):
    steps = _torch_param_steps(pg)
    value = steps[index]
    if not isinstance(value, jt.Var):
        namespace = get_install_context(jt).target_namespace
        on_parameter = pg.get("capturable", False) or pg.get(
            "fused", getattr(opt, "fused", None)) is True
        device = pg["params"][index].device if on_parameter else "cpu"
        value = namespace.tensor(float(value), dtype=namespace.float32,
                                 device=device)
        steps[index] = value
    return value


def _increment_torch_step(steps, index):
    value = steps[index]
    if isinstance(value, jt.Var):
        # Keep the published scalar alive; a caller may hold or mutate it.
        with jt.flag_scope(use_cuda=int(value.location() == "device")):
            value.assign(value + 1)
    else:
        steps[index] = int(value) + 1


def _torch_optimizer_kind(opt):
    """Which optimizer's state layout `opt` has.

    Identity through the MRO, not a substring of the class name -- `SGDW`
    and `MyAdamWrapper` used to match rules they do not implement. This
    answer only describes *state layout* (which keys `state` and
    `state_dict()` expose), so unlike the FSDP2 one it does not refuse a
    subclass that overrides step(): such a subclass still keeps the base
    class's state arrays. It falls back to the lowercased class name so an
    unrecognised optimizer keeps its previous, harmless behaviour here.

    See jittor/compat/optimizer_kinds.py.
    """
    return (_optimizer_kinds.kind_of(opt)
            or type(opt).__name__.lower())


