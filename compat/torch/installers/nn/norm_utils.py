from ...grad import _clip_grads_with_norm_device, _get_total_norm_device
from ....diagnostics import EXPECTED, swallowed

import jittor as _jt

def _get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False,
                    foreach=None):
    del foreach
    if isinstance(tensors, _jt.Var):
        tensors = [tensors]
    tensors = list(tensors)
    # FSDP attaches the shard communicator to local gradients. Reuse its
    # group-aware reduction here too: Accelerate calls this split API for
    # each mesh before combining the resulting replicated scalar norms.
    return _get_total_norm_device(
        tensors, norm_type, error_if_nonfinite,
        shard_reduce=any(hasattr(t, "_fsdp_norm_group") for t in tensors))

def _clip_grads_with_norm_(parameters, max_norm, total_norm,
                           foreach=None):
    del foreach
    if isinstance(parameters, _jt.Var):
        parameters = [parameters]
    params = list(parameters)
    from ...tensor_state import latest_optimizer
    opt = latest_optimizer(_jt)
    grads = []
    for parameter in params:
        grad = None
        if opt is not None:
            try:
                grad = opt.find_grad(parameter)
            except EXPECTED as exc:
                swallowed("torch/installers/nn.py _clip_grads_with_norm_: grad = opt.find_grad(parameter)", exc)
                grad = None
        if grad is None:
            grad = getattr(parameter, "grad", None)
        if grad is not None:
            grads.append(grad)
    _clip_grads_with_norm_device(grads, max_norm, total_norm)
