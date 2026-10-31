"""One operator for a whole parameter list's AdamW update on CUDA.

The per-parameter update is about ten graph nodes: decay, two moment updates,
two bias corrections, a square root, a division, the step, and the casts
between them. A diffusers UNet has 450 parameter tensors, so an AdamW step
built 4500 nodes in Python and sent them through the executor every
iteration -- more host time than the whole forward pass. PyTorch's default
AdamW on CUDA is a ``foreach`` kernel: a handful of launches for the list.

This hands the list to ``fused_adamw`` (src/ops/composite/fused_adamw_op.cc),
whose CUDA kernel updates every tensor of a launch from one argument table.
The arithmetic is the one ``optimizer_api._adam_step`` uses per parameter.
"""

import jittor as jt
from jittor._core.dtypes import var_dtype_name
from jittor._core.var import placement_scope_like
from jittor._runtime.dispatch import register_kernel

_DTYPES = ("float32", "float16", "bfloat16")
# Keep each mapped op small enough for nvcc and the JIT argument table.  The
# CUDA kernel can process up to 36 tensors, but a transformer optimizer group
# contains hundreds of differently shaped tensors; passing the whole group in
# one graph node makes the generated wrapper exceed the compiler's memory
# budget.  Several smaller launches retain the fused update while keeping
# compilation bounded.
_MAX_FUSED_ENTRIES = 16


def _supports(entries, *args, **kwargs):
    """Dense tensors of one supported dtype per entry: the kernel writes raw pointers."""
    if not entries:
        return False
    for parameter, moment, variance, gradient, _ in entries:
        dtype = var_dtype_name(parameter)
        if dtype not in _DTYPES:
            return False
        for tensor in (parameter, moment, variance, gradient):
            if var_dtype_name(tensor) != dtype or not tensor._storage_is_contiguous():
                return False
    return True


def _cuda_fused_adamw_updates(entries, lr, beta1, beta2, weight_decay, eps):
    """(new parameter, new moment, new variance) per entry, in entry order.

    An entry is (parameter, moment, variance, gradient, steps taken so far);
    this is the update for step ``steps + 1``, as the per-parameter path
    computes it. Entries are grouped by step, dtype, and CUDA device, since a
    fused operator cannot take tensors from different devices.

    ``lr`` may instead be a two-element float32 Var ``[step, lr]``, which the
    kernel reads on the device and which then stands for every entry's step:
    what a captured step passes, so that a replay advances the count and
    follows the learning rate without the optimizer running.
    """
    hyper = lr if isinstance(lr, jt.Var) else None
    results = [None] * len(entries)
    groups = {}
    for index, entry in enumerate(entries):
        key = (0 if hyper is not None else int(entry[4]),
               var_dtype_name(entry[0]), entry[0].device_id)
        groups.setdefault(key, []).append(index)
    for (steps, _, device_id), indices in groups.items():
        reference = entries[indices[0]][0]
        # A copy of an unplaced scalar can keep FollowRuntime placement even
        # when its device_id is the parameter's CUDA index. Construct the step
        # under the parameter's explicit placement so fused_adamw sees the same
        # backend and device for every input.
        with placement_scope_like(reference):
            if hyper is not None:
                step, lr = hyper.to_device(device_id), 0.0
            else:
                step = jt.array(float(steps + 1), dtype="float32").stop_grad()
            # An unexecuted copy can still expose its CPU source to the fused
            # op's input placement check. Finish this one small step tensor on
            # the parameter device before constructing the multi-tensor op.
            step.sync()
        for start in range(0, len(indices), _MAX_FUSED_ENTRIES):
            batch = indices[start:start + _MAX_FUSED_ENTRIES]
            count = len(batch)
            out = jt.fused_adamw(
                [entries[i][0] for i in batch], [entries[i][1] for i in batch],
                [entries[i][2] for i in batch], [entries[i][3] for i in batch],
                step, float(lr), float(beta1), float(beta2), float(weight_decay), float(eps))
            for position, index in enumerate(batch):
                results[index] = (out[position], out[count + position], out[2 * count + position])
    return results


#: Takes ``[step, lr]`` as a device Var in place of the learning rate.
_cuda_fused_adamw_updates.accepts_live_step = True

register_kernel("optim.adamw_fused", "cuda", _cuda_fused_adamw_updates, supports=_supports)
