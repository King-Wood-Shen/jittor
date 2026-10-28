"""Gradient-enabled native graph rematerialization ."""
from jittor._core.function import Function
from ._checkpoint_capture import Capture, flatten, unflatten


class _RecordedBoundary(Function):
    def execute(self, *originals):
        for value, enabled in zip(self.ready, self.enabled):
            if enabled:
                value.start_grad()
        return tuple(self.ready)

    def grad(self, *incoming):
        if [(v.id, v._autograd_version()) for v in self.capture.values] != self.versions:
            raise RuntimeError("checkpoint input or captured parameter was modified before backward")
        if [(v.id, v._autograd_version()) for v in self.output_guards] != self.output_versions:
            raise RuntimeError("checkpoint output was modified before backward")
        self.lease.release()
        return incoming


def recorded_checkpoint(function, *args, preserve_rng_state=True, **kwargs):
    import jittor as jt

    if jt.flags.no_grad:
        return function(*args, **kwargs)
    if jt.flags.use_cuda and jt.compiler.has_acl:
        raise NotImplementedError("native non-reentrant checkpoint RNG is not implemented for ACL")
    capture = Capture(function, args, kwargs)
    boundary = list(capture.values) + [v for _, _, v in capture.buffer_slots]
    # Input leaves must own completed storage before a retained graph reads them.
    jt.sync(boundary, True, False)
    versions = [(v.id, v._autograd_version()) for v in capture.values]
    buffers = [(owner, name, value.id, value._autograd_version())
               for owner, name, value in capture.buffer_slots]
    lease = jt.core._RematerializationLease(bool(preserve_rng_state))
    try:
        with jt.flag_scope(auto_graph_replay=0):
            result = function(*args, **kwargs)
            flat = []
            spec = flatten(result, flat, {})
            enabled = [v.dtype.is_float() and bool(v.requires_grad) for v in flat]
            if not any(enabled):
                lease.close()
                lease.release()
                return result
            # Exact native Vars survive holder rebinding. Metadata tape edges
            # keep the original autograd graph, not Python forward reexecution.
            roots = [v.tape() for v in flat]
            auxiliary = []
            for owner, name, old_id, old_version in buffers:
                current = vars(owner)[name]
                if current.id != old_id or current._autograd_version() != old_version:
                    auxiliary.append(current.tape())
            ready = jt.core._materialize_remat_outputs(roots + auxiliary)
        lease.close()
    except BaseException:
        # The native guarded destructor abandons/cleans only this frame and
        # preserves the primary exception, even before a result exists.
        del lease
        raise
    ctx = _RecordedBoundary()._new_call_context()
    ctx.capture = capture
    ctx.versions = versions
    ctx.enabled = enabled
    ctx.roots = roots + auxiliary
    ctx.ready = ready[:len(flat)]
    ctx.lease = lease
    try:
        wrapped = ctx._run_call(*roots)
        # Tape outputs have independent alias metadata from execute's ready
        # leaves. Completed inputless guards share the returned tape epochs
        # without retaining a callback/output Python ownership cycle.
        ctx.output_guards = jt.core._materialize_remat_outputs(list(wrapped))
        ctx.output_versions = [(v.id, v._autograd_version()) for v in ctx.output_guards]
        for value, enabled in zip(wrapped, enabled):
            if not enabled:
                value.stop_grad()
        return unflatten(spec, wrapped)
    except BaseException:
        lease.release()
        raise
