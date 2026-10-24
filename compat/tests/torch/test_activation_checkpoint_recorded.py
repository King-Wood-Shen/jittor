"""CUDA regressions for the native recorded checkpoint frontend."""
import gc
import os
import weakref
import jittor as jt
import numpy as np
import pytest
from _helpers.checkpoint_cuda import accelerator, snapshot, curand_position as _curand_position
import torch
from jittor.autograd._checkpoint_recorded import recorded_checkpoint as recorded
from jittor._runtime.fallback import forbid_backend_fallbacks

pytestmark = pytest.mark.cuda




@pytest.mark.parametrize("input_grad", [False, True])
def test_recorded_hidden_parameter_kwargs_tree_and_update(input_grad):
    calls = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2., 3.], device="cuda"))
        def forward(self, x, *, scale):
            calls.append(bool(torch.is_grad_enabled()))
            value = (x * self.weight).exp()
            return {"value": value, "again": value, "loss": value.square().sum() * scale, "tag": "saved"}
    model = Model()
    x = torch.tensor([.2, .4], device="cuda", requires_grad=input_grad)
    result = recorded(model, x, scale=3.)
    assert result["value"] is result["again"] and result["tag"] == "saved"
    result["loss"].backward()
    exponential = np.exp(2 * np.array([.4, 1.2]))
    expected = 6 * np.array([.2, .4]) * exponential
    np.testing.assert_allclose(snapshot(model.weight.grad), expected, rtol=1e-5)
    if input_grad:
        np.testing.assert_allclose(snapshot(x.grad), 6 * np.array([2., 3.]) * exponential, rtol=1e-5)
    before = snapshot(model.weight)
    torch.optim.SGD(model.parameters(), lr=.001).step()
    np.testing.assert_allclose(snapshot(model.weight), before - .001 * expected, rtol=1e-5)
    assert calls == [True]


def test_recorded_identity_parameter_and_no_grad_contract():
    weight = torch.nn.Parameter(torch.tensor([2., 3.], device="cuda"))
    result = recorded(lambda: weight)
    assert weight.requires_grad and result.requires_grad
    gradient = jt.grad(result.sum(), weight)
    np.testing.assert_array_equal(snapshot(gradient), [1., 1.])
    with torch.no_grad():
        assert recorded(lambda value: value, weight) is weight


def test_recorded_buffer_once_and_inside_grad():
    calls = []
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([2., 3.], device="cuda"))
            self.register_buffer("counter", torch.tensor(0., device="cuda"))
        def forward(self, x):
            calls.append(1)
            self.counter.add_(1)
            value = (x * self.weight).exp()
            inner = torch.autograd.grad(value.square().sum(), value, create_graph=True, retain_graph=True)[0]
            return (inner * value).sum()
    model = Model()
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    result = recorded(model, x)
    assert model.counter.item() == 1
    dx, dw = jt.grad(result, [x, model.weight], retain_graph=True)
    exponential = np.exp(2 * np.array([.4, 1.2]))
    np.testing.assert_allclose(snapshot(dx), 4 * np.array([2., 3.]) * exponential, rtol=1e-5)
    np.testing.assert_allclose(snapshot(dw), 4 * np.array([.2, .4]) * exponential, rtol=1e-5)
    assert model.counter.item() == 1 and calls == [1]


def test_recorded_nested_repeated_backward_and_immediate_release():
    calls, refs = [], []
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    def inner(value):
        calls.append("inner")
        activation = value.exp()
        refs.append(weakref.ref(activation))
        return activation
    def outer(value):
        calls.append("outer")
        return recorded(inner, value).square().sum()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        result = recorded(outer, x)
        first = jt.grad(result, x, retain_graph=True)
        second = jt.grad(result, x, retain_graph=True)
        expected = 2 * np.exp(2 * np.array([.2, .4]))
        np.testing.assert_allclose(snapshot(first), expected, rtol=1e-5)
        np.testing.assert_allclose(snapshot(second), expected, rtol=1e-5)
        assert calls == ["outer", "inner"]
        del result, first, second
        assert all(ref() is None for ref in refs)
    finally:
        if was_enabled:
            gc.enable()


@pytest.mark.parametrize("preserve", [True, False])
def test_recorded_dropout_replay_matches_gradient_and_rng(preserve):
    torch.manual_seed(217)
    calls = []
    x = torch.tensor(np.linspace(.2, 2., 32, dtype=np.float32), device="cuda", requires_grad=True)
    weight = torch.nn.Parameter(torch.ones(32, device="cuda"))
    def forward(value):
        calls.append(1)
        return torch.nn.functional.dropout(value * weight, p=.5, training=True)
    result = recorded(forward, x, preserve_rng_state=preserve)
    observed = snapshot(result)
    before = _curand_position()[1]
    dx, dw = jt.grad(result.sum(), [x, weight])
    jt.sync([dx, dw], True, False)
    after = _curand_position()[1]
    mask = snapshot(dx)
    np.testing.assert_array_equal(snapshot(dw), snapshot(x) * mask)
    if preserve:
        np.testing.assert_array_equal(observed, snapshot(x) * mask)
        assert after == before
    else:
        assert after > before
    assert calls == [1]


@pytest.mark.parametrize("which", ["input", "parameter", "output"])
def test_recorded_rejects_mutated_boundary(which):
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor([2., 3.], device="cuda"))
    result = recorded(lambda value: (value * weight).exp(), x)
    target = {"input": x, "parameter": weight, "output": result}[which]
    with torch.no_grad():
        target.add_(1.)
    with pytest.raises(RuntimeError, match="checkpoint .* modified before backward"):
        jt.grad(result.sum(), [x, weight])


def test_recorded_forward_exception_cleans_owner():
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    def failing(value):
        value.exp().sum().item()
        raise ValueError("recorded forward sentinel")
    with pytest.raises(ValueError, match="recorded forward sentinel"):
        recorded(failing, x)
    result = recorded(lambda value: value.exp().sum(), x)
    np.testing.assert_allclose(snapshot(jt.grad(result, x)), np.exp([.2, .4]), rtol=1e-5)


def test_recorded_original_matmul_storage_is_recomputed():
    xx = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    ww = np.arange(20, dtype=np.float32).reshape(4, 5) / 20
    x = torch.tensor(xx, device="cuda", requires_grad=True)
    weight = torch.nn.Parameter(torch.tensor(ww, device="cuda"))
    jt.sync([x, weight], True, False)
    original_ids, calls = [], []
    def forward(value):
        calls.append(1)
        activation = jt.matmul(value, weight)
        original_ids.append(int(activation.id))
        return activation.square().sum()
    jt.core.step_trace_start(3)
    try:
        result = recorded(forward, x)
        cursor = len(jt.core.step_trace_memory())
        dx, dw = jt.grad(result, [x, weight])
        jt.sync([dx, dw], True, False)
        memory = jt.core.step_trace_memory()
        ops = jt.core.step_trace_ops()
        strings = jt.core.step_trace_strings()
        rows = [(i, memory[i:i+11]) for i in range(0, len(memory), 11)]
        allocations = [(i, r) for i, r in rows if r[1] == 0 and r[3] > 0 and r[7] == original_ids[0]]
        before = [(i, r) for i, r in allocations if i < cursor]
        after = [(i, r) for i, r in allocations if i >= cursor]
        assert before and after, (original_ids, allocations)
        first_index, first = before[0]
        assert any(first_index < i < after[0][0] and r[3] < 0 and r[4:6] == first[4:6] for i, r in rows)
        names = {ops[i]: strings[ops[i+3]] for i in range(0, len(ops), 12)}
        assert all("matmul" in names.get(row[6], "") for _, row in before + after)
        yy = xx @ ww
        np.testing.assert_allclose(snapshot(dx), 2 * yy @ ww.T, rtol=1e-5)
        np.testing.assert_allclose(snapshot(dw), 2 * xx.T @ yy, rtol=1e-5)
        assert calls == [1]
    finally:
        jt.core.step_trace_stop()


def test_recorded_bfloat16_autocast_matches_unwrapped_graph():
    xx = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    ww = np.arange(20, dtype=np.float32).reshape(5, 4) / 20
    outputs, gradients = [], []
    for wrapped in [False, True]:
        x = torch.tensor(xx, device="cuda", requires_grad=True)
        weight = torch.nn.Parameter(torch.tensor(ww, device="cuda"))
        def forward(value):
            return torch.nn.functional.linear(value, weight)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = recorded(forward, x) if wrapped else forward(x)
        assert result.dtype == torch.bfloat16
        loss = result.float().square().sum()
        dx, dw = jt.grad(loss, [x, weight])
        outputs.append(snapshot(result.float()))
        gradients.append((snapshot(dx.float()), snapshot(dw.float())))
    np.testing.assert_array_equal(outputs[0], outputs[1])
    for plain, rematerialized in zip(gradients[0], gradients[1]):
        np.testing.assert_array_equal(plain, rematerialized)


@pytest.mark.parametrize("frontend", ["native", "torch"])
def test_recorded_public_entry_keyword_tensor_gradient(frontend):
    checkpoint = jt.autograd.checkpoint if frontend == "native" else torch.utils.checkpoint.checkpoint
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    offset = torch.tensor([.1, .3], device="cuda", requires_grad=True)
    calls = []
    def forward(value, *, bias):
        calls.append(torch.is_grad_enabled())
        return (value + bias).exp().sum()
    result = checkpoint(forward, x, use_reentrant=False, bias=offset)
    dx, db = jt.grad(result, [x, offset])
    np.testing.assert_allclose(snapshot(dx), np.exp([.3, .7]), rtol=1e-5)
    np.testing.assert_allclose(snapshot(db), np.exp([.3, .7]), rtol=1e-5)
    assert calls == [True]
    with torch.no_grad():
        assert checkpoint(lambda value: value, x, use_reentrant=False) is x


def test_recorded_public_rejects_unsupported_custom_contexts():
    x = torch.tensor([.2, .4], device="cuda", requires_grad=True)
    with pytest.raises(NotImplementedError, match="default context"):
        jt.autograd.checkpoint(lambda value: value.exp(), x, use_reentrant=False, context_fn=lambda: None)
    with pytest.raises(NotImplementedError, match="default replay contexts"):
        torch.utils.checkpoint.checkpoint(lambda value: value.exp(), x, use_reentrant=False, debug=True)
