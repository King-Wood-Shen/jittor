"""Leaf identity survives gradient toggles without retroactive graph edges."""
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def make_leaf(device):
    return torch.tensor([1., 2.], dtype=torch.float64, device=device, requires_grad=True)


def check_grad(tensor, expected):
    assert tensor.grad is not None
    torch.testing.assert_close(
        tensor.grad,
        torch.tensor(expected, dtype=tensor.dtype, device=tensor.device),
    )


def write_frozen(source, alias, writer):
    with torch.no_grad():
        (source if writer == "source" else alias).add_(3)


@pytest.mark.parametrize("writer", ["source", "alias"])
def test_pre_freeze_graph_reaches_thawed_leaf(device, writer):
    source = make_leaf(device)
    alias = source.detach()
    old = source * 2
    source.requires_grad_(False)
    write_frozen(source, alias, writer)
    source.requires_grad_(True)
    old.sum().backward()
    check_grad(source, [2., 2.])
    assert alias.grad is None
    assert not alias.requires_grad


@pytest.mark.parametrize("writer", ["source", "alias"])
def test_frozen_operations_do_not_gain_edges_after_thaw(device, writer):
    source = make_leaf(device)
    alias = source.detach()
    old = source * 2
    source.requires_grad_(False)
    write_frozen(source, alias, writer)
    untracked = source * 7
    assert not untracked.requires_grad
    source.requires_grad_(True)
    # Only the old source version has an autograd edge in this graph.
    (old + untracked).sum().backward()
    check_grad(source, [2., 2.])
    assert not untracked.requires_grad
    assert alias.grad is None


@pytest.mark.parametrize("writer", ["source", "alias"])
def test_independent_detached_leaf_survives_source_toggles(device, writer):
    source = make_leaf(device)
    independent = source.detach().requires_grad_(True)
    old_source = source * 2
    old_independent = independent * 5
    source.requires_grad_(False)
    assert independent.requires_grad
    write_frozen(source, independent, writer)
    source.requires_grad_(True)
    assert independent.requires_grad
    # Distinct leaf identities each appear once, despite sharing storage.
    (old_source + old_independent).sum().backward()
    check_grad(source, [2., 2.])
    check_grad(independent, [5., 5.])


def test_repeated_toggles_and_both_writers_preserve_original_leaf(device):
    source = make_leaf(device)
    alias = source.detach()
    old = source * 2
    for writer in ["alias", "source", "alias", "source"]:
        source.requires_grad_(False)
        write_frozen(source, alias, writer)
        source.requires_grad_(True)
        assert not alias.requires_grad
    old.sum().backward()
    check_grad(source, [2., 2.])
    assert alias.grad is None


@pytest.mark.parametrize("writer", ["source", "alias"])
def test_retained_old_loss_accumulates_after_freeze_write_thaw(device, writer):
    source = make_leaf(device)
    alias = source.detach()
    old_loss = (source * 2).sum()
    old_loss.backward(retain_graph=True)
    check_grad(source, [2., 2.])
    source.requires_grad_(False)
    write_frozen(source, alias, writer)
    source.requires_grad_(True)
    old_loss.backward()
    check_grad(source, [4., 4.])
    assert alias.grad is None


def test_current_frozen_leaf_stays_frozen_during_other_leaf_backward(device):
    source = make_leaf(device)
    other = make_leaf(device)
    alias = source.detach()
    loss = (source * 2 + other * 3).sum()
    source.requires_grad_(False)
    write_frozen(source, alias, "alias")
    loss.backward()
    assert source.grad is None
    assert not source.requires_grad
    check_grad(other, [3., 3.])


def test_version_error_does_not_break_later_legal_old_graph(device):
    source = make_leaf(device)
    alias = source.detach()
    nonlinear = (source * source).sum()
    linear = (source * 2).sum()
    source.requires_grad_(False)
    write_frozen(source, alias, "alias")
    source.requires_grad_(True)
    with pytest.raises(RuntimeError):
        nonlinear.backward()
    assert source.grad is None
    linear.backward()
    check_grad(source, [2., 2.])
