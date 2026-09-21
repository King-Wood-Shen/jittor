"""Saved-value and leaf-identity contracts for detached aliases."""
import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return request.param


def leaf(device, values=(1., 2.)):
    return torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)


def check_grad(tensor, expected):
    torch.testing.assert_close(
        tensor.grad,
        torch.tensor(expected, dtype=tensor.dtype, device=tensor.device),
    )


def mutate(tensor):
    with torch.no_grad():
        tensor.detach().add_(10)


@pytest.mark.parametrize("operation", ["add", "mul", "exp"])
def test_saved_alias_input_mutation_without_saved_input(device, operation):
    x = leaf(device)
    if operation == "add":
        result = x + 3
        expected = torch.ones_like(x)
    elif operation == "mul":
        result = x * 3
        expected = torch.full_like(x, 3)
    else:
        result = x.exp()
        expected = result.detach().clone()
    loss = result.sum()
    mutate(x)
    loss.backward()
    torch.testing.assert_close(x.grad, expected)


def test_saved_alias_square_rejects_changed_input(device):
    x = leaf(device)
    loss = (x * x).sum()
    mutate(x)
    with pytest.raises(RuntimeError):
        loss.backward()




def test_saved_alias_nongrad_input_is_version_checked(device):
    source = leaf(device)
    detached = source.detach()
    other = leaf(device, (3., 4.))
    loss = (detached * other).sum()
    mutate(source)
    with pytest.raises(RuntimeError):
        loss.backward()


@pytest.mark.parametrize("operation", ["square", "add"])
def test_saved_alias_retained_graph_checks_only_saved_values(device, operation):
    x = leaf(device)
    loss = (x * x if operation == "square" else x + 3).sum()
    loss.backward(retain_graph=True)
    check_grad(x, [2., 4.] if operation == "square" else [1., 1.])
    mutate(x)
    if operation == "square":
        with pytest.raises(RuntimeError):
            loss.backward()
    else:
        loss.backward()
        check_grad(x, [2., 2.])


def test_saved_alias_unrelated_group_does_not_invalidate_graph(device):
    x = leaf(device)
    unrelated = leaf(device, (5., 6.))
    loss = (x * x).sum()
    mutate(unrelated)
    loss.backward()
    check_grad(x, [2., 4.])
    assert unrelated.grad is None


def test_saved_alias_broadcast_constant_is_version_checked(device):
    constant = torch.tensor([[2., 3.]], dtype=torch.float64, device=device)
    trainable = leaf(device, [[1., 2.], [3., 4.]])
    loss = (constant * trainable).sum()
    mutate(constant)
    with pytest.raises(RuntimeError):
        loss.backward()


def test_saved_alias_shape_only_backward_accepts_input_mutation(device):
    x = leaf(device, [[1., 2.], [3., 4.]])
    loss = x.sum()
    mutate(x)
    loss.backward()
    check_grad(x, [[1., 1.], [1., 1.]])


def test_saved_alias_detached_leaves_have_independent_gradients(device):
    x = leaf(device)
    other = x.detach().requires_grad_(True)
    old = x * 2
    with torch.no_grad():
        other.add_(3)
    new = other * 3
    (old + new).sum().backward()
    check_grad(x, [2., 2.])
    check_grad(other, [3., 3.])


def test_saved_alias_new_graph_preserves_existing_gradient(device):
    x = leaf(device)
    alias = x.detach()
    (x * 2).sum().backward()
    with torch.no_grad():
        alias.add_(3)
    (x * 3).sum().backward()
    check_grad(x, [5., 5.])
