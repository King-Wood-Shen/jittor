"""Dense optimizer updates composed from the Torch frontend's Tensor APIs.

Scalar eager paths currently support real float32/float64 parameters. Optional
foreach, capture, fused, differentiable and sparse paths are explicit gaps.
"""
import jittor as jt

from .context import get_install_context


def _torch():
    return get_install_context(jt).target_namespace


def _nonnegative(name, value):
    if isinstance(value, jt.Var):
        raise NotImplementedError("Tensor-valued optimizer options are not implemented")
    if not 0 <= value:
        raise ValueError("Invalid " + name + ": " + str(value))


def _check_options(group):
    for name in ("foreach", "capturable", "differentiable", "fused"):
        if group.get(name, False):
            raise NotImplementedError(name + " optimizer execution is not implemented")
    if isinstance(group["lr"], jt.Var):
        raise NotImplementedError("Tensor learning rates are not implemented")
    torch = _torch()
    for parameter in group["params"]:
        if parameter.dtype not in (torch.float32, torch.float64):
            raise NotImplementedError("optimizer parameters require real float32 or float64")


def _initialize(optimizer, params, defaults):
    _torch().optim.Optimizer.__init__(optimizer, params, defaults)
    for group in optimizer.param_groups:
        _check_options(group)


def _new_step():
    torch = _torch()
    dtype = torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    return torch.tensor(0., dtype=dtype, device='cpu')


def _step(optimizer, closure, update, name):
    torch = _torch()
    loss = None
    if closure is not None:
        with torch.enable_grad():
            loss = closure()
    with torch.no_grad():
        for group in optimizer.param_groups:
            _check_options(group)
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if getattr(gradient, "is_sparse", False):
                    if name == 'Adagrad':
                        raise NotImplementedError("Adagrad sparse updates are not implemented")
                    raise RuntimeError(name + " does not support sparse gradients")
                if group["maximize"]:
                    gradient = -gradient
                if group["weight_decay"] != 0:
                    if name in ('RAdam', 'NAdam') and group['decoupled_weight_decay']:
                        parameter.copy_(parameter * (1 - group['lr'] * group['weight_decay']))
                    else:
                        gradient = gradient + parameter * group["weight_decay"]
                update(parameter, gradient, optimizer.state[parameter], group)
    return loss


def _adagrad_update(parameter, gradient, state, group):
    state['step'].add_(1)
    count = state['step'].item()
    learning_rate = group['lr'] / (1 + (count - 1) * group['lr_decay'])
    state['sum'].copy_(state['sum'] + gradient * gradient)
    denominator = state['sum'].sqrt() + group['eps']
    parameter.copy_(parameter - learning_rate * gradient / denominator)


def _adadelta_update(parameter, gradient, state, group):
    torch = _torch()
    if not state:
        state.update(step=_new_step(), square_avg=torch.zeros_like(parameter),
                     acc_delta=torch.zeros_like(parameter))
    state['step'].add_(1)
    rho = group['rho']
    state['square_avg'].copy_(state['square_avg'] * rho + gradient * gradient * (1-rho))
    denominator = (state['square_avg'] + group['eps']).sqrt()
    delta = (state['acc_delta'] + group['eps']).sqrt() / denominator * gradient
    state['acc_delta'].copy_(state['acc_delta'] * rho + delta * delta * (1-rho))
    parameter.copy_(parameter - delta * group['lr'])


def _adamax_update(parameter, gradient, state, group):
    torch = _torch()
    if not state:
        state.update(step=_new_step(), exp_avg=torch.zeros_like(parameter),
                     exp_inf=torch.zeros_like(parameter))
    state['step'].add_(1)
    beta1, beta2 = group['betas']
    state['exp_avg'].copy_(state['exp_avg'] + (gradient - state['exp_avg']) * (1-beta1))
    state['exp_inf'].copy_(torch.maximum(state['exp_inf'] * beta2, gradient.abs() + group['eps']))
    correction = 1 - beta1 ** state['step'].item()
    parameter.copy_(parameter - state['exp_avg'] / state['exp_inf'] * (group['lr']/correction))


class Adagrad:
    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0,
                 initial_accumulator_value=0, eps=1e-10, foreach=None, *,
                 maximize=False, differentiable=False, fused=None):
        for name, value in [('learning rate', lr), ('lr_decay', lr_decay),
                            ('weight_decay', weight_decay),
                            ('initial_accumulator_value', initial_accumulator_value), ('epsilon', eps)]:
            _nonnegative(name, value)
        _initialize(self, params, dict(lr=lr, lr_decay=lr_decay, eps=eps,
                    weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value,
                    foreach=foreach, maximize=maximize, differentiable=differentiable, fused=fused))
        torch = _torch()
        for group in self.param_groups:
            for parameter in group['params']:
                self.state[parameter].update(step=_new_step(),
                    sum=torch.full_like(parameter, initial_accumulator_value))

    def step(self, closure=None):
        return _step(self, closure, _adagrad_update, 'Adagrad')


class Adadelta:
    def __init__(self, params, lr=1., rho=.9, eps=1e-6, weight_decay=0,
                 foreach=None, *, capturable=False, maximize=False, differentiable=False):
        for name, value in [('learning rate', lr), ('epsilon', eps), ('weight_decay', weight_decay)]:
            _nonnegative(name, value)
        if not 0 <= rho <= 1:
            raise ValueError('Invalid rho value: ' + str(rho))
        _initialize(self, params, dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay,
                    foreach=foreach, capturable=capturable, maximize=maximize, differentiable=differentiable))

    def step(self, closure=None):
        return _step(self, closure, _adadelta_update, 'Adadelta')


class Adamax:
    def __init__(self, params, lr=2e-3, betas=(.9,.999), eps=1e-8, weight_decay=0,
                 foreach=None, *, maximize=False, differentiable=False, capturable=False):
        for name, value in [('learning rate', lr), ('epsilon', eps), ('weight_decay', weight_decay)]:
            _nonnegative(name, value)
        for index, beta in enumerate(betas):
            if not 0 <= beta < 1:
                raise ValueError('Invalid beta parameter at index ' + str(index) + ': ' + str(beta))
        _initialize(self, params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                    foreach=foreach, maximize=maximize, differentiable=differentiable, capturable=capturable))

    def step(self, closure=None):
        return _step(self, closure, _adamax_update, 'Adamax')


def _moment_defaults(lr, betas, eps, weight_decay, decoupled_weight_decay,
                     foreach, maximize, capturable, differentiable):
    for name, value in [('learning rate', lr), ('epsilon', eps), ('weight_decay', weight_decay)]:
        _nonnegative(name, value)
    for index in (0, 1):
        if not 0 <= betas[index] < 1:
            raise ValueError('Invalid beta parameter at index ' + str(index) + ': ' + str(betas[index]))
    return dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                decoupled_weight_decay=decoupled_weight_decay, foreach=foreach,
                maximize=maximize, capturable=capturable, differentiable=differentiable)


def _init_moments(parameter, state):
    if not state:
        torch = _torch()
        state.update(step=_new_step(), exp_avg=torch.zeros_like(parameter),
                     exp_avg_sq=torch.zeros_like(parameter))


def _update_moments(gradient, state, beta1, beta2):
    state['exp_avg'].copy_(state['exp_avg'] + (gradient - state['exp_avg']) * (1-beta1))
    state['exp_avg_sq'].copy_(state['exp_avg_sq'] * beta2 + gradient * gradient * (1-beta2))


def _radam_update(parameter, gradient, state, group):
    _init_moments(parameter, state)
    state['step'].add_(1)
    step = state['step'].item()
    beta1, beta2 = group['betas']
    _update_moments(gradient, state, beta1, beta2)
    correction1 = 1 - beta1 ** step
    correction2 = 1 - beta2 ** step
    average = state['exp_avg'] / correction1
    rho_inf = 2 / (1-beta2) - 1
    rho = rho_inf - 2 * step * beta2 ** step / correction2
    update = average * group['lr']
    if rho > 5:
        rectification = ((rho-4) * (rho-2) * rho_inf / ((rho_inf-4) * (rho_inf-2) * rho)) ** .5
        adaptive = correction2 ** .5 / (state['exp_avg_sq'].sqrt() + group['eps'])
        update = update * adaptive * rectification
    parameter.copy_(parameter - update)


def _nadam_update(parameter, gradient, state, group):
    if not state:
        _init_moments(parameter, state)
        state['mu_product'] = _new_step().fill_(1.)
    state['step'].add_(1)
    step = state['step'].item()
    beta1, beta2 = group['betas']
    mu = beta1 * (1 - .5 * .96 ** (step * group['momentum_decay']))
    mu_next = beta1 * (1 - .5 * .96 ** ((step+1) * group['momentum_decay']))
    state['mu_product'].copy_(state['mu_product'] * mu)
    _update_moments(gradient, state, beta1, beta2)
    denominator = (state['exp_avg_sq'] / (1-beta2 ** step)).sqrt() + group['eps']
    product = state['mu_product'].item()
    parameter.copy_(parameter + gradient / denominator * (-group['lr'] * (1-mu) / (1-product)))
    parameter.copy_(parameter + state['exp_avg'] / denominator * (-group['lr'] * mu_next / (1-product * mu_next)))


class RAdam:
    def __init__(self, params, lr=1e-3, betas=(.9,.999), eps=1e-8, weight_decay=0,
                 decoupled_weight_decay=False, *, foreach=None, maximize=False,
                 capturable=False, differentiable=False):
        defaults = _moment_defaults(lr, betas, eps, weight_decay, decoupled_weight_decay,
                                    foreach, maximize, capturable, differentiable)
        _initialize(self, params, defaults)

    def step(self, closure=None):
        return _step(self, closure, _radam_update, 'RAdam')


class NAdam:
    def __init__(self, params, lr=2e-3, betas=(.9,.999), eps=1e-8, weight_decay=0,
                 momentum_decay=4e-3, decoupled_weight_decay=False, *, foreach=None,
                 maximize=False, capturable=False, differentiable=False):
        defaults = _moment_defaults(lr, betas, eps, weight_decay, decoupled_weight_decay,
                                    foreach, maximize, capturable, differentiable)
        _nonnegative('momentum_decay', momentum_decay)
        defaults['momentum_decay'] = momentum_decay
        _initialize(self, params, defaults)

    def step(self, closure=None):
        return _step(self, closure, _nadam_update, 'NAdam')


def install(optim, base):
    for implementation in (Adagrad, Adadelta, Adamax, RAdam, NAdam):
        name = implementation.__name__
        if name not in vars(optim):
            setattr(optim, name, type(name, (implementation, base), {'__module__': 'torch.optim'}))
