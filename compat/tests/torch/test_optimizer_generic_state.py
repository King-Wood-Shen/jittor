"""Oracle contracts for Mapping-defaults optimizer state, CPU only."""
import copy
import unittest
from collections import defaultdict

import torch


def parameter(values=(1., -2.), dtype=torch.float32):
    return torch.nn.Parameter(torch.tensor(values, dtype=dtype))


class Accumulator(torch.optim.Optimizer):
    def __init__(self, params):
        super().__init__(params, {'lr': .1})

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state.setdefault('count', 0)
                    state['count'] += 1
                    state.setdefault('history', {'total': torch.zeros_like(p)})
                    state['history']['total'].add_(p.grad)
                    p.add_(p.grad, alpha=-group['lr'])


class TestGenericOptimizerState(unittest.TestCase):
    def test_generic_and_native_optimizer_share_accumulated_gradients(self):
        for native_first in [False, True]:
            with self.subTest(native_first=native_first):
                p = parameter()
                if native_first:
                    native = torch.optim.SGD([p], lr=.1)
                    generic = Accumulator([p])
                else:
                    generic = Accumulator([p])
                    native = torch.optim.SGD([p], lr=.1)
                (p * p).sum().backward()
                published = p.grad
                (p * p).sum().backward()
                self.assertIs(p.grad, published)
                self.assertEqual(p.grad.tolist(), [4., -8.])
                generic.step()
                native.step()
                self.assertTrue(torch.allclose(p, torch.tensor([.2, -.4]), atol=1e-6))
                self.assertEqual(set(generic.param_groups[0]), {'params', 'lr'})
                p.grad = torch.tensor([.5, .25])
                generic.zero_grad(set_to_none=False)
                self.assertEqual(p.grad.tolist(), [0., 0.])
                native.step()
                generic.zero_grad(set_to_none=True)
                self.assertIsNone(p.grad)

    def test_explicit_group_options_named_like_native_metadata(self):
        p = parameter()
        options = {'grads': {'user': 7}, '_torch_steps': [91]}
        opt = torch.optim.Optimizer([{'params': [p], **options}], {})
        saved = opt.state_dict()['param_groups'][0]
        self.assertEqual(saved['grads'], {'user': 7})
        self.assertEqual(saved['_torch_steps'], [91])

    def test_backward_does_not_inject_parameter_group_metadata(self):
        p = parameter()
        opt = Accumulator([p])
        (p * p).sum().backward()
        self.assertEqual(set(opt.param_groups[0]), {'params', 'lr'})
        opt.step()
        opt.zero_grad(set_to_none=False)
        self.assertEqual(set(opt.param_groups[0]), {'params', 'lr'})

    def test_user_group_metadata_survives_backward_assignment_and_zero(self):
        p = parameter()
        opt = Accumulator([{'params': [p], 'grads': {'user': 7}, '_torch_steps': [91]}])
        (p * p).sum().backward()
        self.assertEqual(p.grad.tolist(), [2., -4.])
        p.grad = torch.tensor([.1, .2])
        opt.step()
        opt.zero_grad(set_to_none=False)
        self.assertEqual(p.grad.tolist(), [0., 0.])
        opt.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)
        self.assertEqual(opt.param_groups[0]['grads'], {'user': 7})
        self.assertEqual(opt.state_dict()['param_groups'][0]['_torch_steps'], [91])
        payload = copy.deepcopy(opt.state_dict())
        other = Accumulator([parameter()])
        other.load_state_dict(payload)
        self.assertEqual(other.param_groups[0]['grads'], {'user': 7})
        self.assertEqual(other.param_groups[0]['_torch_steps'], [91])

    def test_lazy_state_and_arbitrary_keys(self):
        p = parameter()
        opt = torch.optim.Optimizer([p], {})
        self.assertIsInstance(opt.state, defaultdict)
        self.assertEqual(len(opt.state), 0)
        value = opt.state[p]
        self.assertIs(value, opt.state[p])
        self.assertEqual(value, {})
        value['custom'] = {'nested': [3, (4, 5)]}
        self.assertEqual(opt.state[p]['custom']['nested'], [3, (4, 5)])
        opt.state[p] = {'replacement': 9}
        self.assertEqual(opt.state[p], {'replacement': 9})
        del opt.state[p]
        self.assertNotIn(p, opt.state)

    def test_identity_distinguishes_equal_parameters_and_optimizers(self):
        p, q = parameter(), parameter()
        first = torch.optim.Optimizer([p, q], {})
        second = torch.optim.Optimizer([p], {})
        first.state[p]['counter'] = 1
        first.state[q]['counter'] = 2
        second.state[p]['counter'] = 3
        self.assertEqual([first.state[p]['counter'], first.state[q]['counter'], second.state[p]['counter']], [1, 2, 3])

    def test_state_dict_retains_state_aliases(self):
        p = parameter()
        opt = torch.optim.Optimizer([p], {'option': [1]})
        tensor = torch.tensor([3., 4.])
        opt.state[p]['nested'] = {'tensor': tensor}
        saved = opt.state_dict()
        self.assertIs(saved['state'][0], opt.state[p])
        self.assertIs(saved['state'][0]['nested']['tensor'], tensor)
        self.assertEqual(saved['param_groups'][0]['params'], [0])
        self.assertIs(saved['param_groups'][0]['option'], opt.defaults['option'])
        self.assertNotIn('lr', saved['param_groups'][0])

    def test_load_nested_state_by_group_position(self):
        p, q = parameter(), parameter()
        source = torch.optim.Optimizer([{'params': [p], 'lr': .1}, {'params': [q], 'lr': .2}], {})
        source.state[p].update(moment=torch.tensor([.2, .3]), nested={'list': [torch.tensor([4.]), 9], 'tuple': (torch.tensor([5.]), 8)}, step=torch.tensor(3.))
        source.state[q]['other'] = 17
        saved = source.state_dict()
        a, b = parameter(dtype=torch.float64), parameter(dtype=torch.float64)
        target = torch.optim.Optimizer([{'params': [b], 'lr': 8.}, {'params': [a], 'lr': 9.}], {'tag': [7]})
        defaults = target.defaults
        target.load_state_dict(saved)
        self.assertIs(target.defaults, defaults)
        self.assertIs(target.param_groups[0]['params'][0], b)
        self.assertEqual([g['lr'] for g in target.param_groups], [.1, .2])
        self.assertEqual(target.state[a], {'other': 17})
        self.assertEqual(target.state[b]['moment'].dtype, torch.float64)
        self.assertEqual(target.state[b]['nested']['list'][0].dtype, torch.float64)
        self.assertIsInstance(target.state[b]['nested']['tuple'], tuple)
        self.assertEqual(target.state[b]['step'].dtype, torch.float32)
        self.assertEqual(target.state[b]['step'].device.type, 'cpu')
        self.assertEqual(saved['state'][0]['moment'].dtype, torch.float32)
        self.assertEqual(saved['param_groups'][0]['params'], [0])
        self.assertEqual(source.state[q], {'other': 17})

    def test_step_policy_and_integer_parameter(self):
        for capturable in [False, True]:
            p = parameter(dtype=torch.float64)
            opt = torch.optim.Optimizer([{'params': [p], 'capturable': capturable}], {})
            opt.load_state_dict({'state': {0: {'step': torch.tensor(4., dtype=torch.float64), 'count': 3}}, 'param_groups': [{'params': [0], 'capturable': capturable}]})
            self.assertEqual(opt.state[p]['step'].dtype, torch.float32 if capturable else torch.float64)
            self.assertEqual(opt.state[p]['count'], 3)
        p = torch.tensor([1, 2], dtype=torch.int64)
        opt = torch.optim.Optimizer([p], {})
        opt.load_state_dict({'state': {0: {'value': torch.tensor([.5, .7])}}, 'param_groups': [{'params': [0]}]})
        self.assertEqual(opt.state[p]['value'].dtype, torch.float32)

    def test_non_parameter_state_passthrough(self):
        p = parameter()
        opt = torch.optim.Optimizer([p], {})
        metadata = {'counter': [7]}
        payload = {'state': {'global': metadata}, 'param_groups': [{'params': [0]}]}
        opt.load_state_dict(payload)
        self.assertIs(opt.state['global'], metadata)
        self.assertIs(opt.state_dict()['state']['global'], metadata)

    def test_loaded_groups_are_copied_source_unchanged(self):
        p = parameter()
        opt = torch.optim.Optimizer([p], {})
        payload = {'state': {}, 'param_groups': [{'params': [4], 'options': [2, 3]}]}
        opt.load_state_dict(payload)
        opt.param_groups[0]['options'].append(4)
        self.assertEqual(payload['param_groups'], [{'params': [4], 'options': [2, 3]}])

    def test_failed_load_preserves_object_for_observed_errors(self):
        for payload, error in [
            ({'state': {}, 'param_groups': []}, ValueError),
            ({'state': {}, 'param_groups': [{'params': []}]}, ValueError),
            ({'param_groups': [{'params': [0]}]}, KeyError),
            ({'state': [], 'param_groups': [{'params': [0]}]}, AttributeError),
        ]:
            with self.subTest(payload=payload):
                p = parameter()
                opt = torch.optim.Optimizer([p], {'lr': .1})
                opt.state[p]['counter'] = 2
                state, groups, defaults = opt.state, opt.param_groups, opt.defaults
                before = copy.deepcopy(payload)
                with self.assertRaises(error):
                    opt.load_state_dict(payload)
                self.assertIs(opt.state, state)
                self.assertIs(opt.param_groups, groups)
                self.assertIs(opt.defaults, defaults)
                self.assertEqual(opt.state[p], {'counter': 2})
                self.assertEqual(payload, before)

    def test_none_zero_gradient_and_resume_next_step(self):
        p, q = parameter(), parameter()
        opt = Accumulator([p, q])
        p.grad = torch.zeros_like(p)
        q.grad = None
        opt.step()
        self.assertEqual(opt.state[p]['count'], 1)
        self.assertNotIn(q, opt.state)
        saved = copy.deepcopy(opt.state_dict())
        a, b = parameter(), parameter()
        other = Accumulator([a, b])
        other.load_state_dict(saved)
        for x in [p, a]:
            x.grad = torch.tensor([.2, -.4])
        opt.step()
        other.step()
        self.assertTrue(torch.equal(p, a))
        self.assertTrue(torch.equal(opt.state[p]['history']['total'], other.state[a]['history']['total']))
        self.assertEqual(other.state[a]['count'], 2)

    def test_shared_parameter_backward_and_zero_grad(self):
        p = parameter()
        first, second = Accumulator([p]), Accumulator([p])
        (p * p).sum().backward()
        self.assertEqual(p.grad.tolist(), [2., -4.])
        first.step()
        second.step()
        self.assertEqual(first.state[p]['count'], 1)
        self.assertEqual(second.state[p]['count'], 1)
        self.assertIsNot(first.state[p], second.state[p])
        first.zero_grad(set_to_none=False)
        self.assertEqual(p.grad.tolist(), [0., 0.])
        second.zero_grad(set_to_none=True)
        self.assertIsNone(p.grad)


if __name__ == '__main__':
    print('REFERENCE', torch.__version__, 'CUDA', torch.version.cuda, flush=True)
    unittest.main(verbosity=2)
