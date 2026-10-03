"""Lazy optimizer state must survive inspection, updates and checkpoint load."""
import contextlib
import unittest
import torch


class TestOptimizerEmptyState(unittest.TestCase):
    def test_empty_state_is_live_and_round_trips(self):
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            for kind in ("AdamW", "SGD"):
                scope = contextlib.nullcontext()
                if hasattr(torch, "_torch_compat_install_context"):
                    import jittor as jt
                    scope = jt.runtime.scope(use_cuda=int(device == "cuda"))
                with self.subTest(device=device, kind=kind), scope:
                    self._check(device, kind)

    def _check(self, device, kind):
        p = torch.tensor([1., 2.], device=device, requires_grad=True)
        opt = getattr(torch.optim, kind)([p], lr=.01)
        self.assertIsNone(opt.state.get(p))
        self.assertNotIn(p, opt.state)
        state = opt.state[p]
        self.assertEqual(state, {})
        self.assertIn(p, opt.state)
        self.assertEqual(len(opt.state), 1)
        self.assertIs(state, opt.state[p])
        saved = opt.state_dict()
        self.assertEqual(saved["state"], {0: {}})
        q = torch.tensor([1., 2.], device=device, requires_grad=True)
        other = getattr(torch.optim, kind)([q], lr=.01)
        other.load_state_dict(saved)
        self.assertIn(q, other.state)
        self.assertEqual(other.state[q], {})
        p.square().sum().backward()
        opt.step()
        self.assertIs(state, opt.state[p])
        if kind == "AdamW":
            self.assertEqual(set(state), {"step", "exp_avg", "exp_avg_sq"})
            self.assertEqual(int(state["step"]), 1)
        else:
            self.assertEqual(state, {})
            self.assertEqual(opt.state_dict()["state"], {0: {}})
        del opt.state[p]
        self.assertNotIn(p, opt.state)
        self.assertEqual(opt.state_dict()["state"], {})


if __name__ == "__main__":
    unittest.main()
