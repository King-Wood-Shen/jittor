"""Native PyTorch 2.5.1 CPU/CUDA references for non-fused half AdamW."""
import json
from pathlib import Path
import unittest
import numpy as np
import jittor as jt
import torch
from _helpers import capability

class TestAdamWHalfStages(unittest.TestCase):
    def test_three_updates_match_native_moments_and_parameters(self):
        references = json.loads((Path(__file__).parent / "fixtures" /
                                 "adamw_half_native.json").read_text())
        devices = [("cpu", 0)]
        if capability.any_accelerator_enabled(backend=jt):
            devices.append(("cuda", 1))
        for device, use_cuda in devices:
            with jt.flag_scope(use_cuda=use_cuda):
                for name in ("float16", "bfloat16"):
                    p = torch.tensor([1., -2., 0., .003], dtype=getattr(torch, name),
                                     requires_grad=True, device=device)
                    opt = torch.optim.AdamW([p], lr=.001, betas=(.9, .999),
                        eps=1e-8, weight_decay=.01, foreach=False, fused=False)
                    gradients = ([.03125, -.0078125, .25, -1.5],
                                 [-.001, .125, .03125, -.2],
                                 [.25, .0001, -.03, .75])
                    for step, gradient in enumerate(gradients):
                        opt.zero_grad(set_to_none=True)
                        p.grad = torch.tensor(gradient, dtype=p.dtype, device=device)
                        opt.step()
                        values = {"param": p, "m": opt.state[p]["exp_avg"],
                                  "v": opt.state[p]["exp_avg_sq"]}
                        for key, value in values.items():
                            with self.subTest(device=device, dtype=name,
                                              step=step, field=key):
                                self.assertEqual(value.dtype, p.dtype)
                                self.assertEqual(value.device.type, device)
                                np.testing.assert_array_equal(
                                    value.detach().float().cpu().numpy(),
                                    references[device + "/" + name][step][key])
