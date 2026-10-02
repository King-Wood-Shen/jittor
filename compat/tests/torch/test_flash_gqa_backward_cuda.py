"""Official CUDA FlashAttention GQA forward/backward against an analytic reference."""
import os
import unittest

import numpy as np
import jittor as jt
import torch

from _helpers import capability


@unittest.skipUnless(
    capability.check_accelerator("cuda", backend=jt).enabled
    and bool(os.environ.get("JITTOR_FLASH_ATTN_JITTOR_SRC"))
    and "bf16" in os.environ.get("JITTOR_FLASH_ATTN_DTYPES", ""),
    "official BF16 CUDA FlashAttention is unavailable",
)
class TestFlashGQABackward(unittest.TestCase):
    def test_causal_gqa_gradients_and_native_dispatch(self):
        from jittor._runtime.fallback import forbid_backend_fallbacks
        from jittor.compat import diagnostics
        from jittor.compat.torch.installers.nn import attention as attention_impl

        rng = np.random.RandomState(71)
        q = rng.normal(size=(1, 8, 32, 64)).astype(np.float32) * 0.2
        k = rng.normal(size=(1, 2, 32, 64)).astype(np.float32) * 0.2
        v = rng.normal(size=(1, 2, 32, 64)).astype(np.float32) * 0.2
        w = rng.normal(size=q.shape).astype(np.float32) * 0.2
        k4, v4 = np.repeat(k, 4, axis=1), np.repeat(v, 4, axis=1)
        scores = q @ np.swapaxes(k4, -1, -2) * 0.125
        mask = np.arange(32)[:, None] < np.arange(32)[None, :]
        scores = np.where(mask, -1e9, scores)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)
        ref_out = probs @ v4
        dv4 = np.swapaxes(probs, -1, -2) @ w
        dp = w @ np.swapaxes(v4, -1, -2)
        ds = probs * (dp - (dp * probs).sum(axis=-1, keepdims=True))
        dq = (ds @ k4) * 0.125
        dk4 = (np.swapaxes(ds, -1, -2) @ q) * 0.125
        dk = dk4.reshape(1, 2, 4, 32, 64).sum(axis=2)
        dv = dv4.reshape(1, 2, 4, 32, 64).sum(axis=2)

        with jt.flag_scope(use_cuda=1), jt.runtime.scope(backend_fallback="error"), forbid_backend_fallbacks():
            owner = attention_impl.jt
            diagnostics.set_sdpa_flash_stats(
                {"hits": 0, "misses": {}, "casts": {}, "backend": None}, owner=owner)
            torch._torch_sdpa_flash_backend_cache.clear()
            before = jt.core.backend_fallback_count()
            tensors = [torch.tensor(x, dtype=torch.bfloat16, device="cuda", requires_grad=True)
                       for x in (q, k, v)]
            result = torch.nn.functional.scaled_dot_product_attention(
                *tensors, is_causal=True, scale=0.125, enable_gqa=True)
            (result.float() * torch.tensor(w, device="cuda")).sum().backward()
            actual = [result.detach().float().cpu().numpy()] + [
                tensor.grad.detach().float().cpu().numpy() for tensor in tensors]
            self.assertEqual(jt.core.backend_fallback_count() - before, 0)
            self.assertGreaterEqual(diagnostics.sdpa_flash_stats(owner)["hits"], 1)
        for name, got, ref in zip(("out", "q", "k", "v"), actual, (ref_out, dq, dk, dv)):
            np.testing.assert_allclose(got, ref, atol=2e-2, rtol=2e-2, err_msg=name)


if __name__ == "__main__":
    unittest.main()
