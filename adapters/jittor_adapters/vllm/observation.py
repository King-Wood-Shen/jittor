"""Opt-in read-only evidence from vLLM's classification pooler boundary."""

import json
import os
from functools import wraps
from pathlib import Path

from jittor.compat.transaction import set_attr


def patch_pool_capture(module):
    path = os.environ.get("JITTOR_VLLM_POOL_CAPTURE")
    if not path:
        return False
    import jittor as jt
    fallback_before = int(jt.core.backend_fallback_count())
    assert fallback_before == 0, fallback_before
    runner = module.GPUModelRunner
    original = runner._pool

    @wraps(original)
    def pool(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        fallback_after = int(jt.core.backend_fallback_count())
        assert fallback_after == fallback_before == 0, (fallback_before, fallback_after)
        target = Path(path)
        if not target.exists():
            batch = self.input_batch
            assert batch.num_reqs == 1 and len(result.pooler_output) == 1
            prompt_len = int(batch.num_prompt_tokens[0])
            tokens = [int(token) for token in batch.token_ids_cpu[0, :prompt_len]]
            output = result.pooler_output[0]
            assert output is not None
            values = output.numpy().tolist()
            if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
                values = values[0]
            assert len(values) == 2
            record = {
                "request_id": result.req_ids[0],
                "prompt_token_ids": tokens,
                "pooler_output": [float(value) for value in values],
                "pooler_output_kind": "vllm classify raw logits (activation disabled)",
                "fallback_cumulative_before": fallback_before,
                "fallback_cumulative_after": fallback_after,
            }
            with target.open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
        return result

    set_attr(runner, "_pool", pool)
    return True


PATCHES = {"vllm.v1.worker.gpu_model_runner": patch_pool_capture}
