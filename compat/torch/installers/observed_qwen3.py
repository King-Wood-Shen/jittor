"""Bind the observed single-device Qwen3 decoder to the Jittor runtime."""
import json
import os
from types import SimpleNamespace

from ...transaction import set_attr


def bind_observed_qwen3(graph_file, out):
    """Install the verified Qwen3 graph and return its execution counters."""
    import torch
    import jittor
    from jittor.compat.torch.installers.compile_graph_runtime import bind_placeholder_sources, bind_vllm_decoder
    from jittor.compat.torch.installers.compile_graph_capture import DecoderRegionCapture
    from vllm.config import CompilationLevel
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.models.qwen3 import Qwen3Model

    assert torch.owner is jittor and jittor.flags.use_cuda == 1
    assert jittor.flags.auto_graph_replay == 0
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") == "0"
    program = json.loads(graph_file.read_text())
    assert program["vllm"] == "0.11.0" and len(program["nodes"]) == 4230
    resolve = bind_placeholder_sources(program)
    total = {"forward": 0, "attention": 0, "pure_regions": 0, "profile_attention": 0, "decode_attention": 0}

    def unsupported(*args):
        raise RuntimeError("Unexpected device or Triton effect in the verified Qwen3 graph")

    captures = DecoderRegionCapture(torch, capacity=3)

    def run_pure(partition, region, args):
        total["pure_regions"] += 1
        if get_forward_context().attn_metadata is None:
            return region(*args)
        return captures(partition, region, args)

    attention = torch.ops.vllm.unified_attention_with_output

    def checked_attention(query, key, value, output, layer_name, *args, **kwargs):
        context = get_forward_context()
        assert layer_name in context.no_compile_layers
        assert all(x.device.type == "cuda" for x in (query, key, value, output))
        total["attention"] += 1
        total["profile_attention" if context.attn_metadata is None else "decode_attention"] += 1
        return attention(query, key, value, output, layer_name, *args, **kwargs)

    frontend = SimpleNamespace(
        nn=torch.nn, cat=torch.cat, chunk=torch.chunk, empty_like=torch.empty_like,
        rsqrt=torch.rsqrt, zeros=torch.zeros, device=torch.device, bfloat16=torch.bfloat16,
        float32=torch.float32, int64=torch.int64,
        ops=SimpleNamespace(vllm=SimpleNamespace(unified_attention_with_output=checked_attention)),
    )
    execute = bind_vllm_decoder(program, frontend, unsupported, unsupported, run_pure)

    def recorded_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        assert not self.do_not_compile
        assert self.vllm_config.compilation_config.level == CompilationLevel.PIECEWISE
        assert "_jittor_decoder_compiled_callable" in self.__dict__
        assert intermediate_tensors is None and inputs_embeds is None
        assert self.config.model_type == "qwen3" and self.config.hidden_size == 4096
        assert self.start_layer == 0 and self.end_layer == 36
        assert self.quant_config is None and not self.aux_hidden_state_layers
        assert input_ids.ndim == positions.ndim == 1 and input_ids.shape == positions.shape
        assert input_ids.device.type == positions.device.type == "cuda"
        previous = total.copy()
        result, = execute(*resolve({"self": self, "input_ids": input_ids, "positions": positions}))
        assert total["attention"] - previous["attention"] == 36
        assert total["pure_regions"] - previous["pure_regions"] == 37
        assert result.device.type == "cuda" and tuple(result.shape) == (input_ids.shape[0], 4096)
        total["forward"] += 1
        with (out/"regions.jsonl").open("a") as stream:
            stream.write(json.dumps({"tokens": input_ids.shape[0], **total})+"\n")
        return result

    recorded_forward._jittor_fx_lowered = True
    set_attr(Qwen3Model, "forward", recorded_forward)
    return total, captures
