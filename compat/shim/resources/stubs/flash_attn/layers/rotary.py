"""Flash-attn's dense rotary entry point."""

from ..ops.triton.rotary import apply_rotary as apply_rotary_emb

__all__ = ["apply_rotary_emb"]
