"""Compatibility domains maintained by Jittor."""

from ._aliases import install_aliases as _install_aliases

# Importing this optional domain registers its historical spellings, without
# activating Torch or loading backend integrations.
_install_aliases()
del _install_aliases


def bind_observed_qwen3(graph_file, out):
    """Bind the validated Qwen3 graph for the optional vLLM adapter."""
    from .torch.installers.observed_qwen3 import bind_observed_qwen3 as bind

    return bind(graph_file, out)
