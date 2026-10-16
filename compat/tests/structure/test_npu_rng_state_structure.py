"""The native ACL RNG state surface must be published and owned explicitly."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NPU_INSTALLER = ROOT / "compat/torch/installers/npu.py"
CUDA_API = ROOT / "compat/torch/installers/cuda/api.py"


def test_npu_random_state_namespace_has_explicit_exact_owners():
    source = NPU_INSTALLER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(NPU_INSTALLER))
    functions = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    state_apis = {
        "get_rng_state", "get_rng_state_all", "set_rng_state",
        "set_rng_state_all", "initial_seed",
    }
    assert state_apis <= functions
    assert 'registry.publish("torch.npu.random", random, replace=True)' in source
    assert '"torch.npu.random"' in source
    assert 'Fidelity.EXACT' in source
    assert '"Native ACL RNG seed and full stream-state APIs."' in source


def test_cuda_rng_state_routes_to_native_acl_state_only_on_acl():
    source = CUDA_API.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CUDA_API))
    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_acl_rng_active" in functions
    assert "_acl_rng_api" in functions
    for name in ("_api_cuda_get_rng_state", "_api_cuda_get_rng_state_all",
                 "_api_cuda_set_rng_state", "_api_cuda_set_rng_state_all"):
        function_source = ast.get_source_segment(source, functions[name])
        assert "_acl_rng_active()" in function_source
        assert "_acl_rng_api(" in function_source

