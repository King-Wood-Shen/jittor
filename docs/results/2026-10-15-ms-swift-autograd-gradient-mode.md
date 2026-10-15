# CUDA ms-swift resume exposed a Torch autograd merge regression

- Status: The CUDA autograd regression is fixed and its directed regression suite passes. CPU and repository gates are pending.
- Date: 2026-10-15.
- Baseline: recovery HEAD `1446c41b95495d281421e69f995363d6a5a5092a`, integrating latest `origin/2.0-refactor` `626dcd6823dfd4ee45200374fa447c39e91797f2`.
- Owner: Torch compatibility / ms-swift CUDA coverage.
- Review when: autograd API ownership or create-graph behavior changes.

## First failure and cause

The worker's strict CUDA forward/backward probe passed tensor construction and forward, then failed at the first backward with `ImportError: cannot import name '_create_graph_scope' from jittor.compat.torch.autograd`. The merge had selected the latest `compat/torch/autograd.py`, which no longer exported that scope, while the automatically merged `compat/torch/installers/tensor/autograd_api.py` still imported it. The lost scope had been introduced by the earlier custom-Function higher-order-gradient fix. This was a `compat/torch` L2 autograd failure; no CUDA fallback or Jittor kernel failure was involved.

Restoring the scope and its custom-Function grad-mode handling made the scalar CUDA backward probe pass, but the directed suite found another path through the same public contract: `torch.autograd.backward(create_graph=True)` accepted the flag and dropped it when calling `Tensor.backward()`. That left custom backward callbacks with grad disabled and disconnected second derivatives. The public wrapper now forwards `create_graph` to each tensor backward call.

## Verification

All calculations and tests ran on Slurm NVIDIA workers. Job 26246 built the CUDA cache for the integrated Jittor tree and passed `jittor_utils.bootstrap --check`; its first backward probe reproduced the missing-scope failure. Worker jobs 26259 and 26260 did not execute tests because their pytest collection path imported the repository's `compat` package as a top-level package; those harness failures are retained but provide no test evidence. Job 26261 then executed the six directed tests and exposed the public-wrapper forwarding defect: four tests passed, the custom higher-order public backward case failed, and both public second-derivative tests errored. After forwarding `create_graph`, worker job 26262 passed all six tests (including the eight CUDA subcases), with strict CUDA scope and zero backend fallbacks. The direct minimal backward probe returned gradient `[2, 4]` with fallback count `0` after the scope restoration.

Raw worker logs, scripts, and the strict-cache manifest are unversioned under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261015-autograd-createg-scope-cuda-fix-v1/` through `...-v4/`. CPU, core-tier, layout and structure-gate verification remain pending.
