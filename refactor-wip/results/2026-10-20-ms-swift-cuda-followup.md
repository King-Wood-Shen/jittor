# ms-swift CUDA follow-up: SimPO checkpointing and GRPO

- Status: GRPO two-step CUDA worker verified; SimPO gradient-checkpointing comparison blocked and closed for this attempt series.
- Date: 2026-10-20
- Baseline: `origin/2.0-refactor` at `96b6ce9129b991990a247a7ceff4769ce1a7ae21`
- Owner: ms-swift CUDA compatibility
- Review when: a materially different checkpointing harness or model path is available.

## GRPO worker 110379

`sacct` reports job and batch step `COMPLETED`, exit `0:0`; the earlier `squeue` query had no row because the job had left the active queue. The verifier artifact under the unversioned run directory records `pass=true`, both native and shim subprocesses at return code 0, and CUDA marker plus `fallback=0` on the shim. This supports the two optimizer-step execution claim; it does not establish loss parity or reward behavior because reward means were not logged.

| Runtime | Two logged losses | NVML peak | Return code | Fallback |
| --- | --- | ---: | ---: | ---: |
| Native | `[2.1e-7, 0.0182803]` | 12,924 MiB | 0 | n/a |
| Shim | `[-6.6e-7, 0.01105484]` | 17,641 MiB | 0 | 0 |

The preserved raw logs and verifier JSON are unversioned under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-grpo-cuda/`.

## SimPO gradient-checkpointing attempts v5–v12

No valid shim gradient-checkpointing-on result was produced, so there is no CUDA correctness or memory conclusion for this switch.

- v5 and v7 completed both shim runs, but the purported enabled run had `gc_events=0` and identical losses to the disabled run. The harness then established that the public switch never enabled checkpointing on the Transformers model. This is a harness/model activation failure, not evidence of a Jittor recomputation defect.
- v6 stopped before training because its `sys.orig_argv` architecture-probe guard was not observed. That guard was an invalid harness precondition.
- v8–v12 were terminated by Slurm signals before a complete enabled comparison. v12 completed the disabled shim run with losses `[1.84327781, 1.8379277, 1.83255649]`, then was terminated after starting the enabled run. The native oracle stages were reused from job 110042; their NVML peaks were 6,404 MiB (off) and 6,412 MiB (on), but they cannot substitute for shim evidence.

The distinct failures are therefore: checkpointing was not activated on the model; one harness guard did not match the runtime; and later jobs were interrupted before the target stage completed. Repeating the same public toggle in this harness cannot isolate a Jittor defect. Stop this attempt series; resume only with a harness that first proves a checkpoint recomputation event on the actual model and a full, uninterrupted enabled stage.

Raw logs and manifests remain unversioned under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-gradckpt-simpo-cuda-v5..v12/`.

## Cross-entropy CUDA placement

The retained `_CrossEntropyRows` change now uses `dispatch_context(x)` to select the CUDA device for a CPU target and its row weights. This avoids relying on the ambient `use_cuda` flag or an unresolved lazy Var `device_id`. The authoritative rerun was Slurm job 110715 on `cscg-qh04`, CUDA `cuda:0` (RTX 4090, SM 8.9), with an isolated `JITTOR_HOME` and serial first JIT. It completed `COMPLETED 0:0`. CUDA logits with a CPU target produced loss `1.5209492445` versus the independent closed-form reference `1.5209491253`; maximum gradient absolute error was `4.47e-8`; `fallback_delta=0`. The CUDA child-process shim smoke returned `child_cuda=true loss=5.0 fallback_delta=0`. NVML sampled 3,742 points at 0.1-second intervals; peak memory was 831 MiB. Probe output and raw scheduler logs are under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261021-ce-placement-slurm/`.

An earlier direct local probe on GPU 6 also passed, but it was outside the Skill-required Slurm workflow and is retained as exploratory evidence only. The Slurm result validates the targeted placement and numerical path only. It is not a training, performance, or CPU-adaptation claim.

## Shim subprocess and structural checks

After activation in a CUDA parent, a fresh child imported `torch` first, asserted the shim marker and CUDA availability, then computed a square sum on physical GPU 6. The child returned `5.0` with `fallback_delta=0`. Its unversioned run data is under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-child-shim-cuda/`.

The lazy-driver Triton unit `test_benchmarker_install_does_not_initialize_lazy_driver` passed when invoked directly through `unittest`. The repository-wide `tests/structure` command completed `1513 passed, 6 skipped, 1 failed`; the failure was `test_specs_resolve_only_native_source_and_quote_paths`, whose absolute-path check rejects the workspace's parent directory name `ms-swift-cuda` even though all listed source files are under `backends/rocm/libraries`. No ROCm code or that unrelated assertion was changed. The `compat/tests/structure/test_triton_structure.py` pytest entry could not collect from the repository root because pytest imported `compat/__init__.py` as a top-level package; the targeted lazy-driver unit was executed directly instead. These structure results are not reported as a full gate pass.
