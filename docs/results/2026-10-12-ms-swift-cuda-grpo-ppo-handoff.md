# ms-swift CUDA algorithm coverage: GRPO blocker and PPO handoff

- Status: GRPO blocked before training; PPO dependency/CLI surface preflight passed, training not run
- Date: 2026-10-12
- Baseline commit: `da9fb3b9c8d28d7665529a0c366e05d7c7c0f999`
- Owner: ms-swift CUDA/JTorch coverage
- Review when: a worker environment with `trl>=0.26` and a verified writable training store is available

## Scope and outcome

The requested next algorithm was stock `swift rlhf --rlhf_type grpo`; completed SFT, DPO, KTO, ORPO, SimPO, Reward and CPO profiles were not repeated. The existing fixed profile uses Qwen2-0.5B, full tuning, a two-sample prompt batch, two completions, a deterministic local reward plugin and three updates. The prior public CLI invocation did reach ms-swift argument construction, then stopped at `require_version('trl>=0.26')`: the shared Python 3.11 environment contains `trl==0.24.0`. No GRPO training or shim computation occurred.

The single isolated worker install attempt for `trl==0.26.0` failed because pip could not connect through the configured proxy (`ProxyError`, connection refused); the package index returned no candidates. The checked GRPO state, isolated overlay and venv pip cache contain no offline `trl>=0.26` wheel. Per the five-attempt cap, I stopped after this network-level failure and switched to PPO instead of repeating the same install. GRPO therefore remains environment-blocked, not a Jittor compatibility failure. The existing raw evidence is under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261010-qwen2-grpo-stock-cli-v1/` (Slurm job 15287) and `.../20261010-grpo-trl026-preflight-v1/` (job 15288).

Following the coverage matrix, work switched to PPO. Slurm job 19640 on `cscg-qh04` / RTX 4090 (CUDA 12.4) confirmed `trl==0.24.0` `PPOConfig`/`PPOTrainer`, ms-swift `PPOConfig`/`PPOTrainer`, and successfully invoked `python -m swift.cli.main rlhf --help`. This is a dependency and public CLI surface preflight only; it did not construct the PPO model/trainer or train. Job 19639 was an invalid preflight harness run: its CLI subprocess lacked `PYTHONPATH`; the corrected rerun is job 19640. Raw files are under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261012-qwen2-ppo-preflight-v1/`.

## L0–L5 status

| Algorithm/profile | L0 construction | L1 forward | L2 update/convergence | L3 recovery | L4 public algorithm entry | L5 performance |
| --- | --- | --- | --- | --- | --- | --- |
| GRPO fixed Qwen2 profile | blocked at dependency check | not-run | not-run | not-run | CLI invoked but training never started; not passed | not-run |
| PPO Qwen2 profile | dependency imports only; trainer/model construction not-run | not-run | not-run | not-run | CLI help only; algorithm entry not-run | not-run |

## Execution blockers and next step

The task state filesystem is shared NFS (`10.0.0.20:/home/xinshen`), with 31 GiB reported free. `quota -s` produced no applicable quota record (exit 1); a 1 MiB write probe succeeded. The mount is not verified as a user-exclusive export, so the free-space reading cannot be used as the permitted quota substitute for a checkpoint-producing run. PPO training is resource-blocked pending a verified writable store/quota. The required next worker run should use the existing Qwen2 base model, the already-produced Qwen2 Reward checkpoints and a fresh run key/cache; first repeat storage preflight, then run native and strict-shim public PPO CLI through L0–L4 before considering L5.

No implementation defect was reproduced, so no Jittor core/compat change was made. No adapter or ms-swift source was modified. No performance result is claimed.

## Repository verification

`bash tools/check_repo_layout.sh` passed. The required structure suite first timed out at 10 minutes during cold worker-side Jittor compilation (job 19641; progress 56%, no assertion result). Its 30-minute rerun (job 19649) completed with 1382 passed, 8 skipped, and 2 failed: `test_packaging_structure` detected that this report had not yet been added to generated `MANIFEST.in`, and `test_runtime_float32_matmul_precision...` could not build oneDNN because `cmake` is absent on the worker. The generated root manifest has since been refreshed and checked; the packaging structure file then passed on job 19653 (8 passed, 8 subtests passed). The oneDNN/cmake failure is an environment prerequisite and is unrelated to these documentation changes. The final full-suite rerun after the manifest refresh (worker job 19654) completed with 1383 passed, 8 skipped, and 1 failed in 182.61s; the sole failure remained the existing CPU matmul precision test, which cannot build oneDNN because `cmake` is absent. The packaging manifest assertion passed. The worker had no `nvcc`, so this repository structure gate used CPU initialization and makes no CUDA algorithm claim.
