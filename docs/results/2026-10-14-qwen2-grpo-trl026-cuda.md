# Qwen2-0.5B GRPO on ms-swift CUDA

- Status: public native and strict shim GRPO training passed L0–L2; new-process recovery is partial; L4 entry produced a checkpoint, but formal L4/L5 acceptance awaits complete L3.
- Date: 2026-10-14
- Jittor baseline: `fa2f4ddfcb92920f35d6a4ed1b7cd164ce775512` (merged `origin/2.0-refactor` at `260185c6bae29013387325d4b8575f92906c0b83`)
- ms-swift checkout: `88d727951203256baa564c643c651b6f8d90fd7e`
- Owner: ms-swift CUDA/JTorch coverage
- Review when: GRPO same-process restore and distinguishable data cursor are verified, or the source/checkpoint contract changes.

## Profile and evidence

The stock `python -m swift.cli.main rlhf --rlhf_type grpo` path ran on an RTX 4090 through independent native PyTorch and `JITTOR_TORCH_SHIM=1` processes. The profile used the local Qwen2-0.5B model, full FP32 tuning, SGD at `1e-5`, two completions, a one-row arithmetic prompt, deterministic local completion-length reward, and `trl==0.26.0`. The isolated TRL wheel has SHA256 `16224dbc337e88b73efed286832ef3e9d58c76a25f91602f8de74a4b355c6516`; worker jobs 23110 and 23112 downloaded it without the unusable proxy and verified the overlay import and public CLI help. No ms-swift source was edited.

Raw logs, arguments, input NPZ, per-step gradient summaries and checkpoints are under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261014-qwen2-grpo-trl026-full-v2/` (native 23117, shim 23248, comparison 23266). Worker jobs 23237 and 23245 respectively cold-built and then checked the same CUDA JIT cache with `JITTOR_NO_BUILD=1`; the second job changed no core `.so` hash, and both strict device checks had zero fallback. The shim CLI's start/end audit also recorded `cuda=1`, `fallback=0`.

| Layer | Evidence and boundary |
| --- | --- |
| L0 | Native and shim CLI constructed the real model, tokenizer, dataset, optimizer and GRPO trainer. All 290 initial trainable parameters had identical names, shapes, dtypes and value hashes and were on CUDA. |
| L1 | Job 23267 loaded the model through ms-swift's `get_model_processor` API and ran the same captured token IDs and attention mask on CUDA. Full logits shape was `[2, 50, 151936]`, mean absolute difference `5.99e-6`, RMSE `9.71e-6`; losses were `1.14406240` and `1.14406264`, shim fallback zero. The public CLI's sampled completions differed, so its three-step gradient trajectories are diagnostic rather than same-input parity. |
| L2 | A separate, predeclared eight-update run reused one rollout with `num_iterations=8` (`20261014-qwen2-grpo-mu8-loss-v1`, native 23270, shim 23273, verifier 23274). First-two versus last-two mean loss fell from `-0.00026889` to `-0.003493515` natively and from `-0.000400605` to `-0.00523837` under shim. Every step had 290/290 finite nonzero CUDA gradients; 252/247 model keys changed, with update L2 norms `0.0032019233`/`0.0032019053`. Shim fallback remained zero. The earlier three-step smoke run also saved both checkpoints but had a near-zero centered GRPO loss, so it was not counted as convergence. |
| L3 | New-process public CLI resume from checkpoint 8 to step 16 completed for native 23278 and shim 23282. Verifier 23290 matched all 290 restored initial parameter hashes to the source checkpoint, found eight further CUDA gradient/update steps, scheduler epoch 8→16, changed RNG files, and falling resumed loss. This is **partial**: the single repeated prompt cannot distinguish dataloader cursor reset, and same-process restoration was not exercised. Optimizer state was read from the native PyTorch zip and shim pickle files, but a full semantic state comparison remains open. |
| L4 | The actual `swift rlhf` native and strict-shim CLI completed and saved model checkpoints at steps 3, 8 and 16. Public entry executed; formal L4 status remains pending complete L3 under this coverage protocol. |
| L5 | Not run; complete L3/L4 acceptance is a prerequisite. The reported CLI times include different JIT conditions and are not a steady-state performance comparison. |

The worker's `/home` is a per-user NFS4 export; the preflight found about 28 GB available, no applicable quota record and a successful 1 MiB fsync probe. Each checkpoint-producing stage required at least 16 GiB free and kept one checkpoint per runtime. Worker `/tmp` is node-local and was used only for short-lived sockets; the large login-node `/tmp` was not treated as shared storage. The first native attempt (23115) failed before training because a long state-root `TMPDIR` exceeded the Unix socket path limit; the corrected run used `/tmp/g${SLURM_JOB_ID}`. These are harness/storage findings, not Jittor GRPO failures.

## Repository verification and next boundary

The merged optimizer compatibility change passed worker SGD CPU/CUDA tests (4 passed), core (323 passed, 42 skipped, 1 xfailed) and layout. The full structure gate on job 23211 had 1402 passed, 6 skipped and two import-cycle failures; an archive check found the same 165 cyclic modules in the exact `origin/2.0-refactor` tree against its ratchet of 164. No GRPO-specific Jittor fix was needed. The separate NanoVector/DPO fix at `9ffe7cbd9` was reviewed but not copied into this branch because GRPO did not reproduce its slice failure.

PPO remains the next distinct RL mechanism. Its previous job 19640 verified dependency imports and `swift rlhf --help` only; it did not train. Continue from the PPO profile with a fresh run key, short worker storage/model/config preflight, then native and strict-shim public CLI. Do not infer PPO support from this GRPO result.
