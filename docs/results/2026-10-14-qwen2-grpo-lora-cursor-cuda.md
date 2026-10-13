# Qwen2-0.5B GRPO PEFT LoRA via ms-swift CUDA

- Status: L0–L5 passed for the specified single-card FP32 PEFT LoRA profile on native PyTorch and strict Jittor shim CUDA. This does not close the separate full-tuning GRPO profile's distinguishable-cursor gap.
- Date: 2026-10-14
- Jittor implementation baseline: `aed5418df5f02b5328107f7ff0475cdfbc432570`; later evidence ran after documentation-only commit `2401d4330d6e8a85d8c28697be75c9d89d76547f`. Both integrated `origin/2.0-refactor` `3495d443d9d52eb0672f91b22735a7eb585f0c46` and have the same Jittor/compat implementation tree.
- ms-swift checkout: `88d727951203256baa564c643c651b6f8d90fd7e`; no ms-swift source edits.
- Owner: ms-swift CUDA/JTorch coverage.
- Review when: affected implementation, ms-swift checkpoint format or hardware profile changes.

## Public profile and L0–L2

The real `python -m swift.cli.main rlhf --rlhf_type grpo` path ran on the same RTX 4090 worker in independent native and strict `JITTOR_TORCH_SHIM=1` processes. It used local Qwen2-0.5B, FP32 eager attention, PEFT LoRA rank 4 targeting all linear layers, two completions, two distinct arithmetic prompts, a local completion-length reward, SGD and the isolated TRL 0.26.0 overlay. The initial adapter for the controlled L2/L5 windows was the same native checkpoint-1 from the cursor profile; all 336 trainable parameter names, shapes and initial value hashes matched. Raw commands, input hashes, tensor/gradient captures and checkpoints are retained in unversioned run directories under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/`:

| Run key | Purpose |
| --- | --- |
| `20261014-qwen2-grpo-lora-cursor-v1/`, `20261014-qwen2-grpo-lora-new-process-cursor-v1/` | Initial distinguishable-cursor probes. |
| `20261014-qwen2-grpo-lora-loss-window-v1/`, `20261014-qwen2-grpo-lora-fixed-forward-v1/` | Controlled L2 and same-input L1. |
| `20261014-qwen2-grpo-lora-cursor-loss-v2/`, `20261014-qwen2-grpo-lora-new-process-loss-v2/` | Decisive same/new-process L3 with declining resumed loss. |
| `20261014-qwen2-grpo-lora-rng-full-v1/` | Four-source RNG payload restoration audit. |
| `20261014-qwen2-grpo-lora-steady-l5-v1/` | Ten-sample steady-state performance. |

Worker job 23827 loaded the same adapter via ms-swift's model API and forwarded the same captured token IDs and mask on native and strict shim CUDA. Full logits shape was `[2,50,151936]`, mean absolute difference `6.80e-6`, RMSE `1.10e-5`, and losses `1.14405811`/`1.14405668`; shim fallback was zero. This establishes controlled L0/L1 for the LoRA model. The public CLI's independently sampled completions differed, so its stepwise numerical trajectories are diagnostic rather than same-input parity.

The fixed-adapter, eight-update public CLI window used `num_iterations=8`, learning rate `1e-4` and no checkpoint save. Native 23824, shim 23825 and worker verifier 23826 passed: every step had 336/336 finite CUDA gradients; all 336 trainable parameter hashes changed. First-two versus last-two mean loss fell native `-0.000160905→-0.00208366` and shim `-0.00024466→-0.00319599`. Shim recorded `cuda=1 fallback=0`. This establishes L2 with independent per-runtime convergence.

## L3 state restoration and L4 entry

LoRA checkpoints were about 8.5 MiB, enabling a distinguishable-cursor restoration experiment within the worker's observed storage reserve. In the decisive public CLI run, `num_iterations=2` yielded the prompt order `2+2, 2+2, 3+5, 3+5`. A runtime hook stopped after step 2 and resumed the **same Trainer object** from checkpoint-2 (native 23848, shim 23850). Independent **new processes** resumed the same respective checkpoints to step 4 (native 23853, shim 23854). The first resumed prompt was `3+5` in both paths; a cursor reset would have returned to `2+2`. Verifier 23855 confirmed identical resumed token IDs and masks within each runtime and 336/336 finite CUDA gradients on all resumed steps. Native post-resume loss fell `0→-4.39584e-5`; shim fell `-8.94e-8→-3.05772e-5`. All 336 adapter keys changed after checkpoint-2; same/new-process final adapters matched exactly native and within `6.59e-12` maximum absolute difference under shim. Only shim's recomputed old-token-logprob hashes differed; sampled tokens, masks and cursor did not.

The hook forced CPU and CUDA RNG perturbations before each `_load_rng_state`; restored digests agreed between same/new-process paths and differed from the perturbations. A separate new-process public CLI audit from the same checkpoint-2 (native 23860, shim 23861) also perturbed Python and NumPy RNG, then asserted that all four restored CPU/CUDA/Python/NumPy digests exactly matched the respective saved checkpoint payload. Worker state audit 23856 loaded each runtime's own serialization and checked scheduler epoch `2→4`, two intact SGD parameter groups with constant `1e-5` learning rate and empty SGD momentum state; verifier 23855 checked the adapter and trainer global step. Qwen2's RoPE `inv_freq` buffer is reconstructed from unchanged configuration and is not persisted as mutable checkpoint state. Thus L3 passes for this LoRA profile. The actual public CLI saved checkpoint-2 and checkpoint-4 on both sides, establishing L4 with strict fallback zero.

The earlier `num_iterations=1` cursor probes (same-process 23802/23805, new-process 23808/23809, verifier 23818) also showed the `2+2→3+5→2+2` cursor and state restoration, but their centered GRPO loss remained zero. They are retained as diagnostic evidence; the subsequent two-iteration run supplies the required post-resume loss decline. Job 23847 failed before training because its generated batch script pointed to a doubled run-key suffix; corrected native job 23848 passed. State-audit attempts 23828–23830 incorrectly used native PyTorch to open shim serialization or assumed one optimizer parameter group; corrected runtime-specific audits 23831 and 23856 passed. These are harness errors, not algorithm defects.

## L5 steady-state performance

The no-checkpoint public CLI performance key `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261014-qwen2-grpo-lora-steady-l5-v1/` passed worker preflight 23834, native 23835, strict shim 23836 and verifier 23843. Both sides loaded the same adapter and two-row dataset, ran 14 updates over one rollout, excluded the first four warmup steps, and synchronized CUDA around each of the final ten `on_step_begin`–`on_step_end` intervals. Native mean/median/nearest-rank p95 latency was `0.112575`/`0.112592`/`0.114667` seconds (`8.883` steps/s); shim was `0.270329`/`0.270537`/`0.271941` seconds (`3.699` steps/s), a `2.401×` mean-latency ratio. Whole-device `nvidia-smi` readings after measured steps were 2750 MiB native and 2942 MiB shim; these are not process peak allocations. Shim recorded `cuda=1 fallback=0`. The timed interval excludes the cold first training step, initialization and checkpoint I/O; the ten measured samples were stable within each runtime.
