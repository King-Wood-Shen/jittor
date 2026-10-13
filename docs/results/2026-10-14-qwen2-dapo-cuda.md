# Qwen2-0.5B DAPO loss via ms-swift CUDA

- Status: L0–L5 passed for single-card FP32 PEFT LoRA with `swift rlhf --rlhf_type grpo --loss_type dapo` on native PyTorch and strict Jittor shim CUDA.
- Date: 2026-10-14. Jittor baseline `872918634a83581cc985e4f41f428205abbfa62b`, integrated `origin/2.0-refactor` `3495d443d9d52eb0672f91b22735a7eb585f0c46`; ms-swift checkout `88d727951203256baa564c643c651b6f8d90fd7e`, unmodified.
- Owner: ms-swift CUDA/JTorch coverage. Review when the DAPO reduction, checkpoint contract, or hardware profile changes.

## Profile and evidence

The public CLI selected ms-swift's DAPO token-window normalization branch (`WINDOW_NORMALIZED_LOSS_TYPES`) with two completions, the local Qwen2-0.5B model, eager FP32 attention, PEFT LoRA rank 4 on all linear layers, a two-row arithmetic dataset, completion-length reward and SGD. TRL 0.26.0 came from an isolated overlay. Native and shim ran in independent processes on an RTX 4090 worker. The strict shim used `use_cuda=1`, `backend_fallback="error"` and a forbidden-fallback scope. All raw commands, worker logs, captures and checkpoints live under the unversioned `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/` run keys below.

| Run key | Jobs | Evidence |
| --- | --- | --- |
| `20261014-qwen2-dapo-lora-loss-window-v1/` | preflight 23863, native 23864, shim 23866, verifier 23867 | Public CLI, eight updates, gradients and parameter/input captures. |
| `20261014-qwen2-dapo-lora-cursor-loss-v1/` | preflight 23869, native 23870, shim 23871 | Same Trainer object checkpoint-2→4. |
| `20261014-qwen2-dapo-lora-new-process-loss-v1/` | preflight 23872, native 23873, shim 23874, verifier 23875, state audit 23876 | New-process checkpoint-2→4, cursor and state comparison. |
| `20261014-qwen2-dapo-lora-steady-l5-v1/` | preflight 23879, native 23880, shim 23881, verifier 23883 | Four warmup and ten synchronized steady updates. |

L0 used the real CLI model, tokenizer, dataset, LoRA adapter, optimizer and trainer. The controlled training window loaded the same pre-existing adapter checkpoint on both sides; all 336 initial trainable parameter names, shapes and hashes matched. L1 reuses the same-input LoRA model forward at this implementation baseline from [the GRPO LoRA report](2026-10-14-qwen2-grpo-lora-cursor-cuda.md): `[2,50,151936]` CUDA logits, mean absolute error `6.80e-6`, losses `1.14405811` native and `1.14405668` shim, zero fallback. DAPO changes the policy-loss reduction after this model forward. The DAPO public CLI exercised that branch on both sides and captured its `num_items_in_batch` field. Independently sampled completion token IDs differed across runtimes, so stepwise numerical differences are diagnostic.

For L2, eight public CLI updates used `num_iterations=8`, learning rate `1e-4`, and no checkpoint save. Every update had 336/336 present, finite CUDA gradients; all 336 adapter parameter hashes changed. First-two to last-two mean loss fell from `-0.00016091` to `-0.00208366` native and from `-0.000244595` to `-0.003195475` shim. The shim start/end runtime audit recorded `cuda=1 fallback=0`.

## Restoration and public entry

The distinguishable-cursor run used prompts `2+2` and `3+5` with `num_iterations=2` and SGD learning rate `1e-5`. Each public CLI saved checkpoint-2, resumed the same Trainer object to checkpoint-4, and a separate public CLI process resumed its corresponding checkpoint-2 to checkpoint-4. Both runtimes saw `2+2,2+2,3+5,3+5` in the same-process path and `3+5,3+5` in the new-process path. The resumed input token IDs and masks matched within each runtime. Every resumed step had 336 finite CUDA gradients; native resumed loss fell `0→-4.39361e-5`, and shim same/new-process losses fell `-2.24e-7→-3.10789e-5` / `-1.12e-7→-3.04530e-5`. All 336 adapter keys changed from checkpoint-2. Same/new-process final adapters matched exactly native and within `2.79e-12` maximum absolute difference shim. Only recomputed old-token-logprob hashes differed under shim; sampled tokens and cursor did not.

The runtime hook perturbed CPU and CUDA RNG before `_load_rng_state`; restored digests matched between the same/new-process paths and differed from perturbations. The DAPO state audit loaded each runtime's own checkpoint and checked scheduler epoch `2→4`, retained optimizer parameter groups, and the saved CPU/CUDA/Python/NumPy RNG keys. The same trainer state loader's four-source saved-payload equality was separately proven at this implementation baseline by GRPO LoRA jobs 23860/23861; DAPO does not change that loader. This is the shared-mechanism evidence for Python/NumPy RNG restoration. L3 passes for this LoRA profile. Real public CLI training and saved checkpoints establish L4 with strict zero fallback.

## Steady performance and limits

The no-checkpoint public CLI L5 runs loaded the same adapter and excluded four warmup updates. Ten CUDA-synchronized `on_step_begin`–`on_step_end` samples gave native mean/median/p95 `0.112732/0.112642/0.114928` s (`8.871` steps/s) and shim `0.269216/0.269060/0.270395` s (`3.714` steps/s): mean latency ratio `2.388×`. Whole-device `nvidia-smi` memory readings peaked at 2750/2942 MiB; these are not per-process allocations. Shim fallback remained zero. This profile does not establish full-model DAPO, mixed precision, multi-GPU, or PPO support. Full-model GRPO/RLOO checkpoint coverage remains storage-limited, and PPO's separate upstream checkpoint routing issue remains open.
