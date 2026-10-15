# ms-swift CUDA checkpoint continuation and full-tuning performance

- Status: Full-tuning GRPO L3–L5 and CISPO/SAPO PEFT LoRA L3 pass for the measured Qwen2-0.5B FP32 profile. This does not change the recorded upstream FIPO/REAL teacher guard or PPO checkpoint-routing boundaries.
- Date: 2026-10-15, Asia/Shanghai.
- Measurement tree: `5585b416ca5a3c053b2b545286d79c9dfc3ad369`, after integrating `origin/2.0-refactor` `bd17d080ee31c8cb84f3ac2617662f423c343fec`. The subsequent `origin/2.0-refactor` `7a1e66b3858279b40eaec0c4895c6059ce8eb8db` changed only `MANIFEST.in` and `docs/results/index.md`; no measured source changed.
- ms-swift checkout: `88d727951203256baa564c643c651b6f8d90fd7e`, unmodified.
- Owner: ms-swift CUDA/JTorch coverage. Review when checkpoint serialization, GRPO prompt preparation, optimizer behavior or CUDA kernels change.

## L3 checkpoint continuation

The public `swift.cli.main rlhf` path used local Qwen2-0.5B, FP32 eager attention, SGD and strict Jittor torch shim on RTX 4090 workers. The full-tuning GRPO L3 result is run key `20261015-qwen2-grpo-full-l3-cursor-v2` (job 26612): same-Trainer and new-process resumes both continued checkpoint step 6 to step 8, with cursor/RNG checks and zero CUDA fallback.

This continuation removed the storage blocker for CISPO and SAPO PEFT LoRA. Save-peak preflight key `20261015-cispo-sapo-lora-l3-savepeak-v1` (job 26712) measured a maximum single-save peak of 66,060,288 bytes and required reserve of 2,279,604,224 bytes; worker preflight found about 30 GB available before the paired runs. The paired keys were:

| Run key | Worker job | Result |
| --- | ---: | --- |
| `20261015-qwen2-cispo-lora-l3-cursor-v1` | 26835; verifier 26846 | Native and shim same-process and new-process resumes reached step 8 from checkpoint 6. Ten audited updates had finite CUDA gradients; the cursor continued with the expected distinct prompts; 336 adapter tensors changed. Shim reported `cuda=1 fallback=0`. |
| `20261015-qwen2-sapo-lora-l3-cursor-v1` | 26837; verifier 26847 | Same L3 checks passed; 336 adapter tensors changed and shim reported `cuda=1 fallback=0`. |

Both use the existing GRPO LoRA checkpoint-1 as their initial adapter and preserve the per-loss run data and checkpoints. The verifier checks exact restoration for saved CPU, active CUDA, Python and NumPy states. The CUDA checkpoint contains the active device's RNG; the worker exposes eight devices, so comparison is against that saved active-device entry. The shim CPU RNG was not always perturbed by the probe, so perturbation is required for active CUDA, Python and NumPy while CPU state must still restore exactly.

## Full-tuning GRPO L4/L5

Run key `20261015-qwen2-grpo-full-l4l5-v3-promptcap` used the saved full-tuning checkpoint at step 6 and ran the native and strict shim public CLI to step 20 (worker job 26838). The prompts were bounded so measured model-input length was 496 tokens. Four warmup updates were excluded; ten synchronized CUDA timing samples were measured per side.

| Measure | Native PyTorch | Jittor shim |
| --- | ---: | ---: |
| Mean measured step | 0.2921 s | 0.4739 s |
| Median / nearest-rank p95 | 0.2918 / 0.4427 s | 0.4693 / 0.7518 s |
| Sampled process peak (NVML) | 7,782 MiB | 18,280 MiB |
| Max model input | 496 tokens | 496 tokens |

The symmetric mean-step ratio was 1.622× (limit 3×); maximum absolute loss difference over steps 11–20 was 0.00021443 (limit 0.005). Shim runtime was `cuda=1 fallback=0`. These measurements establish the specified single-card profile only; they are not a multi-card or broad throughput claim. The initial JIT build for the post-merge source cache took several minutes, and is excluded from steady-state timings.

## Failure localization

The first L4 retry (job 26824) failed before training because the harness looked for checkpoint-6 under the new run directory instead of the previously validated L3 directory. The corrected run passed explicit per-side checkpoint paths. A later verification attempt (job 26838) completed both 20-step training legs, but its log parser treated the final summary row (which has `global_step/max_steps` but no `loss`) as a step record. Filtering to rows containing `loss` allowed the verifier to accept the existing artifacts without retraining.

CISPO/SAPO startup attempts 26822/26823 stopped at a harness SHA input check that named the wrong reward plugin file; corrected jobs 26835/26837 trained and passed. Their first verifier also made an incorrect all-visible-GPU RNG shape assumption and loaded Jittor optimizer pickle data as a Torch zip. The verifier now checks the saved active CUDA entry and recognizes both serialization formats. These were experiment-harness failures; no Jittor core/compat defect or ms-swift source change was found.

An earlier full-tuning performance attempt, key `20261015-qwen2-grpo-full-l4l5-v2`, reached step 9 before eager attention OOM because ms-swift re-encoded the untruncated original prompt for training despite `max_length=512` in inference preparation. This was external ms-swift sample re-encoding behavior; input preparation for v3 bounded the source prompts and the accepted run observed max length 496. The external checkout remains unmodified.

## Evidence and limits

Raw logs, per-step captures, checkpoint manifests, NVML samples and verifier outputs remain under `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/` in the three run-key directories above. No raw-gradient/update tensor was retained separately; required checkpoints, logs, manifests, summaries and shared JIT cache are preserved. FIPO/REAL and PPO retain their earlier verified upstream failure boundaries. No other completed run key was repeated.

After integrating the latest remote docs/manifest changes, worker job 26876 regenerated and checked `MANIFEST.in`, and `tools/check_repo_layout.sh` passed. Full `tests/structure` reported 1,505 passed, 8 skipped and 1 failed: `test_source_config_is_executable_without_nox_path_rewriting` could not launch because `asv` is absent from the worker venv (`No module named asv`). An earlier worker install attempt was refused by the configured package proxy. The other 1,057 subtests passed; this environment-dependent entry-point check remains unverified.
