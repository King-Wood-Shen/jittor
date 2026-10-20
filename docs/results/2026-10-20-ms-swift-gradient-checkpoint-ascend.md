# ms-swift gradient checkpointing through JTorch on Ascend

- Status: Implemented and verified for the flat-output, bound-module checkpoint path; the tiny fixture showed no memory saving
- Date: 2026-10-20
- Baseline: `fix/msw-ascend-options`, synchronized `origin/2.0-refactor` at `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`
- Fix commits: `2edb94c72` (ACL RNG APIs), `6c40dbbe0` (ACL compile dependency), `b58efb851` (checkpoint recomputation)
- Scope: `torch.utils.checkpoint.checkpoint`, ACL RNG preservation, Llama SFT BF16/SDPA public-entry comparison
- Maintainer: Jittor maintainers
- Recheck when: checkpoint output support or module parameter registration changes, and before claiming a memory benefit on a larger model

## Finding

The public `torch.utils.checkpoint.checkpoint` route was a pass-through in the Jittor shim. The data installer also left a pre-imported checkpoint module untouched. Consequently, `--gradient_checkpointing true` did not recompute layers. The installer now replaces the public function even when the module already exists. The implementation runs the first forward without autograd, then recomputes under grad in backward, explicitly routes gradients to input and bound-module parameters, swaps detached parameter replicas into the module temporarily, and restores the active CPU/CUDA/ACL RNG state around recomputation.

The implementation supports a Var or a flat tuple/list of Vars and `None`; nested output structures raise `TypeError`. It is registered as `APPROXIMATE` to expose that limited contract.

## Evidence

The source was validated on Ascend NPU 0 through the public `swift.cli.main sft` path. Both cases used the same local tiny Llama fixture, training data, seed, accumulation, optimizer, BF16, requested `--attn_impl sdpa`, and six training steps; only `--gradient_checkpointing false/true` differed. Runtime reported `jittor-shim`, `use_acl=1`, and `LlamaDecoderLayer`. The audit counted decoder-layer calls, which doubled from 12 to 24 with checkpointing enabled. The run requested SDPA, but this audit did not trace which ACL SDPA kernel executed.

| Metric | Off | On |
| --- | ---: | ---: |
| Loss by step | 4.90625, 4.875, 4.8125, 4.765625, 4.796875, 4.875 | Same sequence |
| Final logged train loss | 4.83854167 | 4.83854167 |
| Decoder-layer forward calls | 12 | 24 |
| NPU peak used memory | 18,769,920 B | 24,646,656 B |
| NPU peak reserved memory | 22,020,096 B | 44,040,192 B |
| Backend fallback count | 0 (`backend_fallback=error`) | 0 (`backend_fallback=error`) |

Loss and gradient norms remained finite. The trajectory was not monotonic, but both modes matched step-for-step. The small model’s memory is dominated by fixed runtime/parameter costs: checkpointing increased measured peak memory by about 5.9 MB and doubled reserved memory in this run. This demonstrates recomputation, not a memory reduction. A larger activation-dominated model is needed to measure the intended memory benefit.

The focused shim regression compared forward values, input gradients, parameter gradients, and RNG state. It passed on CPU and on ACL with dropout active: `2 passed` in Slurm job 5774. The public SFT on/off comparison passed in Slurm job 5772. `bash tools/check_repo_layout.sh` passed. The generated root source manifest was updated for this report and matched the manifest generator.

The full `tests/structure` run in the shim image ended with `13 failed, 1499 passed, 8 skipped` (job 5777). Failures were from the missing `asv` dependency, eight ACL float64 batched-matmul cases, and four legacy pool structure checks. The first attempt enabled ACL globally and had two extra failures, including an unrelated process-mode child timeout during a cold compile; the CPU-mode rerun completed without that timeout and retained the 13 failures. The required core tier did not complete: its native CPU session failed the host-image assertion that neither CUDA nor ACL is compiled in, while this shim image has ACL support. The CPU core runner completed a cold native build and was stopped after that assertion; its Torch session did not finish. These repository-wide gate results do not pass and are distinct from the focused checkpoint/Swift evidence above.

Run scripts, raw audit records, and logs are retained outside the repository at:

`/home/xinshen/_state/ms-swift-ascend/runs/options-gradient-checkpoint-20261020-r1/`

Re-run the public comparison with the isolated environment and cache already encoded in `toggle.sbatch` in that directory. The script submits two independent SFT processes and writes `gc-off-<job>/events.json` and `gc-on-<job>/events.json`.

## Limits

- The test covers one bound decoder layer and one tiny Llama fixture; no larger model was available in this run.
- Nested outputs and non-tensor output structures are rejected rather than silently losing gradients.
- The NPU comparison confirms the `LlamaDecoderLayer` recomputation path and no fallback. The audit did not identify the exact SDPA kernel.
- The training fixture uses no effective dropout. ACL dropout RNG preservation was separately exercised by the focused regression.

## Priority reference

The run followed the committed checklist at `fix/msw-ascend-algorithms-sync`, commit `d6229b0eb1a9db0c5353af9fdfbb4836d0b2944f`, file `agent/skills/ms-swift-torch-compat/references/priority-optimizations.md`; no uncommitted file from that worktree was copied.
