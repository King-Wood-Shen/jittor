# ms-swift training options through JTorch on Ascend

- Status: SDPA/eager and AdamW RNG save/resume verified at the listed source SHA; packing/padding-free was blocked at that SHA. The later attention-graph result is recorded separately. Activation recomputation is not accepted for this integration and is excluded.
- Date: 2026-10-20
- Baseline: synchronized `origin/2.0-refactor` at `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`
- Implementation: `d64855b4778f858eab259ec73833414468eaeb39` (NPU RNG state API for Trainer checkpoints)
- Scope: public ms-swift SFT options on a tiny Llama fixture, BF16, with the Jittor ACL facade on physical Ascend NPU
- Maintainer: Jittor maintainers
- Recheck when: Transformers changes Trainer RNG serialization or ms-swift changes the attention/packing option contract

## Fixture and runtime identity

The public entry point was `swift.cli.main sft`, with a 2-layer Llama fixture (hidden size 64, intermediate size 128, 2 attention heads, vocabulary 128; 0.0986M parameters) and a fixed local training set. Every run used seed 42, BF16, `use_acl=1`, and `backend_fallback=error`. Slurm jobs ran in isolated task state, `JITTOR_HOME`, and caches. The selected options were tested as small, focused comparisons; FP16/AMP and algorithm changes were outside this scope.

| Input / runtime | Identity |
| --- | --- |
| Jittor source | implementation commit `d64855b4778f858eab259ec73833414468eaeb39`, based on `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64` |
| ms-swift | 4.5.2; installed source SHA `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`; METADATA SHA256 `444b38f77fd86559cc64db30e5135d9a4eccf7ecec41dc0480b5f4959fb085fd` |
| Transformers | 4.57.6; METADATA SHA256 `91601044a74fe02571a825356754dbbaa9e1d3e3dd3c2237254c15e903e421e3` |
| Torch / PEFT | Torch 2.11.0, METADATA SHA256 `ef79e614617651603dd49429ee9e185e1e41cd907138e32f6590464db56c8dd6`; PEFT 0.17.1, RECORD SHA256 `fd9c330897ad07959fa69653916ed8bfaafe30f4881fe754abc49359d4820236` |
| Attention shim / NPU package | flash-attn shim 2.7.4.post1, METADATA SHA256 `f52c020b3495b4038888dc11997e52ed9895a5e381601d57520e37be9fc55df4`; torch-npu manifest 2.10.0 |
| Model / data | model SHA256 `7893e4efd8a02226699179e75ebb1005c506e34d992ea2ec1ab03670b18b1429`; fixed train data SHA256 `bf7bc888ae77413126baa0377178e805b17f9ebf7396424f98b4397d28061d84` |

The installed `torch_npu` name resolves to the Jittor compatibility facade in this runtime. The facade explicitly refuses unimplemented vendor calls; it does not fall back to CPU.

## Attention backend: SDPA and eager

Jobs 5830 (SDPA) and its paired eager run used the same 20-step public SFT setup, AdamW (`--optim adamw_torch`), accumulation, fixture, and seed. The logged loss sequence was identical in both modes: `4.90625 -> 4.59375` over 20 steps. All 20 gradient norms were finite.

| Public option | Actual route | ACL SDPA calls | Gradient norm range | NPU peak used / reserved | Fallbacks |
| --- | --- | ---: | ---: | ---: | ---: |
| `--attn_impl sdpa` | Transformers Llama → `scaled_dot_product_attention` → Jittor ACL `acl_flash_attention_score_v2` | 80 | 4.789553–7.217896 | 24,646,656 / 44,040,192 B | 0 |
| `--attn_impl eager` | Transformers eager attention: matmul, softmax, matmul | 0 | 4.787621–7.225066 | 19,582,976 / 23,068,672 B | 0 |

This verifies both requested public routes and the ACL dispatch. The loss trace is non-monotonic at individual steps; the endpoint and overall trajectory matched. Memory is reported as device allocator peaks, not Trainer's host-side memory field. No timing or speed claim was evaluated.

## AdamW checkpoint save and resume

The public Trainer path initially failed at step 10 in `_save_rng_state`: `torch.npu.random` was absent. The NPU installer already exposed device and seeding APIs but not the RNG namespace Transformers 4.57.6 uses. Commit `d64855b` adds `torch.npu.random` state get/set and all-device variants backed by the existing ACL seed/counter snapshot helpers. A targeted real-NPU regression verified single-stream and all-stream round trips (`1 passed`).

The public SFT continuous-20-step run was compared with a 10-step run saved at step 10 and a fresh public SFT process resumed from `checkpoint-10` to step 20 (job 5843). Both used `--optim adamw_torch`, SDPA, and the same fixture and seed. The Trainer state contained the same 20 loss and gradient-norm entries step by step; every norm was finite. Final loss was `4.59375`. The final model file and RNG state were byte-identical between continuous and resumed runs.

| Run | NPU peak used / reserved | Fallbacks |
| --- | ---: | ---: |
| Continuous 20 steps | 24,646,656 / 45,088,768 B | 0 |
| Save at 10, resume to 20 | 24,646,656 / 45,088,768 B | 0 |

Call path: `swift.cli.main sft` → Swift SFT / Transformers Trainer → `_save_checkpoint` → `_save_rng_state` → `torch.npu.random.get_rng_state`; restore calls `_load_rng_state` → `set_rng_state`. Final `model.safetensors` SHA256 was `65b486de9feb53d54066fb2eb868fad880f25457c8d74165eb75048a6e2dd78e`; final `rng_state.pth` SHA256 was `d25d60bb512ae4b2647f6f4639759ab6e716efda728fcc1972586425b3257412`. `optimizer.pt` pickle files were not byte-identical; equivalence is established by the complete training history plus final model and RNG state.

## Packing / padding-free status at this source SHA

This result records the blocked path at implementation `d64855b` on baseline `d0fd6f577`. A later attention-graph result used a different implementation SHA and reported packing enabled; the two findings remain source-specific. The integrated path is being revalidated separately.

ms-swift documents `packing` as enabling `padding_free`, and its SFT arguments require FlashAttention for these options ([command-line parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md), [SftArguments](https://github.com/modelscope/ms-swift/blob/main/swift/arguments/sft_args.py)). A variable-length dataset was tried through public SFT. The initial `max_length=64` fixture was filtered out by dataset preparation; after correcting it to 128, the first model forward failed explicitly at `torch_npu.npu_fusion_attention` with `NotImplementedError: This NPU API is outside the Jittor ACL device facade`.

This is an implementation boundary, not an acceptable fallback path. Packing/padding-free on/off comparison could not be completed because the mandatory FlashAttention NPU fusion call is not implemented in this facade. No eager or CPU substitution was used. This route remains blocked pending a real ACL `npu_fusion_attention` implementation and NPU validation.

## Activation recomputation disposition

The activation-recomputation implementation is excluded from this integration because it did not pass acceptance. The earlier source-branch checkpointing report is not carried as proof and must not be used to promote that feature.

## Repository gate

The required structure suite ran in a no-NPU-GRES Slurm step and ended with `25 failed, 1487 passed, 8 skipped`. The failures were in repository-wide process-mode/import boundaries, ACL batched-matmul shape cases, legacy pool and CPU functional checks, and a native registry check. Several child-process failures show the shared compiler wrapper rejecting a missing private `CCACHE_DIR`; these failures are not evidence against the focused NPU RNG round-trip test, which passed on the real ACL device. The suite is not green and should be re-run with a correctly isolated compiler cache before using it as a repository health baseline. `tests/structure/test_docs_structure.py` passed all 15 checks, the root/compat/adapters manifest generator completed, and `bash tools/check_repo_layout.sh` passed after the result was recorded.

Raw run scripts, events, and logs are retained outside the repository at:

`$JITTOR_LAB_ROOT/_state/ms-swift-ascend/runs/options-attention-20261020-r1/`

## Scope limits

- The fixture is deliberately tiny; allocator peaks include fixed runtime and parameter costs and do not establish model-scale memory behavior.
- No timing acceptance was requested or performed.
- BF16 was held fixed across these comparisons. FP16/AMP belongs to the parallel workstream.
- Packing/padding-free remains blocked on the explicit fusion-attention API boundary above.
