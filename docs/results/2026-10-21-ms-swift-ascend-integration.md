# Ascend ms-swift/JTorch integration

- **Status:** Combined-tree focused NPU regressions passed at `22cf5495f0f2b28f386d2870d63f0ca4cf245370`; packing SFT passed at `6d49bc7e57ca9f87256ef1285c470944b0341f0d`. Earlier branch evidence remains at its original source SHA.
- **Date:** 2026-10-21
- **Baseline:** `origin/2.0-refactor` at `b2fa61449531341caabf418e639fd41e5e4ff0c7`; final focused regression source: `22cf5495f0f2b28f386d2870d63f0ca4cf245370`.
- **Scope:** JTorch/ACL/`compat.torch` fixes and previously accepted real-NPU ms-swift results from `fix/msw-ascend-fp16-20261020` (`072c61d87`), `fix/msw-ascend-options` (`7e44552f0`), and `fix/msw-ascend-algorithm-followup` (`9a88e996c`). No CPU adaptation or timing/performance acceptance.
- **Maintainer:** Jittor maintainers.
- **Recheck when:** ACL RNG, SDPA/TND, dtype dispatch, ms-swift option or source integration changes.

## Existing NPU evidence carried forward

Evidence below remains tied to its recorded implementation/source SHA and is not silently reattributed to the combined tree:

- FP16 training: two-device BF16 two-step training and inference passed on its recorded source; see [FP16 report](2026-10-20-ms-swift-ascend-fp16.md).
- Options: SDPA/eager and AdamW RNG checkpoint save/resume passed; packing was blocked at that earlier SHA. Activation recomputation is excluded; see [options report](2026-10-20-ms-swift-ascend-options.md).
- Attention: BSND/TND graph tests and packing SFT passed on the attention branch's later graph implementation; see [attention report](2026-10-20-ms-swift-ascend-attention.md). This does not erase the earlier source-specific packing failure.
- Algorithms: reward modeling passed only for the constrained small local-pair case. GKD had zero loss/gradient and failed; PPO stopped before training because its allocated device was occupied. Neither is accepted; see [algorithm report](2026-10-19-ms-swift-ascend-algorithm-training.md).
- World-size two training and inference conclusions retain their separate scopes in the [world2](2026-10-21-ms-swift-ascend-world2.md) and [inference](2026-10-21-ms-swift-ascend-inference.md) reports.

## Combined-tree revalidation

At `c3ed566377fabf8342bb1938e43ecce4ee43020e`, initial focused NPU job 5939 found a real ACL RNG state packing defect: the wrapper treated an already captured native state string as a device index. The two RNG round-trip tests failed; the other four focused tests passed. The wrapper was corrected in commit `6d49bc7e5` and revalidated at the new source SHA.

At `6d49bc7e57ca9f87256ef1285c470944b0341f0d`, NPU job 5956 on `cscg-hw01` passed all six focused tests: ACL random-state round trip, CUDA-shaped ACL random-state round trip, TND packed sample boundaries, BSND forward/backward, RMSNorm dispatch, and BF16 single-token SDPA. The source is recorded in `regression-source-5956.txt`; output is `npu-regressions-retry-5956.out` in `$JITTOR_LAB_ROOT/_state/ms-swift-ascend-integration-20261021/`.

At the same source SHA, public `swift.cli.main sft` job 5958 completed 20 BF16 steps with `packing=true`, `padding_free=true`, `attn_impl=flash_attention_2`, `backend_fallback=error`, and fallback count 0. The audit observed TND `npu_fusion_attention` calls with cumulative sequence boundaries `[4, 8, ..., 128]`. All 20 losses and gradient norms were finite; loss changed from `4.6875` to `3.71875`. The run is recorded in `packing-source-5958.txt`, `packing-5958.log`, and `audit/packing-5958/events.json` under the same isolated run directory. An earlier run completed training but exited on an audit assertion that expected the direct functional SDPA wrapper; the audit was corrected to check the actual TND NPU facade and the rerun exited successfully. This scope establishes execution and training only, not performance or algorithm quality.

After the duplicate ACL RNG pack/unpack helpers were removed and `MANIFEST.in` regenerated, NPU job 5962 reran the same six focused tests at source `22cf5495f0f2b28f386d2870d63f0ca4cf245370`; all six passed on `cscg-hw01` in 6.09 s. `regression-source-5962.txt` and `npu-regressions-final-5962.out` hold the source and output.

## Repository checks

`bash tools/check_repo_layout.sh` passed. The targeted cleanup, packaging, and docs structure set passed 36/36 after removing the duplicate helpers and regenerating the manifest. A full `tests/structure` run before those two fixes reported 1,500 passed, 8 skipped, and 12 failed. Two failures were fixed and the affected 36-test set passed afterward. The remaining observed failures were one missing optional `asv` package in the shim, one child `pytest --collect-only` timeout after 900 seconds, and eight float64 ACL batched-matmul cases that requested a CPU fallback while this check was run under the ACL runtime. The full suite was not rerun after those two fixes; this report does not mark the full structure suite green.

Neither these runs nor the carried-forward results include CPU adaptation or timing/performance acceptance.
