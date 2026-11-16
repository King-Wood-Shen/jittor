# Qwen2 loss_scale=all bounded CUDA regression

This record documents one strict public Swift plugin regression. It does not change the general CUDA support status, current-main runtime status, or the 151-task baseline.

- Parent/step: 188171.36; Slurm COMPLETED 0:0; one RTX 4090 GPU; 29.44 seconds.
- Request SHA-256: 2324a756763acd7b0c2152981bb9a2d14b0054c3dd026d6dd877e5a730f47989.
- Exact pytest node: `test_loss_scale_qwen2_cuda.py::test_public_loss_scale_all_qwen2_cuda_one_step`.
- Result: pytest/JUnit 1 passed, 0 failed, 0 skipped; exit 0.
- Configuration: Qwen2-0.5B, FP32, public Swift `loss_scale=all`, LoRA, one optimizer step.
- Runtime: Python 3.11.15; Jittor 2.0.2; jittor-compat 2.0.2; jittor-ms-swift 2.0.0.dev1; ms-swift 4.5.2; Transformers 4.56.2; datasets 3.6.0; pytest 7.4.4; PEFT 0.17.1. Installed core tree SHA-256 `9b50aad827c416db48307beeb380333c726ef7803529e960e6e2923ad5c64fcb`. Downstream checkout was `f5c8e713d59e63499fdac6d5bab0bb50813a7f00`, with local modifications to `compatibility.json` and `support/training-protocol.json`. This is a bounded regression on that installed combination, not a full current-main runtime acceptance.
- Public Swift source: commit `ff5128777dd3dc5538f1a95fbcc29442c327d7d2`; `examples/train/plugins/loss_scale.sh` SHA-256 `1cc1b9efa4785cbcf0331e77c99798993fff9e387e292a7afa49ef4ed75671a3`.
- Test SHA-256 `b48a511e3a6019e164e5454a4a7ac43e60c7199299565fe26fd4d55c1e6d3d8c`; callback SHA-256 `b88282af416301dbe9cdf9b298ecb8815d32d045beebb7365090f1064e6632b8`.
- Model identity: Qwen2-0.5B config SHA-256 `b2388df676bbbe4f6aae6b617bdfa6e7c471c7a5b23862baae3dfbf85a5fe9ef`; weights `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`; tokenizer `f7c9b2dba4a296b1aa76c16a34b8225c0c118978400d4bb66bff0902d702f5b8`. Fixed sample SHA-256 `9d08dc91615e0b52ce12442cc2ea6771a1ed38b3881391ab60bd419bda92ee1d`.
- Strict checks passed: actual CUDA:0; loss-scale strategy `all`; finite nonzero LoRA gradients and updates; optimizer restricted to LoRA; frozen base unchanged; one finite checkpoint at global step 1 with nonzero adapter; fallback count remained 0.
- Cache safety: the complete 1,748-file Jittor tree retained SHA-256 `83dd9726d3d6876bef45d39c2facfc78a3620d39c55ba78e693d8307ac04bf11`; the separate 24-file torch-shim tree retained SHA-256 `c4cff7c6a934d7e42b42617c9afa90f9f125971ebc49405285c59572b08da798`. Both were exactly unchanged before import, after import, and after pytest.
- Original receipt SHA-256: `250088c46c2ce1fdce1a1bc33ce62c55a90f43e7689bc0bb927e93b0b3d92d14`.
- Original pytest log SHA-256: `82d9fdc80834fe6c373539b54844c58da68f5eb35e064cfd9a0a500b66220efc`; JUnit SHA-256: `62c7103fd718325712cb4839ff999bba7153b5e4da03bc29a8d64dc5c4ab7226`; training trace SHA-256: `d3ad1e8acf3bfe4c66dbadfed8b906a8cbf2ff83d6300b914163e8130efd6498`; plugin trace SHA-256: `e5495c53fa24e279dca23a1b2853621cf0ed32fa6b6810f445e4a8e74534da0f`.

Uncovered: other model sizes, precisions, loss-scale modes, optimizer steps, distributed execution, and the full runtime on current main. Earlier R2 failure remains preserved; this result does not retrain or replace other configurations.
