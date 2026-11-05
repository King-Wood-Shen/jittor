# Custom Torch optimizer compatibility and GaLore integration

The pinned ms-swift GaLore AdamW subclasses `torch.optim.Optimizer` and calls `super().__init__(params, defaults)` with a mapping. At the Jittor `0a23c2410cf3571e15200466b4caf4019f90d8cf` pin, the compat frontend forwarded that mapping to the native numeric learning-rate initializer. Swift's subsequent `LambdaLR` construction failed with `dict * float` before training began.

The compat frontend now handles custom Optimizer defaults and state at the Torch API boundary while leaving native named optimizer initialization on its numeric learning-rate path. A real CUDA integration step also showed that Swift calls `Tensor.addcmul_` in its GaLore AdamW update; the shim had the out-of-place `addcmul` only. `Tensor.addcmul_` now uses the existing storage-preserving in-place owner.

## Verification

- `compat/tests/torch/test_optimizer_subclass_protocol.py`: 7 passed, 0 skipped on RTX 4090 / CUDA 12.2 / SM89.
- Frozen downstream node `tests/integration/test_galore_cuda.py::test_galore_optimizer_projects_and_updates_selected_weights`: 1 passed on `cscg-qh04`, Slurm 157345, 2:05 elapsed. The test source and assertions were unchanged.
- Two training steps completed with losses 0.0587035939 and 0.0573004968. For each of the 168 target parameters, gradients were finite and nonzero and weights changed. Rank-8 moment shapes were `(896, 8)`, `(8, 896)`, `(4864, 8)`, and `(8, 4864)`.
- Jittor remained on CUDA and backend fallback stayed at 0. `checkpoint-2` was produced; its 16 file hashes were independently verified against the result receipt.
- CUDA `numpy_code` SVD requires CuPy; the final attempt used an isolated `cupy-cuda12x==13.6.0` and `fastrlock==0.8.4` target. The pinned Jittor installation remained unchanged, and its existing CUDA cache was reused.

The complete original failure trace, intermediate attempts and final result receipt are preserved in the downstream workspace under `artifacts/cuda-adaptation-board/evidence/CUDA-06/test-acceptance/galore/`.

Scope is limited to this frozen, non-quantized FP32 Qwen2-0.5B recipe. Q-GaLore, other dtypes, devices and broader workloads were not tested.
