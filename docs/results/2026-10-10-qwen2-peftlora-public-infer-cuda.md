# Qwen2-0.5B PEFT LoRA 公开 `swift infer` CUDA 对拍

- 状态：本固定配置的 L0、L1、L4 通过；L2、L3 不适用；L5 未运行。
- 日期：2026-10-10。
- Jittor 基线：集成 HEAD `39c53c359ef9c3288e6f1b57e786ffe6e65d94ec`；同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是 HEAD 祖先。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`（4.6.0.dev0）；Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1。
- 模型与 adapter：Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；PEFT LoRA checkpoint-4 的 `adapter_model.safetensors` SHA256 `74fff4bb1df120a319b122ff59f97faa6d5d1483623486965c76843d4b645cf6`。
- 范围：公开 `python -m swift.cli.main infer`、Transformers backend、PEFT LoRA、FP32/eager、单卡 RTX 4090、两条固定 prompt、batch 2、temperature 0、每条生成 8 token。只适用于此模型、adapter 与请求，不代表其他 ms-swift 模型、tuner、任务或服务面。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor/torch shim CUDA 执行器、Transformers generation、PEFT adapter 加载或 ms-swift infer CLI 改动时；更换模型、adapter、dtype、后端或扩大请求范围时另行验证。

原生 PyTorch CUDA oracle 先于 strict shim 运行。公开推理运行键为 `20261010-qwen2-peftlora-public-infer-v3`（Slurm 15792），logits 运行键为 `20261010-qwen2-peftlora-logits-l1-v2`（Slurm 15816），完整状态元数据审计为 `20261010-qwen2-peftlora-state-l0-v1`（Slurm 15819）。作业均在 `cscg-qh04` 的 NVIDIA RTX 4090 上运行，GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`。ms-swift 使用 `--load_args false` 避免 adapter 目录中的训练插件配置进入推理入口；两侧使用相同模型、adapter、prompt、batch 和生成配置。

L0 审计比较了完整 `state_dict` 键及参数/buffer 名、形状、dtype、device 和参数 `requires_grad`。原生与 shim 均有 386 个 FP32 参数、1 个 buffer、387 个 state key；两侧元数据逐项相同。全部参数位于 `cuda:0`。生成输入为 `[2,35]`，输出为 `[2,43]`，两者也都在 `cuda:0`。状态 JSON 原文件 SHA256 相同：`08d279fcdab7c4d46fa596d58601c203716894a1373369004fe041f8f7fcddb6`。

L1 的 `--logprobs true` 只要求 generation API 返回实际生成步骤 logits，采样仍为 greedy。两侧各保存 8 个完整 logits 张量，shape `[2,151936]`、FP32、CUDA，值均有限；逐步最大绝对误差范围为 `1.52588e-5` 至 `3.88622e-5`，相对 L2 范围为 `4.90482e-7` 至 `2.11718e-6`，8 步 argmax 全部一致。生成 token 序列 SHA256 为 `80f94edb5e18672052e3303620553cbdb0ed70776bda79726a43f82fa18820f1`。L0 审计中的原生与 shim JSONL 整体 SHA256 也相同：`c0025a416300b175069cef4fd89d94786308d31e7690db7251a3fde9d0076da8`。

strict shim 在模型加载和推理进程中使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；父子进程 fallback 计数均为 0。预热后 `jittor_utils.bootstrap --check` 通过，220 个 JIT `.so` 路径及 SHA 在 bootstrap 与正式推理前后保持不变。日志尾部有 datasets 清理 NFS 临时文件的 `Errno 16` 警告，但两边 CLI 均 exit 0，比较器通过且结果完整。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS | 公开 CLI 构造真实 Qwen2/PEFT LoRA；387 个 state key、参数与 buffer 元数据逐项一致，386 个参数全部 FP32 且在 CUDA。 |
| L1 | PASS | 同模型、adapter 与固定请求；8 步完整生成 logits 有限、设备一致、argmax 一致并在误差门槛内；token 序列及公开结果相同。 |
| L2 | not-applicable | 这是纯推理配置，没有反向或 optimizer 更新。 |
| L3 | not-applicable | 单次无状态 CLI 生成，不存在训练续接或持久服务 session。 |
| L4 | PASS | 原生和 strict shim 均端到端运行公开 `swift infer` 并保存结果；L0/L1 已通过。 |
| L5 | not-run | 未运行预热后至少 10 次同步稳态测量、原生比与显存口径审计。单次生成速率不作为性能结论。 |

原始脚本、日志、完整状态 JSON、logits、生成结果与 cache 指纹未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261010-qwen2-peftlora-public-infer-v3/`、`20261010-qwen2-peftlora-logits-l1-v2/` 和 `20261010-qwen2-peftlora-state-l0-v1/`。历史严格训练对拍报告保持原结论；此报告只新增上述 PEFT LoRA 推理场景，不向其他 ms-swift 功能面外推。

文档门禁由 Slurm 15847 在该 HEAD 完成：`bash tools/check_repo_layout.sh` 通过；`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 1384 passed、8 skipped、1019 subtests passed。门禁禁用 MKL CPU 分支以避免构建本任务不涉及的 oneDNN；它不构成模型 CUDA 证据。
