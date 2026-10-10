# Qwen2-0.5B PEFT LoRA 公开 `swift infer` CUDA 对拍

- 状态：本固定配置的 L0、L1、L4、L5 通过；L2、L3 不适用。
- 日期：2026-10-10。
- Jittor 基线：集成 HEAD `e980d238600f7f1c0f381d38992a9ce777eb3aca`；同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是 HEAD 祖先。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`（4.6.0.dev0）；Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1。
- 模型与 adapter：Qwen2-0.5B `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`；PEFT LoRA checkpoint-4 的 `adapter_model.safetensors` SHA256 `74fff4bb1df120a319b122ff59f97faa6d5d1483623486965c76843d4b645cf6`。
- 范围：公开 `python -m swift.cli.main infer`、Transformers backend、PEFT LoRA、FP32/eager、单卡 RTX 4090、两条固定 prompt、batch 2、temperature 0、每条生成 8 token。只适用于此模型、adapter 与请求，不代表其他 ms-swift 模型、tuner、任务或服务面。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor/torch shim CUDA 执行器、Transformers generation、PEFT adapter 加载或 ms-swift infer CLI 改动时；更换模型、adapter、dtype、后端或扩大请求范围时另行验证。

原生 PyTorch CUDA oracle 先于 strict shim 运行。公开推理运行键为 `20261010-qwen2-peftlora-public-infer-v3`（Slurm 15792），logits 运行键为 `20261010-qwen2-peftlora-logits-l1-v2`（Slurm 15816），完整状态元数据审计为 `20261010-qwen2-peftlora-state-l0-v1`（Slurm 15819）。作业均在 `cscg-qh04` 的 NVIDIA RTX 4090 上运行，GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver `580.178.04`。ms-swift 使用 `--load_args false` 避免 adapter 目录中的训练插件配置进入推理入口；两侧使用相同模型、adapter、prompt、batch 和生成配置。

L0 审计比较了完整 `state_dict` 键及参数/buffer 名、形状、dtype、device 和参数 `requires_grad`。原生与 shim 均有 386 个 FP32 参数、1 个 buffer、387 个 state key；两侧元数据逐项相同。全部参数位于 `cuda:0`。生成输入为 `[2,35]`，输出为 `[2,43]`，两者也都在 `cuda:0`。状态 JSON 原文件 SHA256 相同：`08d279fcdab7c4d46fa596d58601c203716894a1373369004fe041f8f7fcddb6`。

L1 的 `--logprobs true` 只要求 generation API 返回实际生成步骤 logits，采样仍为 greedy。两侧各保存 8 个完整 logits 张量，shape `[2,151936]`、FP32、CUDA，值均有限；逐步最大绝对误差范围为 `1.52588e-5` 至 `3.88622e-5`，相对 L2 范围为 `4.90482e-7` 至 `2.11718e-6`，8 步 argmax 全部一致。生成 token 序列 SHA256 为 `80f94edb5e18672052e3303620553cbdb0ed70776bda79726a43f82fa18820f1`。L0 审计中的原生与 shim JSONL 整体 SHA256 也相同：`c0025a416300b175069cef4fd89d94786308d31e7690db7251a3fde9d0076da8`。

strict shim 在模型加载和推理进程中使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`；父子进程 fallback 计数均为 0。公开 CLI 性能运行键 `20261010-qwen2-peftlora-public-infer-l5-v6`（Slurm 15887）在相同 cscg-qh04 RTX 4090 上完成原生 oracle 后 strict shim。两边先各做 2 次同步预热，再对同一 `[2,35]` CUDA 输入进行 10 次完整 greedy 生成计时，每次两条请求共生成 16 token；10 次生成序列 SHA 均相同。中位延迟原生 `190.609 ms`、shim `198.377 ms`，shim/native 比值 `1.04075`（中位吞吐分别为 `83.94` 与 `80.65` token/s，比值 `0.96084`）。按 CUDA allocator 统计，peak allocated 为原生 `2,014,324,224`、shim `2,003,106,304` bytes；peak reserved 为原生 `2,067,791,872`、shim `3,077,570,560` bytes，reserved 反映各自缓存池，不等同物理进程显存。shim 首个预热为 `8.29 s`，不计入稳态计时；`jittor_utils.bootstrap --check` 通过，225 个 `.so` 的路径和 SHA 在 bootstrap 后、预热后、10 次测量后及 CLI 结束时一致，稳态期间无新库生成。原生与 shim 的公开结果 response、labels、messages、dataset 字段相同。两边 CUDA 输入/输出与 386 个参数均在 `cuda:0`，全部父/子事件 fallback 为 0。

L5 前的运行键 `20261010-qwen2-peftlora-public-infer-l5-v1`、`-v2`、`-v3`、`-v4`、`-v5` 分别因缓存清单陈旧、插件路径/旧结果复用、symlink 非幂等、首次预热增量算子库的比较口径、原生路径错误读取 `JITTOR_HOME` 等 harness 问题未满足验收；均未作为兼容失败或性能证据，原始证据留在各自 state 目录。v6 修正后 exit 0。`datasets` 清理 NFS 临时文件的 `Errno 16` 警告仍出现，但不影响 CLI 结果和性能采集。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS | 公开 CLI 构造真实 Qwen2/PEFT LoRA；387 个 state key、参数与 buffer 元数据逐项一致，386 个参数全部 FP32 且在 CUDA。 |
| L1 | PASS | 同模型、adapter 与固定请求；8 步完整生成 logits 有限、设备一致、argmax 一致并在误差门槛内；token 序列及公开结果相同。 |
| L2 | not-applicable | 这是纯推理配置，没有反向或 optimizer 更新。 |
| L3 | not-applicable | 单次无状态 CLI 生成，不存在训练续接或持久服务 session。 |
| L4 | PASS | 原生和 strict shim 均端到端运行公开 `swift infer` 并保存结果；L0/L1 已通过。 |
| L5 | PASS | 真实 Qwen2-0.5B + checkpoint-4 PEFT LoRA；原生与 strict shim 同卡，各 2 次预热、10 次同步测量，输入/输出和生成序列相同，fallback=0；报告中位延迟/吞吐、原生比和 CUDA allocator 峰值。仅适用上述固定请求 profile。 |

原始脚本、日志、完整状态 JSON、logits、生成结果与 cache 指纹未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261010-qwen2-peftlora-public-infer-v3/`、`20261010-qwen2-peftlora-logits-l1-v2/`、`20261010-qwen2-peftlora-state-l0-v1/` 和 `20261010-qwen2-peftlora-public-infer-l5-v6/`。历史严格训练对拍报告保持原结论；此报告只新增上述 PEFT LoRA 推理场景，不向其他 ms-swift 功能面外推。

L0–L4 报告初始门禁由 Slurm 15847 完成；本次 L5 报告更新在 Slurm 15890、Jittor HEAD `e980d238600f7f1c0f381d38992a9ce777eb3aca` 复验：`bash tools/check_repo_layout.sh` 通过；`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 1384 passed、8 skipped、1019 subtests passed。门禁禁用 MKL CPU 分支以避免构建本任务不涉及的 oneDNN；结构门禁不构成模型 CUDA 证据。
