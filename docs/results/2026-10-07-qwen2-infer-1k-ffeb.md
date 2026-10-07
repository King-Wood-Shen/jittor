# Qwen2-0.5B 公开 infer CLI：新基线千 token 非流式复验

- 状态：限定配置的公开 infer CLI 输出逐字段一致；按逐级门槛 L4/L5 blocked。
- 日期：2026-10-07。
- 基线：Jittor `af2cbfa3c`（父提交包含上游 `ffeb7bd80`），ms-swift `88d7279`，隔离 Python 3.11.15。
- 验证范围：真实缓存 Qwen2-0.5B、公开 `python -m swift.cli.main infer`、单 RTX 4090 CUDA、FP32、eager attention、非流式 greedy、最多 32 新 token、两条提示。
- 维护者：ms-swift CUDA 适配。
- 复查条件：改动 Qwen2 加载、tokenizer/template、Torch shim、CUDA executor 或公开推理入口。

Slurm 作业 10968 在同一 GPU 上依次跑原生 PyTorch 与启用严格 CUDA/禁止 fallback 的 Jittor shim，之后用独立审计脚本复核产物；作业 10991 在 NVIDIA worker 另证原生环境为 PyTorch 2.6.0+cu124、CUDA 可用且无 shim 标记。两条真实 tokenizer 提示长分别为 1377、1286 token；公开入口统计合计 2701 prompt tokens，实际生成 49 tokens。两侧保存的完整 JSONL 两行逐字段相等，响应非空且回填到 messages。两侧日志都显示模型映射到 `cuda:0`；候选父子两个进程的启动与正常退出标记均为 `use_cuda=1`、shim 标记真、`fallback=0`。这证明该固定输入和配置的公开入口完成，但不替代前级逐张量验收。

| 层 | 状态 | 现有证据与缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型与公开入口构造并驻留 CUDA；初始状态键、逐项 dtype/device 未直接审计。 |
| L1 | partial | greedy 输出逐字段一致；同权重同输入的 logits/hidden、结构和误差未直接比较。 |
| L2 | not-applicable | 此配置仅推理，不涉及反向与优化器更新。 |
| L3 | not-run | 未审计同进程及新进程的推理状态恢复。 |
| L4 | blocked | 两侧公开 CLI 已完成；L0/L1/L3 尚未全部验收。 |
| L5 | blocked | 前级未通过；没有合格的同步稳态性能对比。 |

本作业包括模型加载和首次 JIT，候选日志中的约 615 秒不构成 L5 稳态性能数据。更长上下文、stream、批量变化、训练与恢复均不由这项结果升级；约 1.3k-token 的 Python 流式完整性能协议仍受 `KI-EXEC-011` 阻断。原始命令、节点/GPU UUID、日志、两份 JSONL 及审计结果位于未版本化的 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-infer-long-context-ffeb/`。
