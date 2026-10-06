# Qwen2-0.5B IA3 adapter 公开 infer CLI 配对

判定：固定 FP32、短文本 greedy 的 `python -m swift.cli.main infer` 分发器与子进程，加载两侧各自历史 IA3 adapter，在真实 NVIDIA CUDA 上产生完全相同的两条公开 JSONL 响应；限定推理 CLI L4 入口合同 PASS。IA3 训练组件精确恢复的历史五轮失败保持 skipped；本结果不升级整个 IA3 tuner 的 L3/L4，也不证明通用 adapter 数值适配。

- 基线：Jittor integration `f3100f92bff0df84741019d2b6b5ec66231f186d`，ms-swift `88d7279`，隔离 Python 3.11.15。缓存 Qwen2-0.5B FP32/eager 权重；adapter 为 `_state/ms-swift-cuda/20261004-qwen2-ia3-export` 下各自导出的 native/shim safetensors，未重训。
- 协议：相同离线 JSONL 两条提示，Qwen 模板、batch 上限 2、greedy、每条 8 个新 token、非流式。10820 在 cscg-qh09 RTX4090 GPU-335d6370-259b-bb0c-b471-930c0b4516a8、driver 580.178.04 顺序运行原生与候选，COMPLETED 0。候选公开 CLI 父子进程由 sitecustomize 在导入开始进入 `use_cuda=1`、`backend_fallback=error` 与 `forbid_backend_fallbacks()`，独立 JITTOR_HOME；两侧日志均记录 `device_map=cuda:0`。
- 独立 10830 同一 NVIDIA worker COMPLETED 0：两条完整官方 JSONL（消息、标签、响应等字段）逐项相等，响应非空；两侧各生成 16 tokens。候选两个不同 PID 的起止 marker 真、CUDA 1、fallback 起始与增量均 0。审计输出 `QWEN2_IA3_PUBLIC_CLI_PASS 2 processes 2 fallback=0`。
- 原始 worker、bootstrap、结果、日志、审计与 Slurm 输出位于 `_state/ms-swift-cuda/20261007-qwen2-ia3-public-infer-cli`。候选 362.86 秒含首次 JIT，不是 L5 性能。未抓取 token ID、逐层 logits、adapter 参数位置或服务并发；stream、长上下文、训练与新进程完整状态恢复、L5 均 not-run/历史 blocked。原始脏工作树 30 修改加 2 未跟踪保持原样；未修改产品源码。
