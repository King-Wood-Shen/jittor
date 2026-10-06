# Qwen2-0.5B LoHa adapter 公开 infer CLI 配对

判定：固定 FP32、短文本 greedy 的 `python -m swift.cli.main infer` 分发器与子进程，加载两侧各自既有 LoHa adapter，在真实 NVIDIA CUDA 上产生完全相同的两条官方 JSONL；限定推理 CLI L4 入口合同 PASS。这不升级整个 LoHa tuner 的训练/恢复层级。

- 基线：Jittor integration `6ac55de312b2ea3eea05037b6841d8975d7f941b`，ms-swift `88d7279`，隔离 Python 3.11.15；缓存 Qwen2-0.5B FP32/eager 权重。adapter 分别为 `_state/ms-swift-cuda/20261004-qwen2-loha-export/native-adapter` 与 `shim-adapter` 的既有 safetensors，未重训。
- 协议：相同离线 JSONL 两条提示、Qwen 模板、batch 上限 2、greedy、每条 8 新 token、非流式。10836 在 cscg-qh09 RTX4090 GPU-335d6370-259b-bb0c-b471-930c0b4516a8（driver 580.178.04）顺序运行原生和候选，COMPLETED 0。候选公开 CLI 父子进程从导入开始由 bootstrap 进入 `use_cuda=1`、`backend_fallback=error`、`forbid_backend_fallbacks()`，使用独立 JITTOR_HOME；两侧加载日志均有 `device_map=cuda:0`。
- 依赖 10837 在 NVIDIA worker 独立审计，COMPLETED 0。两条完整官方 JSONL 逐字段相等，响应非空，两侧各生成 16 tokens；候选两个 PID 起止 marker 真、CUDA 1、fallback 初值和增量均 0。审计输出 `QWEN2_LOHA_PUBLIC_CLI_PASS 2 processes 2 fallback=0`。
- 原始 worker、bootstrap、结果、审计与 Slurm 日志位于 `_state/ms-swift-cuda/20261007-qwen2-loha-public-infer-cli`。候选 357.58 秒含首次 JIT，不作 L5 性能结论。token ID、逐层 logits、adapter 参数设备、stream、长上下文、训练与完整状态恢复未由此验证；这些面维持各自历史状态或 not-run。原始 30 修改加 2 未跟踪补丁未碰，未改产品源码。
