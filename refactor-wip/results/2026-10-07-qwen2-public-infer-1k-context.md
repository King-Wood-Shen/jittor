# Qwen2-0.5B 公开 infer CLI 千 token 输入与缓存解码

判定：固定真实基座、纯 FP32/eager、两条约 1.3k-token 提示的非流式 greedy 配置，原生 PyTorch 与严格 Jittor CUDA 的 `python -m swift.cli.main infer` 官方 JSONL 逐字段相等；限定该输入长度与最多 32 新 token 的公开入口 L4 PASS。它不能外推到模型最大上下文、任意长度或 L5 稳态性能。

- 基线：Jittor integration `d80c2ffb4d13adbcdab517a8d9e2e18c9dd3bb66`、ms-swift `88d7279`，隔离 Python 3.11.15；同一真实缓存 Qwen2-0.5B FP32/eager，离线模板 qwen、batch 上限 2。10866 在 cscg-qh09 RTX4090 GPU-335d6370-259b-bb0c-b471-930c0b4516a8（driver580.178.04）顺序运行 native/shim，COMPLETED0。
- 输入在 Slurm NVIDIA worker 上用真实 tokenizer 生成并记录为 1377、1286 tokens；CLI 实际报告两条合计 2701 prompt tokens（含模板）。配置 `max_new_tokens=32`、temperature0；两条因停止条件共生成 49 tokens。两侧加载日志均为 `device_map=cuda:0`。
- 候选公开 CLI 父子进程从导入起由 bootstrap 进入 `use_cuda=1`、`backend_fallback=error`、`forbid_backend_fallbacks()`，独立 JITTOR_HOME。10867 在 cscg-qh04 RTX4090 GPU-f54b1242-12ae-0c83-66b0-1c08461046ec 独立审计，COMPLETED0；两条完整 JSONL 逐字段相同、响应非空，两个不同 PID 起止 shim marker 真、CUDA1、fallback初值和增量0。审计输出 `QWEN2_LONG_CONTEXT_CLI_PASS [1377, 1286] processes 2 fallback=0`。
- 原始 worker、输入生成器、提示、结果、日志、审计和 Slurm 记录位于 `_state/ms-swift-cuda/20261007-qwen2-public-infer-long-context`。候选生成日志的558.95秒含首次 JIT、缓存构建，不能与原生1.03秒直接作为 L5 比较。逐层 logits、token ID、超过 1.4k 的输入、流式、并发、内存峰值与稳态性能 not-run；历史跳过项不变。原始脏工作树 30 修改加 2 未跟踪保持原状，未改产品源码。
