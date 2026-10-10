# TinyLlama 1.1B Chat 公开 `swift infer` CUDA 对拍

- 状态：本固定 profile 的 L0、L1、L4 通过；L2/L3 不适用；L5 未运行。
- 日期：2026-10-11。
- Jittor 基线：HEAD `20fdc796681101d732ed0a8158a3ae613f792119`；同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是 HEAD 祖先。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`（4.6.0.dev0）；Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、Accelerate 1.10.1。
- 模型：`AI-ModelScope/TinyLlama-1.1B-Chat-v1.0`，`model.safetensors` SHA256 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`；config SHA256 `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`。
- 范围：ms-swift 通用 Llama loader，公开 `swift infer` Transformers backend，FP32/eager、单卡 RTX 4090、2 条固定消息、batch 2、temperature 0、最多 8 个新 token。显式用 `--template default --use_chat_template true --template_backend jinja` 消费 checkpoint 自带的 `<|user|>/<|assistant|>` tokenizer chat template。结论不外推到 Llama 其他变体、tuner、训练、服务或其他 ms-swift 面。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Transformers Llama/torch shim buffer 与生成语义、ms-swift `swift infer`/template 逻辑或 CUDA 执行器变化时；更换 checkpoint、dtype、后端或扩大请求范围时另行验证。

原生 oracle 先运行；strict shim 后运行。完整功能运行键为 `20261011-tinyllama-public-infer-l01-v3`（Slurm 16202，cscg-qh04、NVIDIA RTX 4090，GPU UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`，driver 580.178.04）。Python 3.11.15 使用 Jittor CUDA 12.2.140/SM89。模型权重、config 和固定输入均在两侧相同。v1 因超过 Slurm 分区时限被提交器拒绝，没有作业；v2/job16199 的 native CLI 产出两条响应，但审计过滤器漏掉实际运行模型的 `swift/cli/infer.py` 子进程，故没有审计产物且 shim 未运行。v3 修正过滤器后两侧 CLI 均完成；原始状态比较器先在 buffer `requires_grad` 附加断言处退出。Slurm 16203 只分析已保存产物、没有重跑模型，复核了契约字段和全部 logits。

L0 审计比较了 202 个参数/buffer 条目的键、shape、dtype、device 和有限性；所有条目为 FP32、有限且在 `cuda:0`，名称和其余必需元数据两侧相同。tokenizer chat-template SHA256 两侧均为 `66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362`。另观察到一项未作为本推理 profile 通过条件的状态差异：`model.rotary_emb.inv_freq` 是 Transformers LlamaRoPE 注册的 buffer，原生 `requires_grad=False`、shim `requires_grad=True`；它未进入参数集合，且本次前向 logits 与 token 对齐。此标志差异明确保留为后续 buffer/autograd 语义复查项，不据此声称训练状态合同已验收。

L1 使用 `--logprobs true` 捕获每个 greedy 生成步的完整分数。两侧实际输入和 mask SHA256 相同，输入 shape `[2,48]`；生成序列 shape `[2,56]` 且逐 token 完全相同。各有 8 个 `[2,32000]` FP32 CUDA logits 张量，均有限；逐步最大绝对误差 `1.10e-5` 至 `2.43e-5`，相对 L2 `6.43e-7` 至 `1.47e-6`，全部步骤 argmax agreement 为 1.0。两条公开结果的 `messages` 与 `response` 完全相同。strict shim 模型加载、推理父子进程均使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 和 `forbid_backend_fallbacks()`；记录的 fallback 全为 0。bootstrap check 通过；warm 后与 no-build final 推理后的 126 个 `.so` 路径/SHA 清单相同。

| 层 | 本 profile 状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS | 公开 CLI 构造真实 TinyLlama/tokenizer/template；202 个条目的键、shape、dtype、device 一致，全部 FP32、有限、CUDA。单个 RoPE buffer 的 `requires_grad` 差异如上，不能由本项推断训练状态等价。 |
| L1 | PASS | 同一权重文件 SHA 与固定消息；实际 CUDA 输入/mask 和生成 IDs 一致，8 步完整词表 logits 通过有限值、数值和 argmax 检查。 |
| L2 | not-applicable | 纯推理 profile，没有反向或优化器更新。 |
| L3 | not-applicable | 无训练或持久 session，不适用续训恢复。 |
| L4 | PASS | 原生和 strict shim 均端到端运行公开 `swift infer` 并产出相同两条 response；满足 L0/L1。 |
| L5 | not-run | 未做至少 10 次同步稳态测量、原生比或统一口径显存采集；CLI 单次统计不作为性能证据。 |

原始脚本、完整参数/缓冲元数据、logits、JSONL、缓存清单与失败边界保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-tinyllama-public-infer-l01-v1/`、`-v2/`、`-v3/` 和 `20261011-tinyllama-public-infer-l01-analysis-v1/`。历史严格训练对拍结论不变；本结果只适用于上述 TinyLlama 固定推理 profile。

文档门禁 Slurm job 16204（cscg-qh04）通过：`bash tools/check_repo_layout.sh` 报告仓库布局与 236 个活动 Markdown 文档治理检查通过；`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 1384 passed、8 skipped、1019 subtests passed。该门禁验证文档结构，不扩大本结果的模型或功能范围。
