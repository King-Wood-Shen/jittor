# TinyLlama 1.1B Chat 公开 `swift infer` CUDA 对拍

- 状态：本固定 profile 的 L0、L1、L4、L5 通过；L2/L3 不适用。
- 日期：2026-10-11。
- Jittor 基线：HEAD `20fdc796681101d732ed0a8158a3ae613f792119`；同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`，是 HEAD 祖先。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`（4.6.0.dev0）；Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、Accelerate 1.10.1。
- 模型：`AI-ModelScope/TinyLlama-1.1B-Chat-v1.0`，`model.safetensors` SHA256 `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`；config SHA256 `486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9`。
- 范围：ms-swift 通用 Llama loader，公开 `swift infer` Transformers backend，FP32/eager、单卡 RTX 4090、2 条固定消息、batch 2、temperature 0、最多 8 个新 token。显式用 `--template default --use_chat_template true --template_backend jinja` 消费 checkpoint 自带的 `<|user|>/<|assistant|>` tokenizer chat template。结论不外推到 Llama 其他变体、tuner、训练、服务或其他 ms-swift 面。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Transformers Llama/torch shim buffer 与生成语义、ms-swift `swift infer`/template 逻辑或 CUDA 执行器变化时；更换 checkpoint、dtype、后端或扩大请求范围时另行验证。

原生 oracle 先运行；strict shim 后运行。L0/L1/L4 完整功能运行键为 `20261011-tinyllama-public-infer-l01-v3`（Slurm 16202，cscg-qh04、NVIDIA RTX 4090，driver 580.178.04）。Python 3.11.15 使用 Jittor CUDA 12.2.140/SM89。模型权重、config 和固定输入均在两侧相同。v1 因超过 Slurm 分区时限被提交器拒绝，没有作业；v2/job16199 的 native CLI 产出两条响应，但审计过滤器漏掉实际运行模型的 `swift/cli/infer.py` 子进程，故没有审计产物且 shim 未运行。v3 修正过滤器后两侧 CLI 均完成；原始状态比较器先在 buffer `requires_grad` 附加断言处退出。Slurm 16203 只分析已保存产物、没有重跑模型，复核了契约字段和全部 logits。L5 使用独立运行键 `20261011-tinyllama-public-infer-l5-v1`（Slurm job 16215），复用同一 checkpoint、两条输入、dtype、模板、公开入口与已验证的 JIT 缓存；未复跑前述功能键。

L0 审计比较了 202 个参数/buffer 条目的键、shape、dtype、device 和有限性；所有条目为 FP32、有限且在 `cuda:0`，名称和其余必需元数据两侧相同。tokenizer chat-template SHA256 两侧均为 `66291cf0045c2425a3a667cf3cbb7af2b11f09e025c02f97245323ab79119362`。另观察到一项未作为本推理 profile 通过条件的状态差异：`model.rotary_emb.inv_freq` 是 Transformers LlamaRoPE 注册的 buffer，原生 `requires_grad=False`、shim `requires_grad=True`；它未进入参数集合，且本次前向 logits 与 token 对齐。此标志差异明确保留为后续 buffer/autograd 语义复查项，不据此声称训练状态合同已验收。

L1 使用 `--logprobs true` 捕获每个 greedy 生成步的完整分数。两侧实际输入和 mask SHA256 相同，输入 shape `[2,48]`；生成序列 shape `[2,56]` 且逐 token 完全相同。各有 8 个 `[2,32000]` FP32 CUDA logits 张量，均有限；逐步最大绝对误差 `1.10e-5` 至 `2.43e-5`，相对 L2 `6.43e-7` 至 `1.47e-6`，全部步骤 argmax agreement 为 1.0。两条公开结果的 `messages` 与 `response` 完全相同。strict shim 模型加载、推理父子进程均使用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 和 `forbid_backend_fallbacks()`；记录的 fallback 全为 0。bootstrap check 通过；warm 后与 no-build final 推理后的 126 个 `.so` 路径/SHA 清单相同。

| 层 | 本 profile 状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS | 公开 CLI 构造真实 TinyLlama/tokenizer/template；202 个条目的键、shape、dtype、device 一致，全部 FP32、有限、CUDA。单个 RoPE buffer 的 `requires_grad` 差异如上，不能由本项推断训练状态等价。 |
| L1 | PASS | 同一权重文件 SHA 与固定消息；实际 CUDA 输入/mask 和生成 IDs 一致，8 步完整词表 logits 通过有限值、数值和 argmax 检查。 |
| L2 | not-applicable | 纯推理 profile，没有反向或优化器更新。 |
| L3 | not-applicable | 无训练或持久 session，不适用续训恢复。 |
| L4 | PASS | 原生和 strict shim 均端到端运行公开 `swift infer` 并产出相同两条 response；满足 L0/L1。 |
| L5 | PASS | job 16215 在同一公开 CLI/profile 内每侧执行 2 次预热和 10 次 CUDA 同步生成；记录延迟、吞吐、CUDA allocator 峰值、设备驻留、相同序列与零 fallback，且 JIT `.so` 清单未变化。 |

## L5 稳态性能

job 16215 在 cscg-qh04 的 RTX 4090 上运行，Jittor worktree HEAD 为 `7b2f9d8722bb2fc7ae00cf076716c07d3a8c095c`（相对 L0/L1/L4 的 `20fdc796681101d732ed0a8158a3ae613f792119` 只有文档变化）。它使用相同的 ms-swift commit、FP32/eager、batch 2、固定两条请求和 greedy 8-token profile。公开 `swift infer` 内部每侧先预热 2 次，再对同一已编码 CUDA 输入同步计时 10 次；计时只包围 `generate` 并在前后同步 CUDA。预热耗时不计入稳态数字：native 两次分别约 308.77/142.36 ms，shim 约 8085.40/154.45 ms；shim 首次预热较长，但之后的 10 次测量稳定，且不计入报告的稳态延迟。每次测量均生成 8 个 token，十次序列 SHA 在同一侧及两侧间均完全相同。输入形状为 `[2,48]`，输入和 mask SHA 两侧相同。

| 指标 | PyTorch CUDA | strict JTorch CUDA |
| --- | ---: | ---: |
| 同步生成中位延迟 | 143.93 ms | 154.63 ms |
| 平均延迟 | 143.72 ms | 154.70 ms |
| P95 延迟（10 次样本的最大次序统计量） | 145.30 ms | 155.72 ms |
| 吞吐（batch 2 × 8 token / 中位延迟） | 111.16 token/s | 103.47 token/s |
| shim/native 中位延迟比 | — | 1.0743× |
| shim/native 吞吐比 | — | 0.9308× |
| CUDA allocator 峰值 allocated | 4,432,177,664 B | 4,417,430,016 B |
| CUDA allocator 峰值 reserved | 4,477,419,520 B | 6,365,904,896 B |

两侧模型的 201 个 FP32 参数及 1 个 buffer 都在 `cuda:0`；10 次测量的输出序列完全一致，父/子进程审计均为 fallback 0。shim 执行 `bootstrap --check` 且设置 `JITTOR_NO_BUILD=1`；预检前、预检后和测量后 126 个 `.so` 的路径/SHA 清单相同。allocated/reserved 为 PyTorch CUDA allocator 口径，不与进程级显存或其他 profiler 口径混用。性能结论只代表上述固定 profile。

原始脚本、完整参数/缓冲元数据、logits、JSONL、缓存清单与失败边界保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261011-tinyllama-public-infer-l01-v1/`、`-v2/`、`-v3/`、`20261011-tinyllama-public-infer-l01-analysis-v1/` 和 `20261011-tinyllama-public-infer-l5-v1/`。历史严格训练对拍结论不变；本结果只适用于上述 TinyLlama 固定推理 profile。

文档门禁 Slurm job 16204（cscg-qh04）通过：`bash tools/check_repo_layout.sh` 报告仓库布局与 236 个活动 Markdown 文档治理检查通过；`JITTOR_TORCH_SHIM=1 PYTHONPATH=python python -m pytest -q tests/structure` 为 1384 passed、8 skipped、1019 subtests passed。该门禁验证文档结构，不扩大本结果的模型或功能范围。

L5 补充报告的文档门禁 Slurm job 16216 通过：仓库布局检查及 237 个活动 Markdown 文档治理检查通过；Torch 模式 `tests/structure` 为 1384 passed、8 skipped、1019 subtests passed（180.54 s）。该门禁只验证报告索引与结构。
