# Qwen2-0.5B causal LM `TransformersEngine` Python API CUDA 对拍

- 状态：本报告固定配置的 L0、L1、L4、L5 通过；L2、L3 对纯推理不适用。只覆盖 Qwen2-0.5B FP32/eager、单卡、两条固定请求的 ms-swift `TransformersEngine` 非流式 Python API，不代表 causal LM 所有模型、批次/长度、CLI、服务或整个 ms-swift。
- 日期：2026-10-08。
- Jittor 基线：集成 HEAD `963ce6b3b1e6bad2cf21a54883374051b3e9b947`；`origin/2.0-refactor` `7a18abf295668d9b19da5fa1657f5606e84b65a0`，当前 HEAD 包含此上游基线。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 环境：Python 3.11.15、PyTorch 2.6.0、Transformers 4.57.6、PEFT 0.17.1、ms-swift 4.6.0.dev0；Qwen2-0.5B 本地 `model.safetensors` SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor/torch shim CUDA 执行器、TransformersEngine 或 ms-swift generation/template 改动时；扩大模型、长度、并发或服务面时另立结果。

功能对拍运行键 `20261008-qwen2-causal-transformersengine-api-v1`。Slurm 13204 在 cscg-qh06 RTX 4090（UUID `GPU-e0c8764d-4b51-62e5-f0d3-4aa27a8ddb68`，driver `580.178.04`）先运行独立原生 PyTorch，再运行严格 Jittor shim。两边都通过公开 `TransformersEngine(model_path, model_type='qwen2', task_type='causal_lm', template_type='qwen', attn_impl='eager', device_map='cuda:0')` 构造模型、tokenizer/template 和 engine；FP32，batch 上限 2。请求文本固定为 “Summarize the solar system in one sentence.” 与 “What is the capital of France?”；非流式 `RequestConfig(max_tokens=32, temperature=0, seed=1234, return_details=True)`。

两边参数元数据均为 290 个键，逐键名称、shape、dtype、device 完全一致；权重来自同一 SHA256 checkpoint，全部模型参数、prompt `input_ids` 和 `attention_mask` 均在 cuda:0。捕获的输入 shape `[2,29]` 且逐值相同。生成共 31 次模型 forward（首轮 logits `[2,29,151936]`，之后每次 `[2,1,151936]`）；31 组 logits 均有限，最大绝对差 `1.18256e-4`，最坏逐步相对 L2 `2.77669e-6`。两条返回文本、token IDs 和 finish reason 完全相同，生成 token 数分别 31 与 8。候选整个运行 `use_cuda=1`、`fallback_count=0`。

性能运行键 `20261008-qwen2-causal-transformersengine-perf-v1`，Slurm 13247 在同一 RTX 4090 使用前一运行已构建的 JIT cache；native 与 shim 各 3 次预热、10 次 batch-2 请求，测量前后均同步 CUDA。每次固定生成 39 个合计 token，且两边输出完全相同。中位延迟 native `672.99 ms`、shim `634.34 ms`，shim/native `0.943`；中位生成吞吐分别 `57.96` 与 `61.48 tokens/s`。`torch.cuda.max_memory_allocated(0)` 记录的 allocator high-water 分别为 native `1900.36 MiB`、shim `1889.71 MiB`；两框架 allocator 统计定义不同，不能当成物理进程显存的直接比较。shim fallback 为 0。

原始脚本、NPZ、JSON、逐步误差、10 次延迟和日志未版本化，分别位于 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-transformersengine-api-v1/` 与 `20261008-qwen2-causal-transformersengine-perf-v1/`。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS | 公开 causal 模型、tokenizer/template 与 `TransformersEngine` 构造；290 个参数键 metadata 一致且全在 CUDA。 |
| L1 | PASS | 同 checkpoint、相同 CUDA prompt 张量；所有生成步 logits 达容差、有限，token IDs 和公开文本输出相同。 |
| L2 | not-applicable | 推理 API，不执行反向或 optimizer 更新。 |
| L3 | not-applicable | 无训练/续训状态的无状态生成请求。 |
| L4 | PASS | 端到端通过 ms-swift 公共 Python API `InferRequest`、`RequestConfig`、`TransformersEngine.infer` 完成两条非流式生成。 |
| L5 | PASS（固定配置） | 两侧各 3 warmup + 10 同步稳态请求；报告延迟、吞吐、原生比、allocator 口径与零 fallback。仅限此 Qwen2 短文本 batch-2 配置。 |
