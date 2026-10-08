# Qwen2-0.5B 序列分类 Python 推理 API CUDA 对拍

- 状态：本报告固定配置的 L0、L1、L4、L5 通过；L2、L3 不适用。结论只覆盖公开 `BaseArguments` + `InferRequest` + `TransformersEngine` 的 Qwen2 序列分类 batch 推理。
- 日期：2026-10-08。
- 基线：Jittor `d7be64656986db6554199440d9f7b037f090e9c1`，同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B `Qwen2ForSequenceClassification`，FP32/eager，单 RTX 4090；两条固定中文请求一次组成 batch 2。
- 维护者：ms-swift CUDA 适配。
- 复查条件：ms-swift Python 推理 API、Transformers、Torch shim 或 Jittor CUDA 执行器变更时复验；扩展到 embedding、reranker、其他模型和输入长度时另立证据。

Slurm 13097 先运行独立原生 PyTorch，再以严格 Torch shim 分别构造 `BaseArguments`、模型、processor、`seq_cls` template 与 `TransformersEngine`。两边加载同一公开训练 checkpoint，`model.safetensors` SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`。Python 为 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、Jittor 2.0.0、ms-swift 4.6.0.dev0。GPU 为 `cscg-qh04` RTX 4090，UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。

两侧 `state_dict` 均有 291 个键；逐键名称、shape、dtype、设备及张量值 SHA256 完全一致，所有参数和 state tensor 都在 `cuda:0`。engine 捕获的唯一 batch 中 `input_ids` 与 `attention_mask` shape 为 `[2,12]`、dtype 为 int64、device 为 `cuda:0`，两边输入数组逐值相等。FP32 logits shape `[2,2]`、device 为 `cuda:0`，最大绝对差 `3.8147e-6`、相对 L2 `3.2184e-7`，两侧均有限；公开 Python API 返回分类内容均为 `1`。严格候选从 import 到推理结束处于 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()` 内，fallback 计数为 0。

Slurm 13098 在同一 RTX 4090 上分别测原生与候选 API，每侧先预热 3 次，再进行 10 次每次前后 CUDA synchronize 的 batch-2 请求。中位延迟原生 `21.610 ms`、shim `6.883 ms`，shim/native 中位延迟比 `0.319`；中位吞吐分别为 `92.26` 与 `228.74` samples/s。单次 shim 测量含 `25.149 ms` 的首项离群值，故另列中位数与全部原始延迟。每侧测量前调用 `torch.cuda.reset_peak_memory_stats(0)`，报告随后 `torch.cuda.max_memory_allocated(0)` 高水位：原生 `1894.41 MiB`、shim `1886.39 MiB`。两种 runtime 的 allocator 语义不同；该值是各自框架统计口径，不代表同口径的进程总显存或可直接比较的物理峰值。

| 层 | 本配置状态 | 证据或边界 |
| --- | --- | --- |
| L0 | pass | 公开模型、processor、template、engine 构造；291 个 `state_dict` 键逐项名称、shape、dtype、CUDA 设备和值 hash 一致。 |
| L1 | pass | 完整 batch 输入 ID/attention mask 逐值一致；logits 结构、dtype、device、有限值及数值误差达标，公开类别输出一致。 |
| L2 | not-applicable | 推理 API，无反向、optimizer 或训练更新。 |
| L3 | not-applicable | 该无状态推理调用没有续训状态；原生与候选进程分别从同一持久 checkpoint 构造已由 L0 覆盖。 |
| L4 | pass | 通过 ms-swift 公共 Python API `BaseArguments`、`InferRequest`、`TransformersEngine.infer` 完成端到端 batch 推理。 |
| L5 | pass | 两侧均有 3 次预热和 10 次同步稳态样本，记录中位延迟、吞吐、allocator 高水位与原生比；候选 fallback 0。性能只代表该 checkpoint、短文本 batch 2 与这张 RTX 4090。 |

原始脚本、JSON、NPZ、日志和缓存未版本化：功能对拍在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-seqcls-python-api-v1/`，性能在 `20261008-qwen2-seqcls-python-api-perf-v1/`。另一独立场景 Qwen3-Embedding-0.6B 在 ModelScope 缓存里只有下载标记而无权重文件，本轮未运行；本结果不代表 embedding、reranker、reward、其他模型或整个 ms-swift。
