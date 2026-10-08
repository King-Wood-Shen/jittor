# Qwen2-0.5B 序列分类公开 infer CLI CUDA 对拍

- 状态：固定 checkpoint 与 4 条输入的公开推理已在真实 CUDA 上完成；L0、L1 partial，L2 不适用，L3、L4、L5 blocked。
- 日期：2026-10-08。
- 基线：Jittor `f8647ac05883ab64624ce9aa2969353c15a844d5`，同步的 `origin/2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B `Qwen2ForSequenceClassification`，公开 `python -m swift.cli.main infer` / Transformers engine，单卡 FP32 eager，4 条固定 JSONL 输入，batch size 2。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补充完整模型状态映射、恢复语义与重复稳态性能协议；模型、Transformers、Swift CLI 或 Torch shim 改动时重验。

Slurm 13074 在 `cscg-qh04` 的 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）上依次执行原生 PyTorch CUDA 和严格 Torch shim。两侧复用 Slurm 13054/13067 生成的同一原生公开训练 checkpoint，`model.safetensors` SHA256 为 `5a7b969434ab0ab56d79153fc7ab344a3911b301f8f0b7b847a46c71de2abc7b`；输入 SHA256 为 `fb3fc07a4ba45bda63b28a9fb064e171a12e59af08b13f34e4f80763aa0c8532`。候选在真实 `swift/cli/infer.py` 子进程内启用 `jt.runtime.scope(use_cuda=1, backend_fallback='error')` 与 `forbid_backend_fallbacks()`。

两侧均构造 291 个模型参数，设备清单均为 `cuda:0`；第一批 `input_ids`、`attention_mask` shape 均为 `[2,12]`，dtype 均为 int64，设备均为 `cuda:0`。第一批 logits shape `[2,2]`、FP32、CUDA，最大绝对差 `3.8147e-6`、相对 L2 `3.2184e-7`，两侧值均有限。4 条公开结果中的预测类别与输入标签字段逐行一致；top 类 logprob 对齐，非 top 类 logprob 最大差约 `4.77e-6`。候选 `use_cuda=1` 且 fallback 计数为 0。

| 层 | 本配置状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 checkpoint、Qwen2 序列分类模型、tokenizer/template、Transformers engine 构造完成；参数计数与 CUDA 设备已核对，完整参数名、状态键和值映射尚缺。 |
| L1 | partial | 同一输入首批 logits 与全部公开预测类别已对齐；尚未记录全数据集所有输入 ID 与完整输出张量清单。 |
| L2 | not-applicable | 这是推理面。 |
| L3 | blocked | 未测试模型/engine 恢复与新进程续接。 |
| L4 | blocked | 公开 infer CLI 已实际运行；按门槛，L0/L1 尚为 partial，故本层不标通过。 |
| L5 | blocked | 前置层未通过；只有单次公开推理，不构成预热后至少 10 次稳态协议。首次 Jittor/JIT 时间不作性能结论。 |

两次未计兼容结果的 harness 尝试也保留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/`：Slurm 13070 的空缓存首次导入在父进程持锁等待 `query_cuda_cc` 子进程处停滞，已在 `20261008-qwen2-seqcls-infer-v1/diagnosis.txt` 记录；Slurm 13073 虽生成输出，但监测钩子没有覆盖 CLI 启动的 `swift/cli/infer.py` 子进程，fallback 与设备证据缺失，job 因结果采集失败而结束。Slurm 13074 在复用已预热缓存并修正子进程监测后完整通过本配置。原始日志、结果 JSONL、运行 JSON、logits、脚本及缓存均未版本化，位于 `20261008-qwen2-seqcls-infer-v3/`。

结论只适用于该 Qwen2-0.5B 序列分类 checkpoint、4 条固定输入与单卡 Transformers infer CLI；不代表其他分类模型、embedding、reranker、reward 或整个 ms-swift。
