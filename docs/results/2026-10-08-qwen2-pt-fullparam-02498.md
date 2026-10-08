# Qwen2-0.5B 公开 `swift pt` 全参数 CUDA 三步对拍

- 状态：固定配置下公开入口训练与最终权重对拍完成；L0–L2 证据仍为 partial，L3 未运行，L4 仅确认公开 CLI 端到端完成，L5 blocked。不得推广到 `pt` 全部数据格式、模型或优化器。
- 日期：2026-10-08。
- Jittor 基线：`6b4ae6cf43e2379fa8408a1f13cd03f42ab1a770`；上游同步基线 `origin/2.0-refactor`：`84d60a6d63fc4185fd7746bd1c65d23f06507542`。
- ms-swift：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：`swift pt` 默认 tuner、模型 device placement、Trainer/callback hook 或 checkpoint 语义变化时复验。

Slurm 12874 在 cscg-qh17 RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）按原生 PyTorch、严格 shim 顺序运行公开 `python -m swift.cli.main pt`，相同 Qwen2-0.5B、四条固定文本 JSONL、FP32/eager、SGD `lr=1e-5`、batch 1、seed 1234、三步并保存 checkpoint-3。输入数据 SHA256 为 `a5dd096489a60b049db362aafdd555d7d34501bc42650f27e51ead552884fb87`，基础模型权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`。Jittor 版本 2.0.0，Python 3.11.15；使用隔离 PyTorch/Transformers 环境。

该公开入口在显式 `--tuner_type full` 下报告 494.0328M 参数全部可训练。原生与候选三步的 loss、grad norm、token accuracy 数值逐位相同。Slurm 12882/12889 在 GPU worker 流式逐 tensor 比较 checkpoint，290 个键均有限且原生/候选最终权重完全相同（最大绝对差与全局相对 L2 均为 0）；相对基础模型有 239 个张量改变。候选事件日志记录 `use_cuda=1`、`fallback_count=0`。比较器 12881 因同时物化大数组 OOM；12886 因 NumPy 不能直接转换 safetensors bfloat16 失败；修正为逐张量安全加载后，12889 完成比较。失败日志保留，没有重跑训练。

设备证据仍不足以将 L0 标为 PASS：Trainer 状态采集 hook 未触发，未得到全部参数的逐项 `.device` 清单；CLI 也打印了模型已位于多个设备的提示。严格 CUDA scope 与零 fallback 证明候选启用 CUDA 路径，但不能替代逐参数驻留证明。日志虽逐步报告相同 loss/梯度范数，未采集相同 tokenizer 后的 input IDs、logits、全量参数梯度或 optimizer 状态。因此这些观察不能升级为完整前向或反向合同。

另有 Slurm 12858 使用 `pt` 默认 tuner（LoRA，4.3991M trainable）完成三步，最终 adapter SHA 相同且 loss 逐位一致；它只用于确认默认 tuner 行为，不与下方显式全参数结果混为一项。该运行也未得到逐参数设备清单。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 `pt` 构造真实 tokenizer/model/trainer/SGD，494.0328M trainable；候选 `use_cuda=1`、fallback 0，但没有逐参数 device 清单。 |
| L1 | partial | 三步 loss 与 token accuracy 逐位相同；缺逐项输入 IDs、logits/hidden 比较。 |
| L2 | partial | 290 个最终权重键逐位相同，239 项相对初始模型发生变化；缺全量梯度、每步更新与 optimizer 状态比较。 |
| L3 | not-run | 未检验同进程或新进程恢复后的状态与下一步轨迹。 |
| L4 | partial | 原生与 shim 均通过公开 `swift pt` 三步训练并写出 checkpoint；前级契约证据仍不完整。 |
| L5 | blocked | 前序层未达到完整验收，未运行性能协议。 |

原始数据、脚本、checkpoint、比较 JSON 和日志均未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-pt-fullparam-cuda-three-step-02498-v1/`；默认 tuner 诊断在 `.../20261008-qwen2-pt-cuda-three-step-02498-v1/`。本报告仅覆盖该固定配置。
