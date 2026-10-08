# Qwen2-0.5B 公开全参数 SFT：当前基线三步逐参数梯度对拍

- 状态：三步逐参数梯度、loss 和最终权重数值接近；这不是完整 L0–L5 通过。
- 日期：2026-10-08。
- 基线：Jittor `4a7eae584bb1eb0f2ea371b8f883bb451780cdbf`（包含上游 `25700b208fe58680169312e214c9ee82f125a094`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 验证范围：Qwen2-0.5B、公开 `python -m swift.cli.main sft`、单 RTX 4090、FP32/eager、全参数训练、无动量 SGD、固定四条数据、batch 4、三步、最大长度 64、学习率 `1e-5`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：直接比较逐步参数更新和必要 optimizer 状态，补齐初始状态审计、L1 前向与 L3 恢复，再按 Skill 逐层验收；CUDA shim、训练入口或相关执行器发生变化时重跑。

Slurm 12618 在 cscg-qh17 的 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`，驱动 580.178.04）先运行独立原生 PyTorch oracle，再运行严格 Jittor CUDA shim；两侧使用同一公开 CLI、缓存模型、数据文件、种子与配置。Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1。模型缓存 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，固定数据 SHA256 为 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。作业成功保存两侧 checkpoint-3；候选父子进程起止记录 `use_cuda=1`，三个训练步均 `fallback_count=0`。

Slurm 12620 在同一 GPU worker 上比较每一步首次 `SGD.step` 前、经过 max-norm 1.0 裁剪后的全部 290 个参数梯度。两侧张量清单、形状和 FP32 dtype 一致，采集清单均标记 CUDA。各步最大绝对差分别为 `3.9721e-7`、`4.1723e-7`、`3.5763e-7`；相对 L2 分别为 `5.4307e-6`、`5.1494e-6`、`5.0175e-6`。三步最终 290 个权重键一致，最大绝对差 `7.4506e-9`、相对 L2 `6.7206e-11`。三步 loss 差分别为 `9.5e-7`、`1.43e-6`、`7.2e-7`，token accuracy 均一致。日志出现的数据预处理临时文件清理警告未阻止训练或 checkpoint 保存。

| 层 | 本配置状态 | 缺口 |
| --- | --- | --- |
| L0 | partial | 真实公开 CLI 构造并训练；未逐项保存并比较初始模型完整状态、dtype 和设备清单。 |
| L1 | not-run | 没有独立同权重/同输入的前向 logits、hidden-state 与 loss 张量比较。 |
| L2 | blocked | 诊断采集了全部 290 项裁剪后梯度三步，另有 loss 和最终权重比较；因 L1 未验收，严格层级 blocked。逐步更新轨迹和适用 optimizer 状态也未直接比较。离散 token ID 输入不适用连续输入梯度。 |
| L3 | blocked | 此运行未进行同进程或新进程恢复，也未检查 RNG 与数据游标。 |
| L4 | blocked | 公开 `sft` CLI 确实完成三步并保存 checkpoint，但前置层尚未通过。 |
| L5 | blocked | 前置层未通过；没有执行 Skill 要求的预热后至少 10 次同步稳态性能协议。训练过程的墙钟数字不作为性能比较。 |

失败的准备作业 12615、12616 因本地 AF_UNIX socket 路径过长而在训练前终止；12617 在训练前取消以修正产物命名。日志和目录均保留。成功训练与比较分别为 12618、12620。运行键、逐张量梯度、checkpoint、原始日志、脚本和比较 JSON 未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-fullparam-grad3-4a7e/`。原始比较文件不含性能验收；本报告不把历史基线或单配置结果推广为完整 ms-swift 兼容性。
