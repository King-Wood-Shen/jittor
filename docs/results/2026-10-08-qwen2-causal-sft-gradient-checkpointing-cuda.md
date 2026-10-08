# Qwen2-0.5B 公开 SFT gradient checkpointing CUDA 复验

- 状态：公开 CLI 的三步前向/反向和参数更新数值接近；Torch checkpoint shim 直接执行函数，不重算激活、不节省显存，因此 checkpointing 语义不兼容，不能判为该功能完整通过。
- 日期：2026-10-08。
- 基线：Jittor `6c59d654a58d7cc1de1f4a11d9925df224955884`；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`；Python 3.11.15；PyTorch 2.6.0+cu124；Transformers 4.57.6。
- 验证范围：真实 Qwen2-0.5B、公开 `python -m swift.cli.main sft`、单 RTX 4090、全参数 FP32/eager、`--gradient_checkpointing true`、固定四条数据、batch 4、三步、最大长度 64、无动量 SGD。
- 维护者：ms-swift CUDA 适配。
- 复查条件：实现 Torch checkpoint 的激活重算及显存语义后，复跑同配置三步逐参数梯度、更新、恢复，并按 Skill 补齐 L0–L5。

Slurm 13485 在 cscg-qh17 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`，驱动 580.178.04）先运行独立原生 PyTorch oracle，再运行 strict CUDA shim。Jittor 与 ms-swift SHA、模型 SHA256 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`、数据 SHA256 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`、种子和 CLI 配置均记录在未版本化运行目录 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-sft-gc-r2/`。shim bootstrap 父子进程开始和退出均为 `use_cuda=1`、Torch shim 标记真、fallback 0。

两侧均以公开 CLI 完成三步并保存 checkpoint-3。比较脚本在 Slurm 13499 上读取完整 checkpoint：290 个参数键一致，两侧分别有 246 个张量更新；最大绝对权重差 `7.45058e-9`、相对 L2 `6.65587e-11`。三步 loss 最大差 `2.86102e-6`；梯度范数原生为 `[445.91583, 445.72321, 445.53070]`，shim 为 `[445.91608, 445.72278, 445.53067]`。训练日志报告原生峰值约 5.8 GiB，shim 约 11.5 GiB；该单次训练日志不是规范显存性能对比，不能用作 L5 结果。

shim 运行明确打印警告：`torch.utils.checkpoint.checkpoint` 在 Jittor 中直接运行被包函数，激活仍保留，`use_reentrant` 无效。源码 `python/jittor/compat/torch/installers/data.py::_checkpoint` 也声明当前是 approximate pass-through。故此结果只能证明该路径在此短固定训练中可完成数值接近的反向与更新，不能证明激活重算、梯度逐参数等价或显存节约。

| 层 | 本配置状态 | 证据与缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型、tokenizer、数据和训练器通过公开 CLI 构造；初始参数/状态键、完整 dtype 与逐参数设备清单未比较。 |
| L1 | not-run | 无同权重同输入 logits、hidden-state 和 loss 张量对拍。 |
| L2 | partial | 三步完成 backward、更新并保存；loss、梯度范数和末态权重接近。没有逐 trainable 参数梯度和 optimizer 状态比较；activation checkpoint 实际被降级为直通。 |
| L3 | not-run | 未检查同进程/新进程恢复、优化器、scheduler、RNG 与数据游标。 |
| L4 | blocked | 公开 CLI 已端到端完成三步；因 L0–L3 未通过，按层级门槛不升级。 |
| L5 | blocked | 前层未通过且 checkpointing 内存合同缺失；首次 JIT 混入训练计时，未执行预热加 10 次稳态协议。 |

作废的准备作业 13484 在训练前因默认 `TMPDIR` 过长导致 AF_UNIX socket bind 失败；改用短临时目录后的 13485 完成。比较作业 13496/13498 因 NumPy safetensors reader 不支持 BF16 而失败；修正为 Slurm worker 上经 Torch reader 转 FP32 后，13499 比较成功。所有日志、checkpoint、脚本、缓存均保留在运行目录，不进入 Git。本结果是 Qwen2-0.5B 固定场景证据，不代表 ms-swift 其他任务、模型或 tuner 兼容。
