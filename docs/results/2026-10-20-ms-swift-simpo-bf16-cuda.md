# ms-swift SimPO BF16 training on CUDA

**状态：** 公开 BF16 CUDA 训练路径已执行；loss/gradient finite 且整体下降，shim 无 fallback；没有宣称逐步精确对拍。
**日期：** 2026-10-20
**基线：** `origin/2.0-refactor` / `d0fd6f577b6b025f0e2eecc08962f89a3ff7bb64`
**维护者：** Jittor 维护者（本次验证执行）
**范围：** 单卡 RTX 4090，Qwen2-0.5B，ms-swift `swift rlhf --rlhf_type simpo`，BF16，eager attention，固定单样本，三步更新。
**复查条件：** BF16 JTorch 优化器状态和 logits/gradient 定向对照补齐后，再决定是否设定更严格的数值容差。

公开入口两侧均显式设置 `--torch_dtype bfloat16 --bf16 true --fp16 false`；ms-swift 的保存参数与模型信息均报告 `torch_dtype=torch.bfloat16`。同一 Slurm GPU 分配内先运行原生 PyTorch，再运行 JTorch shim。

| 路径 | 三步 loss | 梯度范数 | NVML 峰值 |
| --- | --- | --- | ---: |
| 原生 PyTorch CUDA | `1.90223730 → 1.90662348 → 1.79914165` | `572.0, 572.0, 536.0` | 3438 MiB |
| JTorch CUDA shim | `1.90237844 → 1.95917380 → 1.82132137` | `570.5242, 568.7207, 516.8339` | 10719 MiB |

两侧 loss 与梯度范数均 finite，三步末 loss 低于首步。Shim sentinel 记录 `cuda=1`、`fallback=0`。最大逐步 loss 差为 `0.05255`（相对原生 `2.76%`），最大梯度范数相对差为 `3.58%`。中间一步的 loss 不是单调趋势。训练子进程均以 0 退出；Slurm 作业 110307 最终以 1 退出，是 harness 将任意原始 loss 差 `0.05` 作为硬门槛所致。该门槛不属于本轮 BF16/RL 矩阵标准，因此保留为“非精确对拍”，不把低精度轨迹宣称为逐步一致，也不因这一阈值错误丢弃真实设备运行证据。

完整日志、参数、NVML 100 ms 采样和阈值审计保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261020-simpo-bf16-cuda/logs/`。Jittor worktree HEAD 为 `a108d2b21f9fdb1d9847fa6770c1af22adb9b83b`，候选 dirty diff SHA256 为 `d1f201fa1181232610b7c2e4abe7ce5ff419cded5366d659834a78b620921ed8`；ms-swift commit 为 `88d727951203256baa564c643c651b6f8d90fd7e`。
