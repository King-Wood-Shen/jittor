# Qwen2-0.5B PEFT LoRA：AdamW 三步公开 SFT 对拍

- 状态：固定配置 L0/L1/L4 通过；L2 数值轨迹通过，但 optimizer `step` 的公开 device 元数据有差异，故记 partial；L3 未运行；L5 blocked。仅覆盖此 PEFT LoRA 配置与 `adamw_torch`。
- 日期：2026-10-08。
- Jittor 基线：`02498fdd9d9cd9bd698aff7b7a53ae056371a30a`，已合入上游 `origin/2.0-refactor` 的 `84d60a6d63fc4185fd7746bd1c65d23f06507542`。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch AdamW optimizer state 的 `step` 张量 device 报告路径或 Jittor placement 实现变化时，复验公开 CLI、checkpoint 和逐步状态。

运行键 `20261008-peft-lora-adamw-l2-ckptinit-v2` 的 Slurm 12826 在 cscg-qh17 RTX 4090（UUID `GPU-2fd350e6-8fcd-9385-4e5e-46405831cfa1`）先原生后严格 shim，通过公开 `swift sft` 完成三步。两侧均以相同 Qwen2-0.5B FP32/eager base、相同 PEFT LoRA adapter、四条固定数据、batch 4、max length 64、seed 1234、无 shuffle 构造并保存 checkpoint。AdamW 参数均为 lr `1e-5`、betas `(0.9, 0.95)`、eps `1e-8`；weight decay 分组相同，为 `0.1` 与 `0.0`。原生 optimizer 名为公开注册项 `adamw_torch`。

完整 386 项模型参数初始值逐项 SHA256 相同，96 项 trainable 参数及输入均在 CUDA。首批 input IDs/labels/attention mask 完全相同，logits shape `[4,8,151936]`，最大绝对差 `1.33336e-4`、相对 L2 `2.51284e-6`；loss 最大差 `1.19209e-6`。三步 loss 原生为 `[3.82736802, 3.75735402, 3.68772626]`，shim 为 `[3.82736683, 3.75735426, 3.68772697]`。

每步 96 项梯度、更新及 288 项 optimizer 状态数组均有限。跨运行误差如下：

| 步 | 梯度 max abs / 最坏相对 L2 | 权重更新 max abs / 最坏相对 L2 | optimizer 状态 max abs / 最坏相对 L2 |
| --- | --- | --- | --- |
| 1 | `5.28991e-7` / `7.05224e-6` | `6.58355e-7` / `1.44220e-3` | `5.30854e-8` / `1.31130e-5` |
| 2 | `9.22475e-7` / `2.89433e-5` | `1.10938e-6` / `1.28346e-3` | `9.92732e-8` / `3.32557e-5` |
| 3 | `4.80562e-7` / `1.49084e-5` | `1.43499e-6` / `1.09443e-3` | `1.32452e-7` / `2.08161e-5` |

公开状态观察到一个设备元数据差异：PyTorch 的每参数 `state['step']` 在三步均为 CPU；shim 第一步报告 CPU，第二、三步的 `.device` 报告 `cuda:0`。独立 CUDA 小型 optimizer 探针 Slurm 12842 复现了同一现象：shim counter 的 `location()` 仍是 `cpu`，但 `.device` 为 `cuda:0`；参数与 `exp_avg`/`exp_avg_sq` 均为 CUDA，fallback 0。源码追踪到 `compat/torch/installers/tensor/method_api.py::_device` 优先读取 `placement_backend`，而不是先检查实际 host residency；这能解释公开属性的报告路径，但尚未确认 optimizer step scalar 的 placement 标记何时变成 CUDA，因此不把完整状态语义标为 PASS。

严格 shim 从构造、前向到每次 optimizer step 均 `use_cuda=1`、fallback 0。Slurm 12839/12841 在 GPU worker 上比较状态；12841 的 key 对齐版本为权威汇总。Slurm 12824 使用非法 CLI 名 `--optim adamw`，在模型构造前退出；按本 checkout 的参数注册名改用 `adamw_torch` 后，12826 完成训练。早期比较器 12832、12834、12835 的断言/排序假设错误，未触发模型重跑；所有产物与错误日志保留。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | PASS（固定配置） | 公开 CLI 构造真实模型、PEFT、数据、Trainer 与 AdamW；386 项参数初态相同且 CUDA，optimizer 超参数分组一致。 |
| L1 | PASS（固定配置） | 相同输入；logits/loss 结构、有限性和数值误差通过。 |
| L2 | partial | 三步 96 项梯度、更新与状态数组数值通过；optimizer `step` 的公开 device 元数据第 2、3 步与原生不同。 |
| L3 | not-run | 未验证同进程/新进程 optimizer、scheduler、RNG 与 dataloader 状态恢复。 |
| L4 | PASS（单卡公开 `swift sft`） | 两侧均完成三步并保存 checkpoint。 |
| L5 | blocked | L2 状态语义仍 partial，未运行性能协议。 |

原始日志、参数、前向、梯度、状态和比较文件保存在未版本化目录 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-peft-lora-adamw-l2-ckptinit-v2/`；无效 CLI 参数记录在 `...-v1/`，单参数 device 探针在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-peft-lora-adamw-step-device-probe-c1/`。该结果不能推广为全部 AdamW、Muon、PEFT 或 ms-swift 已兼容。

文档门禁 Slurm 12847 通过发布清单生成与 `--check`（203 active Markdown files）、仓库布局检查，以及 Torch 模式结果索引可达性定向测试（1 passed）。
