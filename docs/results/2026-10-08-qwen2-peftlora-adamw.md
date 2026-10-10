# Qwen2-0.5B PEFT LoRA：AdamW 三步公开 SFT 对拍

- 状态：历史严格逐值结论保留；新协议下固定配置 L0–L4 通过，L5 blocked。L5 原生有 10 条稳态样本，但 strict shim 在第 3 步同步时触发 Jittor slice overflow，未达到 10 条有效样本。仅覆盖此 PEFT LoRA 配置与 `adamw_torch`。
- 日期：2026-10-08。
- Jittor 基线：`02498fdd9d9cd9bd698aff7b7a53ae056371a30a`，已合入上游 `origin/2.0-refactor` 的 `84d60a6d63fc4185fd7746bd1c65d23f06507542`。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch AdamW optimizer state 的 `step` 张量 device 报告路径或 Jittor placement 实现变化时，复验公开 CLI、checkpoint 和逐步状态。

## 2026-10-10：固定窗口收敛与同进程恢复复验

以下是按新训练验收协议完成的独立复验，不改写上方 2026-10-08 的严格逐值结论。历史 L2 仍为 partial：当时要求 optimizer `step` 的 device 元数据与原生相同；本协议把跨实现逐值误差留作诊断，要求两侧分别证明固定训练窗口收敛、完整梯度与有效状态，并要求恢复语义正确。

原生 oracle 使用运行键 `20261010-peftlora-adamw-l3-same-process-v1`（job15563 的原生轨迹），严格 shim 使用 `20261010-peftlora-adamw-l3-same-process-shim-v2`（job15577）；公开 CLI 无构建预检为 job15576。两侧沿用同一 Qwen2-0.5B FP32/eager、q/v LoRA、四条固定样本、batch 2、`adamw_torch` 与四步配置。运行前数据及初始 adapter 摘要已固定；GPU 为 qh04 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）。strict shim 事件的 `use_cuda=1`、`fallback_count=0`；训练钩子断言输入与全部 96 个可训练参数及梯度位于 CUDA，梯度存在且有限，AdamW 的 96 组状态每步有限。

每条轨迹以首两步和末两步的平均 loss 为固定窗口：原生 continuous 从 `3.98078406` 降至 `3.85156130`，shim continuous 从 `3.98078299` 降至 `3.85156107`；原生同进程恢复轨迹从 `3.98078406` 降至 `3.85156130`，shim 同进程恢复轨迹从 `3.98078453` 降至 `3.85155821`。四步均有 96 个 optimizer state，step 从 1 到 4；96 个 adapter 张量均不同于初始值，最大绝对更新约 `4.0031e-5`。连续训练与 checkpoint-1 恢复后的四个有序 batch（`input_ids`、`labels`、`attention_mask`）逐数组相同；恢复发生在同一训练进程，加载 96 个 AdamW state，scheduler `last_epoch=1`，Python、NumPy、CPU Torch、CUDA RNG 均匹配 checkpoint。

随后以运行键 `20261010-peftlora-adamw-l3-newprocess-v1` 对 native 与 shim 分别启动两个公开 CLI 进程：第一个进程保存 step-1 checkpoint，第二个新进程从该 checkpoint 续训到 step 4。Slurm job15725 通过 strict shim `JITTOR_NO_BUILD=1` bootstrap 检查；续训恢复了 96 个 AdamW state、scheduler `last_epoch=1` 与四类 RNG，native/shim 均由新 PID 完成恢复。恢复后的 step2–4 输入与之前连续轨迹逐数组一致，四步 loss 窗口均下降，96 个 adapter 张量实际更新，shim runtime `use_cuda=1`、fallback 0，Jittor `.so` 前后指纹不变。只读 GPU worker job15729 的数值审计通过；原始审计 JSON 有一个硬编码 job ID 字段误写为15726，状态目录的勘误记录了实际生成 job15729，不影响所审计的训练数据。

| 层 | 新协议状态 | 证据或边界 |
| --- | --- | --- |
| L0 | PASS（固定配置） | 公开 `swift sft` CLI 构造模型、LoRA、固定数据与 AdamW；strict shim CUDA 标记有效且零 fallback。 |
| L1 | PASS（训练 loss 路径） | 固定 CUDA 输入、形状与有限 loss；训练按该 loss 完成反向。不以跨实现 logits/hidden 误差作为训练门槛；未保存逐层 logits。 |
| L2 | PASS（固定四步窗口） | 全部 96 个可训练梯度逐项由训练钩子断言存在、CUDA 且有限；AdamW state 每步完整有限；adapter 确实更新，原生和 shim 各自末段窗口 loss 低于初段窗口。跨实现梯度/更新差异不作通过门槛。 |
| L3 | PASS（同进程与新进程恢复） | checkpoint-1 分别由同 PID 与新 CLI 进程恢复；optimizer、scheduler 与四类 RNG 有效，恢复后的有序 batch 与连续轨迹相同。 |
| L4 | PASS（单卡公开 CLI） | 两侧通过公开 `swift sft` CLI 完成连续及同进程恢复轨迹并保存 checkpoint。 |
| L5 | blocked（strict shim 运行失败） | 原生阶段采集到 10 条稳态样本；strict shim 第 3 步同步时触发 `nano_vector.h:41: slice overflow`，未生成有效性能 JSON，故无 shim 延迟、吞吐或原生比。 |

### 2026-10-10：AdamW 稳态性能阶段

新运行键 `20261010-peftlora-adamw-l5-v3` 使用同一公开 `swift sft`、Qwen2-0.5B FP32/eager、q/v PEFT LoRA rank 8、固定四条训练样本、batch 2 和 `adamw_torch`，原生 job 15759 先完成，strict shim job 15760 后运行。两阶段均在 cscg-qh04 RTX 4090（UUID `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`）；ms-swift SHA `88d727951203256baa564c643c651b6f8d90fd7e`，Jittor HEAD `54aef30b2e05a07f6bba9ad08aeb4e618ea9863a`。原始日志与 JSON 保存在未版本化的 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261010-peftlora-adamw-l5-v3/`。

原生公开 CLI 完成 12/12 步。同步计时覆盖 Trainer `training_step` 与 `AdamW.step`；首两步作预热，后十步均值 72.154851 ms、中位数 72.170864 ms、nearest-rank p95 73.074367 ms，对应该计时口径 13.8591 steps/s。原生 Torch allocator 峰值 allocated 2,193,025,536 bytes、reserved 2,220,883,968 bytes。首批三种张量均为 `[2,34]`，96 个可训练参数驻留 `cuda:0`。

strict shim 的 launcher 和训练进程均记录 `use_cuda=1`、`fallback_count=0`，并已实际进入公开训练；训练到第 3/12 步时，性能钩子在 `torch.cuda.synchronize()` 中调用兼容层 `jt.sync_all(True)`，触发 `nano_vector.h:41: slice overflow: 93941780761789 0 1`。仅两条有效训练计时，shim 未生成完成标记或性能 JSON，作业 15760 以 exit 1 结束。因此不能给出 shim 稳态性能或 native/shim 比值，L5 保持 blocked。该 slice overflow 与已记录的 Qwen2 全参数训练故障属于同一 Jittor 内部不变量问题；其根因调查已达五轮上限，本结果不启动第六轮，也不以本次未复现/短轨迹推翻历史故障。此处仅记录失败边界，不推断其他 LoRA 配置。

该新结论仅覆盖以上固定 Qwen2-0.5B PEFT LoRA/AdamW 配置，不能推广到其他 optimizer、trainer、模型或整个 ms-swift。same-process 只读审计运行键为 `20261010-peftlora-adamw-l3-shim-audit-v3` / job15710；new-process 训练为 `20261010-peftlora-adamw-l3-newprocess-v1` / job15725，独立审计 job15729。结构化审计结果与两侧训练原始产物保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/` 下相应运行目录。

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
