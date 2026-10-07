# Qwen2-0.5B 公开全参数 SFT：新基线三步数值复验

- 状态：公开 CLI 三步数值、首次更新前逐参数裁剪后梯度及新进程一步续训对拍完成；严格 L0–L5 尚未逐级验收。
- 日期：2026-10-07。
- 基线：Jittor `24483d232`（包含上游 `ffeb7bd80`）、ms-swift `88d7279`、隔离 Python 3.11.15。
- 验证范围：真实缓存 Qwen2-0.5B、单 RTX 4090、公开 `python -m swift.cli.main sft`、全参数 FP32、SGD 无动量、固定数据、三步、每步 batch 4、最大长度 64、学习率 1e-5、无 AMP 与梯度累积。
- 维护者：ms-swift CUDA 适配。
- 复查条件：补齐逐 trainable 参数和输入梯度对拍、适用 optimizer 状态、新进程恢复，再按 L0–L4 分层验收；Torch shim、CUDA executor 或训练入口变更时重跑。

Slurm 作业 11014 在同一 GPU 上先原生 PyTorch、后严格 Jittor CUDA，均经公开 ms-swift CLI 完成三步并保存 checkpoint-3。候选父子进程启动和退出均证明 `use_cuda=1`、shim 标记真、`fallback_count=0`。独立比较脚本读取两侧真实 safetensors、trainer state 与原始模型：最终 290 个权重键相同，最大绝对差 `7.45058e-9`，整体 L2 差 `5.54662e-8`；两侧各有 246 个权重张量发生更新，更新 L2 分别为 `0.001200761539` 和 `0.001200761583`。第 3 步 loss 差 `1.66893e-6`、梯度范数相对差 `4.79480e-7`，token accuracy 相同。

| 层 | 本配置状态 | 缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型、tokenizer、数据集及训练器经公开入口构造；初始状态键、全部 dtype 与设备的逐项审计未运行。 |
| L1 | not-run | 尚无同权重同输入的独立前向张量与逐层误差对拍。 |
| L2 | blocked | 首次更新前 290 项裁剪后梯度已直接比较，三步 loss/最终权重接近；裁剪前及第 2、3 步梯度、逐步 optimizer 状态和更新轨迹未直接比较。输入为离散 token ID，不适用输入梯度。 |
| L3 | blocked | 新进程从 checkpoint-3 续训一步并对齐 global step 与最终权重；optimizer 保存状态完全同构，scheduler 核心步数与学习率一致，RNG 四类条目均可加载；同进程恢复、RNG 值及数据游标的直接审计未运行。 |
| L4 | blocked | 公开 CLI 两侧确已完成三步并保存 checkpoint；前级验收未全通过，不能升级为 L4 PASS。 |
| L5 | blocked | 前级未通过；未做真实尺寸的两次预热和十次同步稳态计时。 |

在同一代码基线的后续运行键 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-fullparam-grad-step1-ffeb/` 中，Slurm 11078 先完成独立原生 PyTorch 三步训练；11081 在同一 RTX 4090 上完成严格 shim 三步训练。两侧各在首次 `SGD.step` 前保存 290 个 CUDA 梯度张量（按 optimizer 参数组位置索引）；此时训练已执行最大范数 1.0 的梯度裁剪。11085 在 worker 上复核形状、dtype、CUDA 标记和两侧完整清单：最大绝对差 `3.52971e-7`，整体相对 L2 `5.50140e-6`；裁剪后梯度整体 L2 分别为 `1.00000002` 和 `1.00000067`。该次第 3 步 loss 差 `5.72205e-6`，最终权重 290 键最大差 `7.45058e-9`，两侧各 246 张量更新。候选父子进程开始、采样及退出的 `fallback_count` 均为 0，`use_cuda=1`。原始 `*.npy`、清单、脚本、两侧 checkpoint、`gradient-comparison.json`、`comparison3.json` 和 Slurm 日志均未版本化。

11078 候选在训练前因多进程通信的 UNIX socket 路径过长失败；11081 使用较短的隔离 `TMPDIR` 后训练完成，但作业壳层读取了旧 checkpoint 路径而返回 1，11085 已独立确认实际 checkpoint 与梯度产物。两侧数据预处理均打印 NFS 临时文件清理警告，未阻止训练。梯度只直接采集首次裁剪后一步；无动量 SGD 的 optimizer 状态为空，但逐步状态和更新未直接审计。RNG、通用数据游标、BF16、AdamW、LoRA、双卡与 L5 均未由这两次三步训练验证。因此严格矩阵中 L2 尚不完整，L3–L5 不升级为 PASS。

后续新进程恢复运行键为 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-fullparam-resume-ffeb/`。Slurm 11087 的原生 PyTorch CUDA 从原生 checkpoint-3 续训至 checkpoint-4；11088 的严格 shim CUDA 从候选 checkpoint-3 续训至 checkpoint-4，父子进程启动及退出均为 `use_cuda=1`、shim 标记真、fallback 0。11089 在 worker 上对拍：第 4 步 `global_step=4`，290 个模型权重键最大绝对差 `7.45058e-9`、整体差 L2 `6.49349e-8`；两侧各有 246 个权重张量继续更新，更新 L2 分别为 `0.000400253221` 与 `0.000400253236`；loss 差 `2.14577e-6`，token accuracy 一致。11087 的首次候选启动误用了梯度实验的 `sitecustomize`，未安装 shim 并在读候选 optimizer 文件时失败；该次结果作废，原始日志保留，11088 使用固定脚本独立重跑候选。

Slurm 11093 在各自运行时加载 checkpoint-4 的 `optimizer.pt`、`scheduler.pt` 与 `rng_state.pth`，候选加载时 fallback 0。两侧无动量 SGD 的 optimizer `state` 均为空，两个参数组的已保存字段和值完全相同；scheduler 的 `last_epoch=4`、`_step_count=5`、`base_lrs` 和 `_last_lr` 一致。scheduler 结构存在非核心字段差异：原生有 `verbose=false`，shim 有 `_is_initial=false`。RNG 两侧均保存并可加载 `python`、`numpy`、`cpu`、`cuda` 四类条目，但不同运行时的内部状态未做逐值等价断言；数据游标也未直接审计。同进程恢复未运行，因此 L3 仍 blocked。

未版本化证据：`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-fp32-ffeb/` 中的 `worker3.sh`、`run.sh`、两侧日志、checkpoint-3、`compare3.py`、`comparison3.json` 和候选启动/退出标记。
