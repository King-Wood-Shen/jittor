# Qwen2-0.5B 公开全参数 SFT：新基线三步数值复验

- 状态：公开 CLI 三步端到端数值诊断完成；严格 L0–L5 尚未逐级验收。
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
| L2 | blocked | 三步 loss/权重接近，但逐参数梯度、适用输入梯度和每步 optimizer 状态未直接比较。 |
| L3 | blocked | 同进程和新进程恢复后的 RNG、游标、scheduler、global step 与后续轨迹未运行。 |
| L4 | blocked | 公开 CLI 两侧确已完成三步并保存 checkpoint；前级验收未全通过，不能升级为 L4 PASS。 |
| L5 | blocked | 前级未通过；未做真实尺寸的两次预热和十次同步稳态计时。 |

这证明该固定配置的公开训练入口可完成并产生贴近原生的三步权重轨迹，不能代替每个 trainable 参数及适用输入梯度的直接对拍。无动量 SGD 的 optimizer 状态为空；RNG、通用数据游标、新进程恢复、BF16、AdamW、LoRA、双卡与 L5 均未由本次验证。因此严格矩阵中 L2 尚不完整，L3–L5 不升级为 PASS；不能以训练完成或总梯度范数推断全面兼容。

未版本化证据：`$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261007-qwen2-public-causal-fullparam-fp32-ffeb/` 中的 `worker3.sh`、`run.sh`、两侧日志、checkpoint-3、`compare3.py`、`comparison3.json` 和候选启动/退出标记。
