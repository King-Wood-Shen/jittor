# Qwen2-0.5B 公开全参数 SFT：双卡 NCCL 三步新基线复验

- 状态：公开 `swift sft` 双卡三步及 checkpoint-3 对拍完成；L0/L1/L2/L4 仅有部分证据，完整 L0–L5 验收未完成。
- 日期：2026-10-08。
- 基线：Jittor `1ba2eafb827ab4a4bf36351956bf0611e512bb83`（上游 `2.0-refactor` 为 `7a18abf295668d9b19da5fa1657f5606e84b65a0`）；ms-swift `88d727951203256baa564c643c651b6f8d90fd7e`。
- 范围：Qwen2-0.5B、公开 `python -m swift.cli.main sft`、双 RTX 4090 NCCL、全参数 FP32、eager attention、SGD、每卡 batch 2、三步、固定数据、最大长度 64。原生对照为同一 worker、同一配置与已完成的 Slurm 12912；严格 shim 候选为 Slurm 12992。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Jittor NCCL/torch shim、Swift trainer/launcher 或 CUDA optimizer 改动；补齐逐参数梯度和 checkpoint 恢复后再评估层级。

模型权重 SHA256 为 `9cd8fc8c85a197b8c551d6b931b5709fe2611889d6b44945876472fecdf77cad`，训练 JSONL SHA256 为 `f38c72953cf933f85bf12abd50d56ed5d7fe1ec13d4a5952c1d2296fec5a65af`。环境为 Python 3.11.15、PyTorch 2.6.0+cu124、Transformers 4.57.6、PEFT 0.17.1、Accelerate 1.10.1，NCCL header 2.21.5。作业均在 `cscg-qh04`；两卡 UUID 为 `GPU-3c43713b-3ee9-956f-d6cd-95e55d2cdfea` 和 `GPU-98ae29e5-fa7c-45fd-34d1-fe31214339a4`。原生 oracle 先于 shim 运行。候选 rank0/rank1 的 Torch `RANK/WORLD_SIZE/LOCAL_RANK` 与 Jittor rank 均对应 `0/2/0`、`1/2/1`，两个 rank 均记录 NCCL 初始化成功、`use_cuda=1`、shim 开启、启动及退出 `fallback=0`。Torchrun 使用端口 32992，Jittor rendezvous 使用 52992。候选使用隔离 `JITTOR_HOME`，首次 JIT 串行；首次构建造成长等待，运行日志确认编译进程活动且随后完成。

候选公开 CLI 完成三步并保存 checkpoint-3、optimizer、scheduler、trainer state 和两个 rank RNG 文件。Slurm 13016 在 GPU worker 上逐键对比原生与候选 `model.safetensors`：290 键、494,032,768 个元素，键/shape/dtype 一致且均为有限值，最大绝对差 `7.45058e-9`，相对 L2 `6.69715e-11`。三步 loss 原生为 `3.82737017 / 3.82289529 / 3.81841826`，候选为 `3.82736969 / 3.82289505 / 3.81841850`，最大差 `4.77e-7`；梯度范数最大相对差约 `1.71e-6`，token accuracy 三步均为 `0.4375`。梯度范数不是逐参数梯度证据；optimizer/scheduler 文件虽已保存，尚未完成两侧状态逐项比较。

| 层 | 本配置状态 | 缺口 |
| --- | --- | --- |
| L0 | partial | 真实模型、tokenizer、dataset、trainer 与双 rank 构造完成；逐参数初态的 device/dtype 清单未单独导出。 |
| L1 | partial | 三步 loss 可对齐；同权重同输入的完整 logits/hidden-state 与 greedy 输出对拍未运行。 |
| L2 | partial | 三步 loss、梯度范数和最终全参数 checkpoint 对齐；未采集逐参数梯度、optimizer 状态和每步独立更新。离散 token ID 不适用输入梯度。 |
| L3 | blocked | checkpoint 及 rank RNG 已保存；同进程/新进程恢复、后续轨迹和 dataloader 游标未比较。 |
| L4 | partial | 已完成真实公开 `swift sft` 双卡 CLI；前级尚未通过，不能声称完整 L4。未验证纯 Jittor launcher、FSDP、DeepSpeed 或多机。 |
| L5 | blocked | 前级未通过；没有排除 JIT 后的十次稳态计时协议。训练墙钟不作性能结论。 |

原始脚本、环境、checkpoint、日志、比较 JSON 和 JIT 缓存未版本化，保存在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-causal-ddp-02498-7a18-v4/`；oracle 保存在同级 `20261008-qwen2-causal-ddp-02498-merge-fb73-v1/`。逐键比较脚本在 Slurm 13016 运行，exit 0。该结果只适用于上述单一 Qwen2 全参数 FP32/SGD 双卡配置，不能推广为 ms-swift 的双卡、NCCL 或整体 CUDA 兼容结论。
