# Qwen2-0.5B PEFT LoRA 公开 SFT：初始化随机状态未对齐

- 状态：此固定配置 L0/L1 partial；补充的同状态 Transformers/PEFT 前向数值通过；L2-L3 blocked；公开 L4 已执行但按前级门槛 blocked；L5 blocked。
- 日期：2026-10-08。
- Jittor 基线：`8964952c3c4124628fb9f2fe5991afb12a09d275`（包含上游 `origin/2.0-refactor` 的 `25700b208fe58680169312e214c9ee82f125a094`）。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch CUDA RNG/初始化语义修正，或构造相同初始 adapter 的确定性对拍方案可用时；之后重做同权重前向与三步轨迹。

运行键为 `20261008-qwen2-peftlora-sft-l0l2-8ac1`。Slurm 12634 在 cscg-qh17 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`）上先原生后严格 shim，经当前 ms-swift 公开 `swift sft` CLI，对 Qwen2-0.5B FP32/eager 使用 PEFT LoRA（q_proj/v_proj、rank 8、alpha 32、dropout 0）、SGD、四条固定样本、batch 4、长度 64 训练三步。两侧均构造 PEFT `PeftModelForCausalLM`，494,573,440 个总参数、540,672 个可训练参数（96 个张量），并产生 adapter checkpoint。Shim 参数与每步梯度记录均在 CUDA；每步 96 个梯度非空，候选运行开始/结束及各步 fallback 均为 0。原生 loss 为 `[3.8273680210, 3.8268311024, 3.8262913227]`，shim 为 `[3.8273668289, 3.8268554211, 3.8263378143]`。

这些数值不能作为同起点训练对拍：公开 Trainer 捕获钩子未命中 Swift 覆盖的 `compute_loss`，因此公开 SFT 运行没有保存 L1 logits/输入；参数初始值审计和后续窄探针发现 LoRA A 初值不同。Slurm 12645 对三步 96 项梯度及最终 adapter checkpoint 的诊断比较分别得到最大绝对差约 `0.1036`，梯度相对 L2 最大约 `3.63`，最终 adapter 权重最大绝对差 `0.06679`、相对 L2 `3.03`。因初始 adapter 状态不同，这些差值不构成梯度/更新语义回归的结论。

为区分 base model 随机消耗的影响，运行键 `20261008-qwen2-peftlora-initprobe-5e71` 与 `20261008-qwen2-peftlora-resetseed-39bd` 分别在 Slurm 12652/12655 和 12656/12661 对相同 Qwen2 base 在原生与 shim 进程独立构造 PEFT LoRA；第二组又在 `get_peft_model` 前重置 seed。两组相同的 96 个键中，LoRA A 初值仍不相同，最大绝对差 `0.066790726`、相对 L2 `1.43842`。这表明仅重置全局 seed 不足以建立相同 adapter 初态。

运行键 `20261008-peft-kaiming-init-0d4f` 的直接 CUDA primitive 探针在 Slurm 12666 使用相同 seed、同一 cscg-qh17 RTX 4090，对 `(8,896)` FP32 CUDA 张量分别调用原生与严格 shim `torch.nn.init.kaiming_uniform_(a=sqrt(5))`。两侧均有限且位于 `cuda:0`；shim `use_cuda=1`、`fallback_count=0`。Slurm 12668 在 worker 上比较，得到最大绝对差 `0.06606726`、相对 L2 `1.41119`，输出不相等。至此已把 PEFT 初始值差异缩小到匹配 seed 的 CUDA `kaiming_uniform_` primitive 行为；该探针尚未把差异进一步归因为 RNG 生成还是初始化映射，故不声称已定位到具体内部实现。

为建立有效的 PEFT 前向 oracle，运行键 `20261008-peft-lora-l1-transplant-try2-9c1e` 在 Slurm 12690 的 cscg-qh04 RTX 4090 上，以相同 Qwen2-0.5B checkpoint、FP32/eager、PEFT LoRA q/v 配置构造两侧模型；原生生成 96 个 adapter 参数后，将其逐项转入 shim 模型并断言每个参数完全相等。输入 IDs/mask 由原生 tokenizer 生成并保存，两侧都在 CUDA 上执行；全部 386 个模型参数、输入、logits 和 25 组 hidden states 均为 CUDA 且有限。Slurm 12692 在 worker 上比较：logits shape `[2,7,151936]`，最大绝对差 `6.7234e-5`、相对 L2 `2.0526e-6`；hidden shape 均为 `[2,7,896]`，最大绝对差 `6.2561e-4`（h24），最大相对 L2 `2.7906e-6`（h23）。Shim `use_cuda=1`、`fallback_count=0`。该直接 Transformers/PEFT 固定输入前向满足数值门槛，但它没有通过 Swift Trainer 或公开 CLI，且未记录完整 386 项参数状态清单，故不能升级此面的层级 L0/L1 总状态。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 CLI 成功构造并训练，参数数量、键/shape 可枚举且候选设备 CUDA；相同 seed 下 adapter 初值不同，不满足同状态对拍。 |
| L1 | partial | Slurm 12690/12692 的直接 Transformers/PEFT 前向使用相同 checkpoint、CUDA 输入及逐项完全相同的 adapter；logits/25 组 hidden shapes、dtype、有限值与误差已比较。此路径未覆盖公开 CLI，且受 L0 状态约束。 |
| L2 | blocked | 虽采集三步 96 项梯度与 checkpoint，但初态不相同且 L1 未过，差异不作为验收。 |
| L3 | blocked | 保存 adapter 文件不等于恢复轨迹；未验证同进程/新进程恢复及后续状态。 |
| L4 | blocked | 两侧都实际执行公开 `swift sft` CLI；因前置 L0-L3 未过，不升级形式等级。 |
| L5 | blocked | 前级未通过，未运行稳态性能协议。 |

脚本、日志、梯度、checkpoint、元数据及比较文件保留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-peftlora-sft-l0l2-8ac1/`、`20261008-peftlora-initprobe-5e71/`、`20261008-peftlora-resetseed-39bd/` 与 `20261008-peft-kaiming-init-0d4f/`。该结果只限 Qwen2-0.5B 此 LoRA 配置，不代表 PEFT LoRA 或 ms-swift tuner 功能面整体兼容。
