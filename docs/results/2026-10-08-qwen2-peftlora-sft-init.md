# Qwen2-0.5B PEFT LoRA 公开 SFT：初始化随机状态未对齐

- 状态：此固定配置 L0 partial；L1-L3 blocked；L4 已执行但按前级门槛 blocked；L5 blocked。
- 日期：2026-10-08。
- Jittor 基线：`8964952c3c4124628fb9f2fe5991afb12a09d275`（包含上游 `origin/2.0-refactor` 的 `25700b208fe58680169312e214c9ee82f125a094`）。
- ms-swift checkout：`88d727951203256baa564c643c651b6f8d90fd7e`。
- 维护者：ms-swift CUDA 适配。
- 复查条件：Torch CUDA RNG/初始化语义修正，或构造相同初始 adapter 的确定性对拍方案可用时；之后重做同权重前向与三步轨迹。

运行键为 `20261008-qwen2-peftlora-sft-l0l2-8ac1`。Slurm 12634 在 cscg-qh17 RTX 4090（UUID `GPU-be5af850-cef8-4838-1a48-cf8f4e5d7102`）上先原生后严格 shim，经当前 ms-swift 公开 `swift sft` CLI，对 Qwen2-0.5B FP32/eager 使用 PEFT LoRA（q_proj/v_proj、rank 8、alpha 32、dropout 0）、SGD、四条固定样本、batch 4、长度 64 训练三步。两侧均构造 PEFT `PeftModelForCausalLM`，494,573,440 个总参数、540,672 个可训练参数（96 个张量），并产生 adapter checkpoint。Shim 参数与每步梯度记录均在 CUDA；每步 96 个梯度非空，候选运行开始/结束及各步 fallback 均为 0。原生 loss 为 `[3.8273680210, 3.8268311024, 3.8262913227]`，shim 为 `[3.8273668289, 3.8268554211, 3.8263378143]`。

这些数值不能作为同起点训练对拍：公开 Trainer 捕获钩子未命中 Swift 覆盖的 `compute_loss`，因此没有保存 L1 logits/输入；参数初始值审计和后续窄探针发现 LoRA A 初值不同。Slurm 12645 对三步 96 项梯度及最终 adapter checkpoint 的诊断比较分别得到最大绝对差约 `0.1036`，梯度相对 L2 最大约 `3.63`，最终 adapter 权重最大绝对差 `0.06679`、相对 L2 `3.03`。因初始 adapter 状态不同，这些差值不构成梯度/更新语义回归的结论。

为区分 base model 随机消耗的影响，运行键 `20261008-qwen2-peftlora-initprobe-5e71` 与 `20261008-qwen2-peftlora-resetseed-39bd` 分别在 Slurm 12652/12655 和 12656/12661 对相同 Qwen2 base 在原生与 shim 进程独立构造 PEFT LoRA；第二组又在 `get_peft_model` 前重置 seed。两组相同的 96 个键中，LoRA A 初值仍不相同，最大绝对差 `0.066790726`、相对 L2 `1.43842`。这表明仅重置全局 seed 不足以建立相同 adapter 初态。

运行键 `20261008-peft-kaiming-init-0d4f` 的直接 CUDA primitive 探针在 Slurm 12666 使用相同 seed、同一 cscg-qh17 RTX 4090，对 `(8,896)` FP32 CUDA 张量分别调用原生与严格 shim `torch.nn.init.kaiming_uniform_(a=sqrt(5))`。两侧均有限且位于 `cuda:0`；shim `use_cuda=1`、`fallback_count=0`。Slurm 12668 在 worker 上比较，得到最大绝对差 `0.06606726`、相对 L2 `1.41119`，输出不相等。至此已把 PEFT 初始值差异缩小到匹配 seed 的 CUDA `kaiming_uniform_` primitive 行为；该探针尚未把差异进一步归因为 RNG 生成还是初始化映射，故不声称已定位到具体内部实现。

| 层 | 状态 | 证据或缺口 |
| --- | --- | --- |
| L0 | partial | 公开 CLI 成功构造并训练，参数数量、键/shape 可枚举且候选设备 CUDA；相同 seed 下 adapter 初值不同，不满足同状态对拍。 |
| L1 | blocked | 未取得相同输入和初始 adapter 下 logits；捕获钩子未命中。 |
| L2 | blocked | 虽采集三步 96 项梯度与 checkpoint，但初态不相同且 L1 未过，差异不作为验收。 |
| L3 | blocked | 保存 adapter 文件不等于恢复轨迹；未验证同进程/新进程恢复及后续状态。 |
| L4 | blocked | 两侧都实际执行公开 `swift sft` CLI；因前置 L0-L3 未过，不升级形式等级。 |
| L5 | blocked | 前级未通过，未运行稳态性能协议。 |

脚本、日志、梯度、checkpoint、元数据及比较文件保留在 `$JITTOR_LAB_ROOT/_state/ms-swift-cuda/20261008-qwen2-peftlora-sft-l0l2-8ac1/`、`20261008-peftlora-initprobe-5e71/`、`20261008-peftlora-resetseed-39bd/` 与 `20261008-peft-kaiming-init-0d4f/`。该结果只限 Qwen2-0.5B 此 LoRA 配置，不代表 PEFT LoRA 或 ms-swift tuner 功能面整体兼容。
